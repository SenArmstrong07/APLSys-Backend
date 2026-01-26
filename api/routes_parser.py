from fastapi import APIRouter, UploadFile, File, Query, Request, Depends
from pathlib import Path
from doctr.io import DocumentFile
import fitz
from services.parsing_service import (
    export_tables_to_csv,
    parse_document_text,
)
from utils.task_store import TaskStore
from model.request_schema import DocumentParseRequest, ResumeTextRequest
from typing import Optional
from services.ocr_service import create_doctr_ocr, dispose_doctr_ocr
from services.parsing_service import run_ner_in_subprocess
import asyncio
import gc
import requests
import json
from os import getenv
from dotenv import load_dotenv

router = APIRouter()
task_store = TaskStore()
load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash"
BASE_URL = "https://generativelanguage.googleapis.com/v1"
GEMINI_API_KEY = getenv("VITE_GEMINI_API_KEY")

# Reuse the doctr dependency from routes_ocr
async def get_doctr_dependency(request: Request):
    """
    FastAPI dependency that creates a doctr predictor for the lifetime of
    a parser request and disposes it after processing to save memory.
    """
    predictor = create_doctr_ocr()
    request.state.doctr_predictor = predictor
    try:
        yield predictor
    finally:
        try:
            dispose_doctr_ocr(predictor)
        finally:
            if hasattr(request.state, "doctr_predictor"):
                del request.state.doctr_predictor
                print("Disposed Doctr OCR predictor after parser request.")

@router.get("/tasks")
async def get_tasks(status: Optional[str] = Query(None), limit: int = Query(100)):
    """Get status of parser/AI tasks"""
    return {"tasks": task_store.list_tasks(status=status, limit=limit)}

@router.get("/tasks/{task_id}")
async def get_task(task_id: int):
    """Get specific task status"""
    try:
        return {"task": task_store.get_task(task_id)}
    except KeyError:
        return {"error": "Task not found"}

@router.post("/extract-resume-text")
async def extract_resume_txt(request: Request, file: UploadFile = File(...)):
    task_id = task_store.create_task(
        task_type="resume_extract",
        filename=file.filename
    )
    
    # We'll lazily create a Doctr predictor only if we need to OCR an image.
    ocr_model = None
    _temp_predictor = None
        
    temp_path = None
    try:
        task_store.update_task(task_id, status="processing")
        print("File received. Waiting for extraction...")
            
        # Read uploaded file bytes once
        file_bytes = await file.read()
        ext = Path(file.filename or "").suffix.lower()
    
        # Helper to save bytes to a temp file (used for docx / image fallbacks)
        def _save_temp(bts):
            p = f"temp_{file.filename}"
            with open(p, "wb") as tf:
                tf.write(bts)
            return p
    
            # 1) DOCX -> use parsing_service (assumed to handle docx)
        if ext == ".docx":
            temp_path = _save_temp(file_bytes)
            try:
                plain_text = parse_document_text(temp_path, ocr_model)
                if not plain_text or not plain_text.strip():
                    raise ValueError("DOCX extraction returned empty text")
            except Exception as e:
                print(f"DOCX extraction failed: {e}")
                raise RuntimeError(f"Failed to extract text from DOCX: {e}")

            # 2) Images -> use parsing_service (DocTR or image OCR path)
        elif ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            print("Image file detected — using image OCR path (DocTR via parsing_service).")
                # Lazily create predictor only for image path
            try:
                _temp_predictor = create_doctr_ocr()
                ocr_model = _temp_predictor
            except Exception as e:
                    # If predictor creation fails, surface helpful error
                raise RuntimeError(f"Failed to create DocTR predictor for image OCR: {e}")

            temp_path = _save_temp(file_bytes)
            plain_text = parse_document_text(temp_path, ocr_model)
    
            # 3) PDF -> fast path with PyMuPDF, fallback to parsing_service if it fails
        else:
            try:
                print("Trying PyMuPDF for PDF text extraction...")
                pdf = fitz.open(stream=file_bytes, filetype="pdf")
                pages_text = []
                for p in pdf:
                    pages_text.append(p.get_text("text"))
                plain_text = "\n".join(pages_text)
                print("PyMuPDF extraction successful.")
            except Exception:
                print("PyMuPDF failed — falling back to parsing_service (DocTR) if available.")
                temp_path = _save_temp(file_bytes)
                plain_text = parse_document_text(temp_path, ocr_model)
    
        task_store.update_task(
            task_id, 
            status="completed",
            details={"text_length": len(plain_text) if isinstance(plain_text, str) else 0}
        )
        return {"text": plain_text, "task_id": task_id}
    except Exception as e:
        task_store.update_task(task_id, status="error", details={"error": str(e)})
        raise
    finally:
            # Dispose any predictor we created for image OCR
        try:
            if _temp_predictor is not None:
                dispose_doctr_ocr(_temp_predictor)
        except Exception:
            pass
        if temp_path and Path(temp_path).exists():
            Path(temp_path).unlink()

async def extract_resume_sections_with_gemini(text: str) -> dict:
    """
    Use Gemini to extract education and work experience sections from resume text.
    Returns dict with sections or empty dict if extraction fails.
    """
    if not GEMINI_API_KEY:
        return {}
    
    prompt = (
        "Extract the following sections from this resume text. Return ONLY a JSON object with these exact keys:\n"
        "- education: list of education entries (school, degree, field, dates)\n"
        "- work_experience: list of work experiences (company, title, dates, description)\n\n"
        "If a section is not found or empty, use an empty list for that key.\n"
        "Return ONLY valid JSON, no markdown, no explanation.\n\n"
        f"Resume Text:\n{text}\n\n"
        "JSON:"
    )
    
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        
        result_text = getattr(response, "text", "")
        
        if not result_text:
            return {}
        
        # Try to extract JSON from response
        start = result_text.find('{')
        end = result_text.rfind('}') + 1
        
        if start == -1 or end == 0:
            return {}
        
        parsed = json.loads(result_text[start:end])
        
        # Validate that we have meaningful data
        education = parsed.get("education", [])
        work_experience = parsed.get("work_experience", [])
        
        # Only return if at least one section has content
        if education or work_experience:
            return {
                "education": education,
                "work_experience": work_experience,
                "source": "gemini"
            }
        
        return {}
    
    except Exception as e:
        print(f"Gemini section extraction failed: {str(e)}")
        return {}

@router.post("/ner-extract-resume-profile")
async def ner_extract_resume_profile(req: ResumeTextRequest, request: Request):
    """
    Use subprocess NER pipeline to avoid keeping large transformer models resident.
    Also attempts to extract structured sections (education and work_experience) via Gemini.
    """
    if not req.text or len(req.text.strip()) < 20:
        return {"error": "Input text too short or invalid."}

    task_id = task_store.create_task(
        task_type="ner_resume_extract",
        details={"text_length": len(req.text)}
    )

    model_res_parser = "DeezNutz1337/Resume-Parser-BERT_Based"

    mem_before = None
    try:
        task_store.update_task(task_id, status="processing")
        mem_before = None
        try:
            from services.ocr_service import get_memory_usage
            mem_before = get_memory_usage()
        except Exception:
            pass

        # Run NER in a subprocess (guarantees memory freed on exit)
        ner_out = await run_ner_in_subprocess(req.text, model_id=model_res_parser)
        mem_peak = ner_out.get("mem_peak_mb")
        results = ner_out.get("result", [])

        # Group entities
        def group_entities(results):
            grouped = {}
            for ent in results:
                label = ent.get("entity_group", ent.get("entity", "UNKNOWN"))
                value = ent.get("word") or ent.get("token") or ent.get("value") or ""
                if not value:
                    continue
                grouped.setdefault(label, []).append(value)
            # deduplicate values
            for k in list(grouped.keys()):
                grouped[k] = list(dict.fromkeys([v for v in grouped[k] if v]))
            return grouped

        parsed_entities = group_entities(results)

        # Attempt to extract structured sections via Gemini
        gemini_sections = await extract_resume_sections_with_gemini(req.text)
        
        # Merge Gemini sections with NER entities if extraction succeeded
        structured_output = {
            "parsed_entities": parsed_entities,
            "summary": {
                "entity_count": len(parsed_entities),
                "text_length": len(req.text)
            }
        }
        
        # Add Gemini sections only if they were successfully extracted
        if gemini_sections:
            structured_output["gemini_sections"] = {
                "education": gemini_sections.get("education", []),
                #"skills": gemini_sections.get("skills", []),
                "work_experience": gemini_sections.get("work_experience", [])
            }

        # Aggressively delete pipeline and collect
        try:
            from services.ocr_service import get_memory_usage
            mem_after = get_memory_usage()
        except Exception:
            mem_after = None

        task_store.update_task(
            task_id,
            status="completed",
            details={
                "entity_types": list(parsed_entities.keys()),
                "gemini_sections_present": bool(gemini_sections),
                "mem_before_mb": round(mem_before,1) if mem_before else None,
                "mem_peak_mb": round(mem_peak,1) if mem_peak else None,
                "mem_after_mb": round(mem_after,1) if mem_after else None,
            }
        )

        print("DEBUG RESULTS:", structured_output)

        return {
            "task_id": task_id,
            **structured_output
        }

    except Exception as e:
        task_store.update_task(task_id, status="error", details={"error": str(e)})
        raise