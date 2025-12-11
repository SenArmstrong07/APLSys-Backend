from fastapi import APIRouter, UploadFile, File, Query, Request, Depends
import pandas as pd
import os
from doctr.io import DocumentFile
import fitz
from services.parsing_service import (
    extract_tables_from_pdf,
    #extract_tables_from_pdf_with_camelot,
    #extract_tables_from_docx_with_camelot,
    export_tables_to_csv,
    parse_document_text,
)
from utils.task_store import TaskStore
from model.request_schema import DocumentParseRequest, ResumeTextRequest
from typing import Optional
from services.ocr_service import create_doctr_ocr, dispose_doctr_ocr

router = APIRouter()
task_store = TaskStore()

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

@router.post("/tabula_extract")
async def tabula_extract(file: UploadFile = File(...), pages: Optional[str] = Query("all")):
    task_id = task_store.create_task(
        task_type="tabula_extract",
        filename=file.filename,
        details={"pages": pages}
    )
    
    try:
        task_store.update_task(task_id, status="processing")
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
            
        tables = extract_tables_from_pdf(temp_path, pages=pages if pages is not None else "all")
        
        # Convert tables to JSON for API response
        tables_json = []
        for table in tables:
            if isinstance(table, pd.DataFrame):
                tables_json.append(table.to_dict(orient="records"))
            else:
                tables_json.append(table)
                
        task_store.update_task(
            task_id, 
            status="completed",
            details={"table_count": len(tables_json)}
        )
        return {"tables": tables_json, "task_id": task_id}
    except Exception as e:
        task_store.update_task(task_id, status="error", details={"error": str(e)})
        raise
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# @router.post("/camelot_extract")
# async def camelot_extract(file: UploadFile = File(...), pages: Optional[str] = Query("all")):
#     task_id = task_store.create_task(
#         task_type="camelot_extract",
#         filename=file.filename,
#         details={"pages": pages}
#     )
    
#     temp_path = f"temp_{file.filename}"
#     try:
#         task_store.update_task(task_id, status="processing")
#         with open(temp_path, "wb") as f:
#             f.write(await file.read())
            
#         _, ext = os.path.splitext(temp_path)
#         ext = ext.lower()
        
#         if ext == ".docx":
#             tables = extract_tables_from_docx_with_camelot(temp_path)
#         else:
#             tables = extract_tables_from_pdf_with_camelot(temp_path, pages=pages if pages is not None else "all")
            
#         tables_json = [table.to_dict(orient="records") for table in tables]
        
#         task_store.update_task(
#             task_id, 
#             status="completed",
#             details={"table_count": len(tables_json)}
#         )
#         return {"tables": tables_json, "task_id": task_id}
#     except Exception as e:
#         task_store.update_task(task_id, status="error", details={"error": str(e)})
#         raise
#     finally:
#         if os.path.exists(temp_path):
#             os.remove(temp_path)

# @router.post("/export_csv")
# async def export_csv(
#     file: UploadFile = File(...), 
#     method: str = Query("camelot"),
#     base_filename: Optional[str] = Query("table"),
#     pages: Optional[str] = Query("all")
# ):
#     task_id = task_store.create_task(
#         task_type="export_csv",
#         filename=file.filename,
#         details={
#             "method": method,
#             "base_filename": base_filename,
#             "pages": pages
#         }
#     )
    
#     temp_path = f"temp_{file.filename}"
#     try:
#         task_store.update_task(task_id, status="processing")
#         with open(temp_path, "wb") as f:
#             f.write(await file.read())
            
#         _, ext = os.path.splitext(temp_path)
#         ext = ext.lower()
        
#         if ext == ".docx":
#             tables = extract_tables_from_docx_with_camelot(temp_path)
#         else:
#             if method == "tabula":
#                 tables = extract_tables_from_pdf(temp_path, pages=pages if pages is not None else "all")
#             else:
#                 tables = extract_tables_from_pdf_with_camelot(temp_path, pages=pages if pages is not None else "all")
                
#         safe_base_filename = base_filename if base_filename is not None else "table"
#         export_tables_to_csv(tables, base_filename=safe_base_filename)
        
#         task_store.update_task(
#             task_id, 
#             status="completed",
#             details={
#                 "table_count": len(tables),
#                 "output_base": safe_base_filename
#             }
#         )
#         return {
#             "message": f"Exported {len(tables)} tables to CSV with base filename '{safe_base_filename}'.",
#             "task_id": task_id
#         }
#     except Exception as e:
#         task_store.update_task(task_id, status="error", details={"error": str(e)})
#         raise
#     finally:
#         if os.path.exists(temp_path):
#             os.remove(temp_path)

@router.post("/extract-resume-text")
async def extract_resume_txt(request: Request, file: UploadFile = File(...), _: None = Depends(get_doctr_dependency)):
    task_id = task_store.create_task(
        task_type="resume_extract",
        filename=file.filename
    )
    
    # Get Doctr predictor from request.state (created by dependency)
    ocr_model = getattr(request.state, "doctr_predictor", None)
    
    temp_path = None
    try:
        task_store.update_task(task_id, status="processing")
        print("File received. Waiting for extraction...")
        
        # Read uploaded file bytes once
        file_bytes = await file.read()
        _, ext = os.path.splitext(file.filename or "")
        ext = ext.lower()

        # Helper to save bytes to a temp file (used for docx / image fallbacks)
        def _save_temp(bts):
            p = f"temp_{file.filename}"
            with open(p, "wb") as tf:
                tf.write(bts)
            return p

        # 1) DOCX -> use parsing_service (assumed to handle docx)
        if ext == ".docx":
            temp_path = _save_temp(file_bytes)
            plain_text = parse_document_text(temp_path, ocr_model)

        # 2) Images -> use parsing_service (DocTR or image OCR path)
        elif ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            print("Image file detected — using image OCR path (DocTR via parsing_service).")
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
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@router.post("/ner-extract-resume-profile")
async def ner_extract_resume_profile(req: ResumeTextRequest, request: Request):
    """
    Extract a structured resume profile using the single NER pipeline stored
    as app.state.get_ner_resume_pipeline (lazy loader in main.py).
    """
    if not req.text or len(req.text.strip()) < 20:
        return {"error": "Input text too short or invalid."}

    task_id = task_store.create_task(
        task_type="ner_resume_extract",
        details={"text_length": len(req.text)}
    )

    try:
        task_store.update_task(task_id, status="processing")

        # Load the single resume NER pipeline (lazily cached by main.get_resume_parser)
        ner_pipeline = request.app.state.get_ner_resume_pipeline()

        # Run the pipeline once and group entities by label
        results = ner_pipeline(req.text)

        def group_entities(results):
            grouped = {}
            for ent in results:
                label = ent.get("entity_group", ent.get("entity", "UNKNOWN"))
                # pipeline outputs may use 'word' or 'word' field; fall back safely
                value = ent.get("word") or ent.get("word", ent.get("token", ""))
                if not value:
                    continue
                grouped.setdefault(label, []).append(value)
            # deduplicate values
            for k in list(grouped.keys()):
                grouped[k] = list(dict.fromkeys([v for v in grouped[k] if v]))
            return grouped

        parsed_entities = group_entities(results)

        task_store.update_task(
            task_id,
            status="completed",
            details={"entity_types": list(parsed_entities.keys())}
        )
        
        print("DEBUG RESULTS:", parsed_entities)

        return {
            "task_id": task_id,
            "parsed_entities": parsed_entities,
            "summary": {
                "entity_count": len(parsed_entities),
                "text_length": len(req.text)
            }
        }

    except Exception as e:
        task_store.update_task(task_id, status="error", details={"error": str(e)})
        raise