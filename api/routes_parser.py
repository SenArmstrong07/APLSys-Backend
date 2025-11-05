from fastapi import APIRouter, UploadFile, File, Query, Request
import pandas as pd
import os
from doctr.io import DocumentFile
from services.parsing_service import (
    extract_tables_from_pdf,
    extract_tables_from_pdf_with_camelot,
    extract_tables_from_docx_with_camelot,
    export_tables_to_csv,
    parse_document_text,
)
from utils.task_store import TaskStore
from model.request_schema import DocumentParseRequest, ResumeTextRequest
from typing import Optional

router = APIRouter()
task_store = TaskStore()

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

@router.post("/camelot_extract")
async def camelot_extract(file: UploadFile = File(...), pages: Optional[str] = Query("all")):
    task_id = task_store.create_task(
        task_type="camelot_extract",
        filename=file.filename,
        details={"pages": pages}
    )
    
    temp_path = f"temp_{file.filename}"
    try:
        task_store.update_task(task_id, status="processing")
        with open(temp_path, "wb") as f:
            f.write(await file.read())
            
        _, ext = os.path.splitext(temp_path)
        ext = ext.lower()
        
        if ext == ".docx":
            tables = extract_tables_from_docx_with_camelot(temp_path)
        else:
            tables = extract_tables_from_pdf_with_camelot(temp_path, pages=pages if pages is not None else "all")
            
        tables_json = [table.to_dict(orient="records") for table in tables]
        
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

@router.post("/export_csv")
async def export_csv(
    file: UploadFile = File(...), 
    method: str = Query("camelot"),
    base_filename: Optional[str] = Query("table"),
    pages: Optional[str] = Query("all")
):
    task_id = task_store.create_task(
        task_type="export_csv",
        filename=file.filename,
        details={
            "method": method,
            "base_filename": base_filename,
            "pages": pages
        }
    )
    
    temp_path = f"temp_{file.filename}"
    try:
        task_store.update_task(task_id, status="processing")
        with open(temp_path, "wb") as f:
            f.write(await file.read())
            
        _, ext = os.path.splitext(temp_path)
        ext = ext.lower()
        
        if ext == ".docx":
            tables = extract_tables_from_docx_with_camelot(temp_path)
        else:
            if method == "tabula":
                tables = extract_tables_from_pdf(temp_path, pages=pages if pages is not None else "all")
            else:
                tables = extract_tables_from_pdf_with_camelot(temp_path, pages=pages if pages is not None else "all")
                
        safe_base_filename = base_filename if base_filename is not None else "table"
        export_tables_to_csv(tables, base_filename=safe_base_filename)
        
        task_store.update_task(
            task_id, 
            status="completed",
            details={
                "table_count": len(tables),
                "output_base": safe_base_filename
            }
        )
        return {
            "message": f"Exported {len(tables)} tables to CSV with base filename '{safe_base_filename}'.",
            "task_id": task_id
        }
    except Exception as e:
        task_store.update_task(task_id, status="error", details={"error": str(e)})
        raise
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.post("/extract-resume-text")
async def extract_resume_txt(request: Request, file: UploadFile = File(...)):
    task_id = task_store.create_task(
        task_type="resume_extract",
        filename=file.filename
    )
    
    try:
        task_store.update_task(task_id, status="processing")
        print("File received. Waiting for extraction...")
        
        # Access your DocTR model
        model = request.app.state.ocr_model
        # Read uploaded file into bytes
        file_bytes = await file.read()
        # Load PDF or image into DocTR
        doc = DocumentFile.from_pdf(file_bytes)
        # Run OCR prediction
        result = model(doc)
        exported = result.export()
        
        plain_text = ""
        for page in exported["pages"]:
            for block in page["blocks"]:
                for line in block["lines"]:
                    line_text = " ".join(word["value"] for word in line["words"])
                    plain_text += line_text + " \n"
        
        task_store.update_task(
            task_id, 
            status="completed",
            details={"text_length": len(plain_text)}
        )
        return {"text": plain_text, "task_id": task_id}
    except Exception as e:
        task_store.update_task(task_id, status="error", details={"error": str(e)})
        raise
    
    
@router.post("/ner-extract-resume-profile")
async def ner_extract_resume_profile(req: ResumeTextRequest, request: Request):
    """
    Extract a structured resume profile using your fine-tuned NER models.
    """
    if not req.text or len(req.text.strip()) < 20:
        return {"error": "Input text too short or invalid."}

    task_id = task_store.create_task(
        task_type="ner_resume_extract",
        details={"text_length": len(req.text)}
    )

    try:
        task_store.update_task(task_id, status="processing")

        # Access resume-ner model
        basic_pipeline = request.app.state.ner_resume_pipeline_basic
        semantic_pipeline = request.app.state.ner_resume_pipeline_semantic

        # Run both models
        general_results = basic_pipeline(req.text)
        semantic_results = semantic_pipeline(req.text)

        # Merge outputs by label
        def group_entities(results):
            grouped = {}
            for ent in results:
                label = ent.get("entity_group", ent.get("entity", "UNKNOWN"))
                value = ent["word"]
                grouped.setdefault(label, []).append(value)
            return grouped

        general_entities = group_entities(general_results)
        semantic_entities = group_entities(semantic_results)

        # Combine intelligently — prioritize semantic fields
        combined = {**general_entities, **semantic_entities}

        # Join duplicates
        for k in combined:
            combined[k] = list(set(combined[k]))

        task_store.update_task(
            task_id,
            status="completed",
            details={"entity_types": list(combined.keys())}
        )

        return {
            "task_id": task_id,
            "parsed_entities": combined,
            "summary": {
                "entity_count": len(combined),
                "text_length": len(req.text)
            }
        }

    except Exception as e:
        task_store.update_task(task_id, status="error", details={"error": str(e)})
        raise