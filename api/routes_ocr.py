# app/routers/ocr_router.py
from fastapi import APIRouter, UploadFile, File, Query, Request, HTTPException
from services.ocr_service import (
    run_ocr,
    extract_on_document,
    search_word,
    collect_all_pages,
    google_vision_ocr,
    extract_context,
    ocr_space_ocr,
    pdf_to_images_bytes,
    save_pdf_images,           
    pdf_to_images_b64,
    get_processing_status,
    update_processing_status
)
from utils.task_store import TaskStore
from typing import Optional
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from model.request_schema import SearchRequest, TaskCreateRequest
router = APIRouter()

task_store = TaskStore()

@router.get("/status")
async def get_ocr_status():
    """
    Get current OCR processing status.
    Returns:
    {
        "processing": bool,  # True if any files are being processed
        "activeFiles": List[str],  # List of filenames currently being processed
        "files": [  # Detailed status list (kept for compatibility)
            {
                "filename": str,
                "status": str,  # "processing", "completed", "error"
                "error": str | None,
                "timestamp": str
            }
        ]
    }
    """
    return get_processing_status()

@router.post("/tasks", status_code=201)
async def create_task(payload: TaskCreateRequest):
    """
    Create a new task. Expects JSON body { task_type, filename?, details? }.
    Returns: {"task_id": int}
    """
    try:
        task_id = task_store.create_task(
            task_type=payload.task_type,
            filename=payload.filename,
            details=payload.details or {}
        )
        # optional: immediately set status to pending
        task_store.update_task(task_id, status="pending", progress=0.0)
        return {"task_id": task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create task: {str(e)}")

# New endpoints to query tasks
@router.get("/tasks")
async def list_tasks(status: Optional[str] = Query(None), limit: int = Query(100)):
    """
    List tasks. Optional filter by status.
    """
    tasks = task_store.list_tasks(status=status, limit=limit)
    return {"tasks": tasks}

@router.get("/tasks/{task_id}")
async def get_task(task_id: int):
    """
    Get task by id.
    """
    try:
        task = task_store.get_task(task_id)
        return {"task": task}
    except KeyError:
        return {"error": "Task not found"}, 404


@router.post("/run-ocr-on-document-upload")
async def run_ocr_document_upload(ocrreq: Request,file: UploadFile = File(...)):
    content = await file.read()
    doc, exported = extract_on_document(content,ocrreq)
    return {"pages": exported["pages"]}

@router.post("/pdf-to-images")
async def pdf_to_images_endpoint(
    ocrreq: Request,
    file: UploadFile = File(...),
    save: bool = Query(True, description="Save page images to disk under DOCUMENTS_FOLDER if true"),
    include_b64: bool = Query(False, description="Return base64 image data if true"),
    dpi: int = Query(300, description="DPI for conversion")
):
    task_id = task_store.create_task(
        task_type="pdf_to_images",
        filename=file.filename,
        details={"save": save, "dpi": dpi}
    )
    
    try:
        task_store.update_task(task_id, status="processing")
        content = await file.read()
        base_name = Path(file.filename or "uploaded_file").stem

        if save:
            meta = save_pdf_images(content, output_root=Path(os.getenv("DOCUMENTS_FOLDER", "./documents")), 
                                 base_name=base_name, dpi=dpi)
            task_store.update_task(
                task_id, 
                status="completed",
                details={"page_count": len(meta["pages"]), "output_dir": meta["dir"]}
            )
            return {
                "original_file": file.filename,
                "images_dir": meta["dir"],
                "pages": meta["pages"],
                "task_id": task_id
            }
        else:
            if include_b64:
                pages = pdf_to_images_b64(content, dpi=max(72, min(dpi, 300)))
                task_store.update_task(
                    task_id, 
                    status="completed",
                    details={"page_count": len(pages)}
                )
                return {"original_file": file.filename, "pages": pages, "task_id": task_id}
            else:
                pages_bytes = pdf_to_images_bytes(content, dpi=dpi)
                task_store.update_task(
                    task_id, 
                    status="completed",
                    details={"page_count": len(pages_bytes)}
                )
                return {"original_file": file.filename, "page_count": len(pages_bytes), "task_id": task_id}
    except Exception as e:
        task_store.update_task(task_id, status="error", details={"error": str(e)})
        raise

@router.post("/search-word")
async def search_word_endpoint(request: SearchRequest, ocrreq: Request):
    doc, exported = extract_on_document(request.file_path,ocrreq)
    matches = []
    matches = search_word(exported, request.query)
    return {"matches": matches}

@router.post("/process-folder")
async def process_folder(request: Request, files: List[UploadFile] = File(...)):
    """Process multiple documents with progress tracking."""
    batch_task_id = task_store.create_task(
        task_type="batch_folder_process",
        details={"file_count": len(files)}
    )
    
    try:
        task_store.update_task(batch_task_id, status="processing")
        ocr_results = await batch_ocr(request, files)
        
        processed_results = []
        for idx, item in enumerate(ocr_results["results"], 1):
            if "error" in item:
                processed_results.append({
                    "filename": item["filename"],
                    "status": "error",
                    "error": item["error"]
                })
            else:
                processed_results.append({
                    "filename": item["filename"],
                    "status": "success",
                    "text": item["text"]
                })
            # Update batch progress
            task_store.update_task(batch_task_id, progress=idx/len(files))
        
        task_store.update_task(
            batch_task_id, 
            status="completed",
            details={"processed_count": len(processed_results)}
        )
        return {"results": processed_results, "task_id": batch_task_id}
    except Exception as e:
        task_store.update_task(batch_task_id, status="error", details={"error": str(e)})
        raise

@router.post("/batch-ocr")
async def batch_ocr(ocrreq: Request, files: List[UploadFile] = File(...)):
    """
    Run OCR on multiple uploaded files and return only the extracted text.
    """
    model = ocrreq.app.state.ocr_model
    
    # create a batch task
    batch_task_id = task_store.create_task("batch_ocr", filename=None, details={"file_count": len(files)})
    # mark batch as started
    task_store.update_task(batch_task_id, status="running", details={"started_by": "api"})
    
    results = []
    total = len(files)
    for idx, file in enumerate(files, start=1):
        file_task_id = task_store.create_task("ocr_file", filename=file.filename, details={"batch_id": batch_task_id, "index": idx})
        try:
            update_processing_status(file.filename, "processing")
            # set file task to processing
            task_store.update_task(file_task_id, status="processing", progress=0.0)
            # Await the async OCR function
            ocr_result = await run_ocr(model, file)
            
            # Extract only text from OCR result
            extracted_text = ""
            for page in ocr_result["pages"]:
                page_text = []
                for block in page["blocks"]:
                    for line in block["lines"]:
                        line_text = " ".join(str(word["value"]) for word in line["words"])
                        page_text.append(line_text)
                extracted_text += "\n".join(page_text) + "\n"
                
             # mark file task done
            task_store.update_task(file_task_id, status="completed", progress=1.0, details={"text_snippet": extracted_text[:300]})
            # update batch progress
            task_store.update_task(batch_task_id, progress=float(idx) / total)
            update_processing_status(file.filename, "completed")
            results.append({
                "filename": file.filename,
                "text": extracted_text.strip()
            })
        except Exception as e:
            task_store.update_task(file_task_id, status="error", details={"error": str(e)})
            update_processing_status(file.filename, "error", str(e))
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    # finish batch
    task_store.update_task(batch_task_id, status="completed", progress=1.0)
    return {"results": results}

@router.post("/collect-all-pages")
def collect_all_pages_endpoint(ocrreq:Request, file_path: str = Query(..., description="Path to image or PDF file")):
    pages = collect_all_pages(file_path, ocrreq)
    # Remove image objects from response for JSON serialization
    for page in pages:
        if "image" in page:
            page["image"] = "Image data omitted"
    return {"pages": pages}

# ...existing code...

@router.post("/extract-full")
async def extract_text_full(file: UploadFile, ocrreq: Request):
    task_id = task_store.create_task(
        task_type="extract_full",
        filename=file.filename
    )
    
    try:
        task_store.update_task(task_id, status="processing")
        model = ocrreq.app.state.ocr_model
        content = await file.read()
        result = model(content)
        task_store.update_task(
            task_id, 
            status="completed",
            details={"page_count": len(result.get("pages", []))}
        )
        return {"result": result, "task_id": task_id}
    except Exception as e:
        task_store.update_task(task_id, status="error", details={"error": str(e)})
        raise

@router.post("/extract-region")
async def extract_text_region(ocrreq: Request, file: UploadFile = File(...)):
    task_id = task_store.create_task(
        task_type="extract_region",
        filename=file.filename
    )
    
    try:
        task_store.update_task(task_id, status="processing")
        model = ocrreq.app.state.ocr_model
        content = await file.read()
        result = model(content)
        
        text = []
        confidences = []
        for page in result["pages"]:
            for block in page["blocks"]:
                for line in block["lines"]:
                    line_text = " ".join([word["value"] for word in line["words"]])
                    text.append(line_text)
                    for word in line["words"]:
                        if "confidence" in word and word["confidence"] is not None:
                            confidences.append(word["confidence"])
        
        avg_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0
        task_store.update_task(
            task_id, 
            status="completed",
            details={
                "text_length": len("\n".join(text)),
                "confidence": avg_conf
            }
        )
        return {
            "text": "\n".join(text), 
            "confidence": avg_conf,
            "task_id": task_id
        }
    except Exception as e:
        task_store.update_task(task_id, status="error", details={"error": str(e)})
        raise

@router.post("/extract-metadata")
async def extract_metadata_from_image(ocrreq: Request, file: UploadFile = File(...)):
    task_id = task_store.create_task(
        task_type="extract_metadata",
        filename=file.filename
    )
    
    try:
        task_store.update_task(task_id, status="processing")
        content = await file.read()
        exported = ocr_space_ocr(content)
        
        full_text = []
        for page in exported["pages"]:
            for block in page.get("blocks", []):
                for line in block.get("lines", []):
                    line_text = " ".join([word["value"] for word in line.get("words", [])])
                    full_text.append(line_text)
        plain_text = "\n".join(full_text)
        
        task_store.update_task(
            task_id, 
            status="completed",
            details={
                "text_length": len(plain_text),
                "page_count": len(exported["pages"])
            }
        )
        return {
            "text": plain_text,
            "ocr_layer": exported,
            "task_id": task_id
        }
    except Exception as e:
        task_store.update_task(task_id, status="error", details={"error": str(e)})
        raise

@router.post("/search-results")
async def search_results(
    request: Request,
    query: str = Query(..., description="Text to search for"),
    scripts: List[UploadFile] = File(...)
):
    """
    Search through uploaded script files from frontend for matching text.
    """
    matches = []
    
    for script_file in scripts:
        try:
            content = await script_file.read()
            script_content = json.loads(content)
            
            # Extract text from OCR results
            text_content = ""
            for page in script_content["ocr_results"]["pages"]:
                for block in page.get("blocks", []):
                    for line in block.get("lines", []):
                        line_text = " ".join(
                            word["value"] for word in line.get("words", [])
                        )
                        text_content += line_text + " "
            
            # Check if query exists in text
            if query.lower() in text_content.lower():
                matches.append({
                    "document": script_content.get("original_file", script_file.filename),
                    "script_file": script_file.filename,
                    "context": extract_context(text_content, query)
                })
                
        except Exception as e:
            print(f"Error processing {script_file.filename}: {str(e)}")
            continue
    
    return {"matches": matches}
