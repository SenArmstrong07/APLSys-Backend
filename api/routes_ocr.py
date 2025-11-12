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
from model.request_schema import SearchRequest, TaskCreateRequest, TaskCreateBatchRequest
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

# New: create tasks endpoint that supports batch creation and returns per-file task ids
@router.post("/create-task", status_code=201)
async def create_tasks(body: TaskCreateBatchRequest):
    """
    Create a task or a batch of tasks.
    For batch_ocr provide `files: ["a.pdf", "b.jpg"]`.
    Returns:
      { "task_id": <batch_id>, "file_tasks": { "a.pdf": <id>, ... } }
    """
    try:
        # If a list of files is provided, create a parent batch and child tasks
        if body.files and len(body.files) > 0:
            # create batch task
            batch_id = task_store.create_task(
                task_type=body.task_type,
                filename=body.filename,
                details={"file_count": len(body.files), **(body.details or {})}
            )

            file_task_map: Dict[str, int] = {}
            for idx, fname in enumerate(body.files, start=1):
                file_task_id = task_store.create_task(
                    task_type="ocr_file" if body.task_type == "batch_ocr" else "subtask",
                    filename=fname,
                    details={"batch_id": batch_id, "index": idx}
                )
                file_task_map[fname] = file_task_id

            # persist mapping in batch details for easy reconciliation
            task_store.update_task(batch_id, details={
                **(body.details or {}),
                "files": body.files,
                "file_task_map": file_task_map
            })

            return {"task_id": batch_id, "file_tasks": file_task_map}

        # Otherwise create a single task
        single_id = task_store.create_task(
            task_type=body.task_type,
            filename=body.filename,
            details=body.details or {}
        )
        return {"task_id": single_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create task(s): {str(e)}")

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
        # Normalize input (pdf/docx/image bytes) into the DocumentFile/exported dict
        doc, exported = extract_on_document(content, model)
        task_store.update_task(
            task_id, 
            status="completed",
            details={"page_count": len(exported.get("pages", []))}
        )
        return {"result": exported, "task_id": task_id}
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
        
        # Normalize bytes/path into exported OCR structure (dict with "pages")
        doc, result = extract_on_document(content, model)

        text = []
        confidences = []
        for page in result.get("pages", []):
            for block in page.get("blocks", []):
                if not isinstance(block, dict):
                    continue
                for line in block.get("lines", []):
                    # sort words left-to-right using geometry if available to preserve correct reading order
                    words = line.get("words", []) or []
                    def _word_x(w):
                        geom = w.get("geometry") or []
                        # geometry expected as list of [ [x,y], ... ] normalized coordinates
                        if isinstance(geom, list) and len(geom) and isinstance(geom[0], list):
                            xs = [pt[0] for pt in geom if isinstance(pt, list) and len(pt) >= 2]
                            return min(xs) if xs else 0.0
                        return 0.0
                    words_sorted = sorted(words, key=_word_x)
                    line_text = " ".join([w.get("value", "") for w in words_sorted]).strip()
                    if line_text:
                        text.append(line_text)
                    for word in words_sorted:
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
                # skip if page is not a dict (guard against malformed input)
                if not isinstance(page, dict):
                    continue
                blocks = page.get("blocks", []) if isinstance(page, dict) else []
                for block in blocks:
                    # skip if block is not a dict
                    if not isinstance(block, dict):
                        continue
                    lines = block.get("lines", [])
                    for line in lines:
                        # skip if line is not a dict
                        if not isinstance(line, dict):
                            continue
                        words = line.get("words", [])
                        line_text = " ".join(
                            word.get("value", "") for word in words if isinstance(word, dict)
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
