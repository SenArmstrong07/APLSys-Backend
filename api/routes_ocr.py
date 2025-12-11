# app/routers/ocr_router.py
from fastapi import APIRouter, UploadFile, File, Query, Request, HTTPException, Depends
from services.ocr_service import (
    run_ocr,
    extract_on_document,
    search_word,
    collect_all_pages,
    extract_context,
    ocr_space_ocr,
    pdf_to_images_bytes,
    save_pdf_images,           
    pdf_to_images_b64,
    get_processing_status,
    update_processing_status,
    create_doctr_ocr,
    dispose_doctr_ocr,
    get_memory_usage,
    _aggressive_model_unload,  # ADD THIS
    run_ocr_in_subprocess
)
from utils.task_store import TaskStore
from utils.ocr_rate_limiter import ocr_limiter
from utils.ocr_queue import ocr_queue
from typing import Optional
from PIL import Image
import numpy as np
import cv2
import io
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from model.request_schema import SearchRequest, TaskCreateRequest, TaskCreateBatchRequest
import asyncio
import tempfile

task_store = TaskStore()

async def get_doctr_dependency(request: Request):
    """
    FastAPI dependency that creates a doctr predictor for the lifetime of
    an OCR request and disposes it after processing to save memory.
    Attach this dependency at the router-level so all OCR endpoints get a fresh
    predictor only while handling the request.
    """
    predictor = create_doctr_ocr()
    # attach to request.state so handlers can access it
    request.state.doctr_predictor = predictor
    try:
        yield predictor
    finally:
        try:
            dispose_doctr_ocr(predictor)
        finally:
            # ensure it's removed from request state
            if hasattr(request.state, "doctr_predictor"):
                del request.state.doctr_predictor
                print("Disposed Doctr OCR predictor after request.")
        

router = APIRouter()

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

@router.post("/create-task", status_code=201)
async def create_tasks(body: TaskCreateBatchRequest):
    """
    Create a task or a batch of tasks.
    For batch_ocr provide `files: ["a.pdf", "b.jpg"]`.
    Returns:
      { "task_id": <batch_id>, "file_tasks": { "a.pdf": <id>, ... } }
    """
    try:
        if body.files and len(body.files) > 0:
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

            task_store.update_task(batch_id, details={
                **(body.details or {}),
                "files": body.files,
                "file_task_map": file_task_map
            })

            return {"task_id": batch_id, "file_tasks": file_task_map}

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
    """List tasks. Optional filter by status."""
    tasks = task_store.list_tasks(status=status, limit=limit)
    return {"tasks": tasks}

@router.get("/tasks/{task_id}")
async def get_task(task_id: int):
    """Get task by id."""
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
async def process_folder(request: Request, files: List[UploadFile] = File(...), _: None = Depends(get_doctr_dependency)):
    """Process multiple documents with progress tracking using Doctr."""
    batch_task_id = task_store.create_task(
        task_type="batch_folder_process",
        details={"file_count": len(files)}
    )
    
    try:
        task_store.update_task(batch_task_id, status="processing")
        # Get Doctr predictor from request.state (created by dependency)
        predictor = getattr(request.state, "doctr_predictor", None)
        if predictor is None:
            raise HTTPException(status_code=500, detail="Doctr OCR predictor not available")
        
        processed_results = []
        for idx, file in enumerate(files, 1):
            file_task_id = task_store.create_task(
                task_type="folder_process_file",
                filename=file.filename,
                details={"batch_id": batch_task_id, "index": idx}
            )
            
            try:
                task_store.update_task(file_task_id, status="processing")
                content = await file.read()
                
                # Use Doctr predictor via extract_on_document
                doc, exported = extract_on_document(content, predictor)
                
                # If doc is a PIL Image, run Doctr predictor on it
                temp_path = None
                if isinstance(doc, Image.Image):
                    import importlib
                    doctr_io = importlib.import_module("doctr.io")
                    
                    # Save PIL Image to temporary file
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        temp_path = tmp.name
                        doc.save(temp_path)
                    
                    # Run Doctr predictor
                    doc_file = doctr_io.DocumentFile.from_images([temp_path])
                    result = await asyncio.to_thread(predictor, doc_file)
                    exported = result.export()
                
                # Extract text from exported OCR result
                text = []
                for page in exported.get("pages", []):
                    for block in page.get("blocks", []):
                        if not isinstance(block, dict):
                            continue
                        for line in block.get("lines", []):
                            words = line.get("words", []) or []
                            line_text = " ".join([w.get("value", "") for w in words]).strip()
                            if line_text:
                                text.append(line_text)
                
                plain_text = "\n".join(text)
                
                task_store.update_task(
                    file_task_id,
                    status="completed",
                    progress=1.0,
                    details={"text_length": len(plain_text)}
                )
                processed_results.append({
                    "filename": file.filename,
                    "status": "success",
                    "text": plain_text
                })
                
            except Exception as e:
                task_store.update_task(file_task_id, status="error", details={"error": str(e)})
                processed_results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e)
                })
            finally:
                # Clean up temp file
                if "temp_path" in locals() and temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
            
            # Update batch progress
            task_store.update_task(batch_task_id, progress=idx / len(files))
        
        task_store.update_task(
            batch_task_id,
            status="completed",
            progress=1.0,
            details={"processed_count": len(processed_results)}
        )
        return {"results": processed_results, "task_id": batch_task_id}
    except Exception as e:
        task_store.update_task(batch_task_id, status="error", details={"error": str(e)})
        raise

# Add this helper
def get_client_ip(request: Request) -> str:
    """Extract client IP from request"""
    if request.client:
        return request.client.host
    return request.headers.get("x-forwarded-for", "unknown").split(",")[0].strip()


@router.post("/batch-ocr")
async def batch_ocr(request: Request, files: List[UploadFile] = File(...), _: None = Depends(get_doctr_dependency)):
    """Run OCR on multiple uploaded files with rate limiting and memory checks."""
    client_ip = get_client_ip(request)
    
    # 1) Check rate limit
    allowed, reason, retry_after = ocr_limiter.check_rate_limit(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=reason,
            headers={"Retry-After": str(retry_after)}
        )
    
    # 2) Check batch size
    batch_ok, batch_reason = ocr_limiter.check_batch_size(len(files))
    if not batch_ok:
        raise HTTPException(status_code=400, detail=batch_reason)
    
    # 3) Check individual file sizes
    for file in files:
        if file.size:
            file_ok, file_reason = ocr_limiter.check_file_size(file.size)
            if not file_ok:
                raise HTTPException(status_code=413, detail=file_reason)
    
    batch_task_id = task_store.create_task("batch_ocr", filename=None, details={"file_count": len(files)})
    task_store.update_task(batch_task_id, status="running", details={"started_by": "api"})
    
    results = []
    
    try:
        predictor = getattr(request.state, "doctr_predictor", None)
        if predictor is None:
            raise HTTPException(status_code=500, detail="Doctr OCR predictor not available")
        
        for idx, file in enumerate(files, 1):
            file_task_id = task_store.create_task(
                "ocr_file",
                filename=file.filename,
                details={"batch_id": batch_task_id, "index": idx}
            )
            
            temp_path = None
            try:
                task_store.update_task(file_task_id, status="processing")
                content = await file.read()
                
                # Use Doctr predictor via extract_on_document
                doc, exported = extract_on_document(content, predictor)
                
                # If doc is a PIL Image, run Doctr predictor on it
                if isinstance(doc, Image.Image):
                    import importlib
                    doctr_io = importlib.import_module("doctr.io")
                    
                    # Save PIL Image to temporary file
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        temp_path = tmp.name
                        doc.save(temp_path)
                    
                    # Run Doctr predictor
                    doc_file = doctr_io.DocumentFile.from_images([temp_path])
                    result = await asyncio.to_thread(predictor, doc_file)
                    exported = result.export()
                
                # Extract text from exported OCR result
                text = []
                for page in exported.get("pages", []):
                    for block in page.get("blocks", []):
                        if not isinstance(block, dict):
                            continue
                        for line in block.get("lines", []):
                            words = line.get("words", []) or []
                            line_text = " ".join([w.get("value", "") for w in words]).strip()
                            if line_text:
                                text.append(line_text)
                
                extracted_text = "\n".join(text)
                
                task_store.update_task(
                    file_task_id,
                    status="completed",
                    progress=1.0,
                    details={"text_snippet": extracted_text[:300]}
                )
                update_processing_status(file.filename, "completed")
                results.append({"filename": file.filename, "text": extracted_text.strip()})
                
            except Exception as e:
                task_store.update_task(file_task_id, status="error", details={"error": str(e)})
                update_processing_status(file.filename, "error", str(e))
                results.append({"filename": file.filename, "error": str(e)})
            finally:
                # Clean up temp file
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
                ocr_limiter.release_request(client_ip)
            
            task_store.update_task(batch_task_id, progress=float(idx) / len(files))
    
    except Exception as e:
        task_store.update_task(batch_task_id, status="error", details={"error": str(e)})
        raise
    
    task_store.update_task(batch_task_id, status="completed", progress=1.0)
    return {"results": results}

# Add worker helpers that create/dispose Doctr predictor inside the worker
async def _worker_process_full(content: bytes, filename: str, client_ip: str):
    """
    Run OCR in isolated subprocess to guarantee memory cleanup.
    """
    mem_before = get_memory_usage()
    
    try:
        print(f"Starting OCR in subprocess (mem: {mem_before:.1f}MB)...")
        
        # Run OCR in separate process—memory is freed when process exits
        exported = await run_ocr_in_subprocess(content, filename)
        
        return exported
    except Exception as e:
        print(f"Subprocess OCR failed: {e}")
        raise
    finally:
        # No need to dispose_doctr_ocr—subprocess already exited and freed memory
        mem_after = get_memory_usage()
        print(f"After subprocess OCR: {mem_after:.1f}MB (delta: {mem_after - mem_before:+.1f}MB)")
        
        try:
            ocr_limiter.release_request(client_ip)
        except Exception:
            pass

async def _worker_process_region(content: bytes, filename: str, client_ip: str):
    """
    Run OCR region extraction in isolated subprocess.
    """
    mem_before = get_memory_usage()
    
    try:
        print(f"Starting OCR region in subprocess (mem: {mem_before:.1f}MB)...")
        
        exported = await run_ocr_in_subprocess(content, filename)
        
        # Parse region results (same logic as before)
        text_lines = []
        confidences = []
        for page in exported.get("pages", []):
            for block in page.get("blocks", []):
                if not isinstance(block, dict):
                    continue
                for line in block.get("lines", []):
                    words = line.get("words", []) or []
                    # preserve left-to-right order if geometry exists
                    def _word_x(w):
                        geom = w.get("geometry") or []
                        if isinstance(geom, list) and len(geom) and isinstance(geom[0], list):
                            xs = [pt[0] for pt in geom if isinstance(pt, list) and len(pt) >= 2]
                            return min(xs) if xs else 0.0
                        return 0.0
                    words_sorted = sorted(words, key=_word_x)
                    line_text = " ".join([w.get("value", "") for w in words_sorted]).strip()
                    if line_text:
                        text_lines.append(line_text)
                    for w in words_sorted:
                        if "confidence" in w and w["confidence"] is not None:
                            confidences.append(w["confidence"])
        
        avg_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0
        return {"text": "\n".join(text_lines), "confidence": avg_conf, "ocr_layer": exported}
    finally:
        mem_after = get_memory_usage()
        print(f"After subprocess OCR region: {mem_after:.1f}MB (delta: {mem_after - mem_before:+.1f}MB)")
        
        try:
            ocr_limiter.release_request(client_ip)
        except Exception:
            pass

@router.post("/extract-full")
async def extract_text_full(ocrreq: Request, file: UploadFile = File(...)):
    task_id = task_store.create_task(
        task_type="extract_full",
        filename=file.filename
    )

    # Rate limit & file-size checks (uses existing limiter)
    client_ip = get_client_ip(ocrreq)
    allowed, reason, retry_after = ocr_limiter.check_rate_limit(client_ip)
    if not allowed:
        task_store.update_task(task_id, status="error", details={"error": reason})
        raise HTTPException(status_code=429, detail=reason, headers={"Retry-After": str(retry_after)})

    # file size guard (if available)
    size = getattr(file, "size", None)
    if size is not None:
        ok, msg = ocr_limiter.check_file_size(size)
        if not ok:
            task_store.update_task(task_id, status="error", details={"error": msg})
            ocr_limiter.release_request(client_ip)
            raise HTTPException(status_code=413, detail=msg)

    temp_path = None
    try:
        task_store.update_task(task_id, status="processing")
        content = await file.read()

        # Submit job to the OCR queue; worker creates/disposes predictor
        exported = await ocr_queue.submit(_worker_process_full, content, file.filename or "uploaded", client_ip)

        task_store.update_task(
            task_id,
            status="completed",
            details={"page_count": len(exported.get("pages", [])) if isinstance(exported, dict) else 0}
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

    client_ip = get_client_ip(ocrreq)
    allowed, reason, retry_after = ocr_limiter.check_rate_limit(client_ip)
    if not allowed:
        task_store.update_task(task_id, status="error", details={"error": reason})
        raise HTTPException(status_code=429, detail=reason, headers={"Retry-After": str(retry_after)})

    # file size guard
    size = getattr(file, "size", None)
    
    if size is not None:
        ok, msg = ocr_limiter.check_file_size(size)
        if not ok:
            task_store.update_task(task_id, status="error", details={"error": msg})
            ocr_limiter.release_request(client_ip)
            raise HTTPException(status_code=413, detail=msg)

    try:
        task_store.update_task(task_id, status="processing")
        content = await file.read()

        # Submit to queue; worker returns text + confidence
        result = await ocr_queue.submit(_worker_process_region, content, file.filename or "uploaded", client_ip)

        # Update task and return the extracted text with confidence
        task_store.update_task(
            task_id,
            status="completed",
            details={"text_length": len(result.get("text", "")), "confidence": result.get("confidence", 0.0)}
        )
        return {
            "text": result.get("text", ""),
            "confidence": result.get("confidence", 0.0),
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
