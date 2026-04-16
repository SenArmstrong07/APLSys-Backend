# app/routers/ocr_router.py
from fastapi import APIRouter, UploadFile, File, Query, Request, HTTPException, Depends, Form
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
    _aggressive_model_unload,
    bbox_to_original,
    intersects,
    run_ocr_in_subprocess,
    classify_document_type,
    google_cloud_vision_ocr
)
from utils.task_store import TaskStore
from utils.ocr_rate_limiter import ocr_limiter
from utils.ocr_queue import ocr_queue
from typing import Optional, cast
from os import getenv
from PIL import Image
import numpy as np
import cv2
import io
import json
from pathlib import Path
from typing import List, Dict, Any
from model.request_schema import SearchRequest, TaskCreateRequest, TaskCreateBatchRequest
import asyncio
import tempfile
import base64
import requests

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
            meta = save_pdf_images(content, output_root=Path(getenv("DOCUMENTS_FOLDER", "./documents")), 
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
    """Process multiple documents with progress tracking using Cloud Vision first, then DocTR fallback."""
    batch_task_id = task_store.create_task(
        task_type="batch_folder_process",
        details={"file_count": len(files)}
    )
    
    try:
        task_store.update_task(batch_task_id, status="processing")
        
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
                
                # Try Cloud Vision first
                ocr_provider = "cloud_vision"
                plain_text = ""
                
                try:
                    # Convert content to image bytes for Cloud Vision
                    if content.startswith(b'%PDF'):
                        # PDF - convert first page to image
                        pages_images = pdf_to_images_bytes(content, dpi=300)
                        if pages_images:
                            image_bytes = pages_images[0]
                        else:
                            raise ValueError("Could not convert PDF to image")
                    else:
                        # Assume it's an image
                        image_bytes = content
                    
                    # Call Cloud Vision OCR
                    vision_result = await asyncio.to_thread(google_cloud_vision_ocr, image_bytes)
                    
                    # Extract text from Cloud Vision result
                    text_parts = []
                    for page in vision_result.get("pages", []):
                        if isinstance(page, dict) and "text" in page:
                            text_parts.append(page["text"])
                    plain_text = "\n".join(text_parts)
                    
                    if not plain_text.strip():
                        raise ValueError("Cloud Vision returned empty text")
                        
                except Exception as e:
                    print(f"Cloud Vision failed for {file.filename}: {str(e)}, falling back to DocTR")
                    ocr_provider = "doctr"
                    
                    # Fallback to DocTR
                    # Get Doctr predictor from request.state (created by dependency)
                    predictor = getattr(request.state, "doctr_predictor", None)
                    if predictor is None:
                        raise HTTPException(status_code=500, detail="Doctr OCR predictor not available")
                    
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
                    
                    # Clean up temp file
                    if "temp_path" in locals() and temp_path and Path(temp_path).exists():
                        try:
                            Path(temp_path).unlink()
                        except Exception:
                            pass
                
                task_store.update_task(
                    file_task_id,
                    status="completed",
                    progress=1.0,
                    details={"text_length": len(plain_text), "ocr_provider": ocr_provider}
                )
                processed_results.append({
                    "filename": file.filename,
                    "status": "success",
                    "text": plain_text,
                    "ocr_provider": ocr_provider
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
                if "temp_path" in locals() and temp_path and Path(temp_path).exists():
                    try:
                        Path(temp_path).unlink()
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


# --- Add helper: compute normalized bbox for words (attach as 'bbox' on each word) ---
def _attach_normalized_bboxes(exported: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mutate `exported` in-place: for each word with 'geometry' (list of points),
    compute a normalized bbox {x,y,width,height} based on min/max of points.
    Frontend can multiply these normalized values by image dimensions to render boxes.
    """
    if not isinstance(exported, dict):
        return exported
    pages = exported.get("pages", [])
    for page in pages:
        if not isinstance(page, dict):
            continue
        for block in page.get("blocks", []) or []:
            if not isinstance(block, dict):
                continue
            for line in block.get("lines", []) or []:
                if not isinstance(line, dict):
                    continue
                for word in line.get("words", []) or []:
                    geom = word.get("geometry")
                    if not geom or not isinstance(geom, list):
                        continue
                    xs = []
                    ys = []
                    # geometry may be list of [x,y] or nested lists; flatten defensively
                    for pt in geom:
                        try:
                            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                                xs.append(float(pt[0]))
                                ys.append(float(pt[1]))
                        except Exception:
                            continue
                    if not xs or not ys:
                        continue
                    min_x = min(xs)
                    min_y = min(ys)
                    max_x = max(xs)
                    max_y = max(ys)
                    word['bbox'] = {
                        "x": min_x,
                        "y": min_y,
                        "width": max_x - min_x,
                        "height": max_y - min_y
                    }
    return exported


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
                
                # Try quick OCR via ocr.space first; if it yields text, use it and skip DocTR.
                try:
                    oscr = ocr_space_ocr(content)
                    pages = oscr.get("pages", []) or []
                    ocr_lines = []
                    for page in pages:
                        for block in page.get("blocks", []) or []:
                            for line in block.get("lines", []) or []:
                                words = line.get("words", []) or []
                                line_text = " ".join([w.get("value", "") for w in words if isinstance(w, dict)]).strip()
                                if line_text:
                                    ocr_lines.append(line_text)
                    if ocr_lines:
                        extracted_text = "\n".join(ocr_lines)
                        task_store.update_task(
                            file_task_id,
                            status="completed",
                            progress=1.0,
                            details={"text_snippet": extracted_text[:300], "source": "ocr_space"}
                        )
                        update_processing_status(file.filename, "completed")
                        results.append({"filename": file.filename, "text": extracted_text.strip(), "source": "ocr_space"})
                        # Skip DocTR processing and continue with next file
                        continue
                except Exception as e:
                    # Log and fall back to DocTR path below
                    print(f"ocr.space failed for {file.filename}: {e}")

                # Use Doctr predictor via extract_on_document (fallback)
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
                if temp_path and Path(temp_path).exists():
                    try:
                        Path(temp_path).unlink()
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
    task_id = task_store.create_task("ocr_full", filename=filename, details={})
    mem_before = get_memory_usage()
    try:
        task_store.update_task(task_id, status="processing", details={"mem_before_mb": round(mem_before,1)})
        # run OCR (subprocess returns dict possibly containing mem_peak_mb)
        exported = await run_ocr_in_subprocess(content, filename)
        # exported may be {"result": ..., "mem_peak_mb": X} or the result directly
        mem_peak = None
        if isinstance(exported, dict) and "mem_peak_mb" in exported:
            mem_peak = exported.get("mem_peak_mb")
            result = exported.get("result")
        else:
            result = exported
        mem_after = get_memory_usage()
        # persist snapshots
        task_store.update_task(task_id, status="completed", details={
            "mem_before_mb": round(mem_before,1),
            "mem_peak_mb": round(mem_peak,1) if mem_peak else None,
            "mem_after_mb": round(mem_after,1),
            "filename": filename
        })
        return result
    except Exception as e:
        task_store.update_task(task_id, status="error", details={"error": str(e)})
        raise
    finally:
        try:
            ocr_limiter.release_request(client_ip)
        except Exception:
            pass

async def _worker_process_region(content: bytes, filename: str, client_ip: str, region: dict):
    """
    Run OCR region extraction in isolated subprocess.
    Records mem_before / mem_peak / mem_after into TaskStore.
    region: normalized bbox in original-image space {"x","y","width","height"} or None
    """
    task_id = task_store.create_task("ocr_region", filename=filename, details={})
    mem_before = get_memory_usage()
    try:
        task_store.update_task(task_id, status="processing", details={"mem_before_mb": round(mem_before,1)})

        print(f"Starting OCR region in subprocess (mem: {mem_before:.1f}MB)...")
        exported = await run_ocr_in_subprocess(content, filename)

        # exported may be {"result": ..., "mem_peak_mb": X} or the result directly
        mem_peak = None
        if isinstance(exported, dict) and "mem_peak_mb" in exported and "result" in exported:
            mem_peak = exported.get("mem_peak_mb")
            result = exported.get("result", {})
        elif isinstance(exported, dict) and "pages" in exported:
            result = exported
        else:
            result = {}

        # Parse region results (filter words by region if provided)
        text_lines = []
        confidences = []
        all_words = []  # <- new accumulator for words across pages/blocks/lines

        for page in result.get("pages", []):
            for block in page.get("blocks", []):
                if not isinstance(block, dict):
                    continue
                for line in block.get("lines", []):
                    words = line.get("words", []) or []
                    # preserve left-to-right order if geometry exists
                    def _word_x(w):
                        geom = w.get("geometry")
                        if not geom:
                            return 0.0
                        if isinstance(geom, list) and len(geom) and isinstance(geom[0], list):
                            xs = [pt[0] for pt in geom if isinstance(pt, list) and len(pt) >= 2]
                            return min(xs) if xs else 0.0
                        return 0.0

                    # If region provided, filter words by intersection
                    if region:
                        filtered = []
                        for w in words:
                            geom = w.get("geometry")
                            if geom and intersects(geom, region):
                                filtered.append(w)
                        words = filtered

                    # collect words for later averaging/processing
                    for w in words:
                        all_words.append(w)

                    line_text = " ".join([w.get("value", "") for w in words]).strip()
                    if line_text:
                        text_lines.append(line_text)
                    for w in words:
                        if "confidence" in w and w["confidence"] is not None:
                            confidences.append(w["confidence"])

        # compute avg_conf using collected all_words (avoid using 'words' which may be undefined)
        total = 0.0
        weight = 0.0
        for w in all_words:
            conf = w.get("confidence")
            if conf is not None:
                length = max(len(w.get("value", "")), 1)
                total += conf * length
                weight += length

        avg_conf = round(total / weight, 2) if weight else 0.0
        mem_after = get_memory_usage()

        # persist snapshots
        task_store.update_task(task_id, status="completed", details={
            "mem_before_mb": round(mem_before,1),
            "mem_peak_mb": round(mem_peak,1) if mem_peak else None,
            "mem_after_mb": round(mem_after,1),
            "filename": filename,
            "text_length": len("\n".join(text_lines))
        })

        return {"text": "\n".join(text_lines), "confidence": avg_conf, "ocr_layer": result}
    except Exception as e:
        task_store.update_task(task_id, status="error", details={"error": str(e)})
        raise
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

        # Attempt to read rotation from multipart form (defaults to 0)
        try:
            form = await ocrreq.form()
            rotation_value = form.get("rotation", 0)
            rotation = float(str(rotation_value)) if rotation_value else 0.0
        except Exception:
            rotation = 0.0

        # If rotation provided and the uploaded bytes are an image, straighten before OCR
        if rotation and rotation % 360 != 0:
            try:
                img = Image.open(io.BytesIO(content))
                # Frontend rotation indicates how much the image was rotated; rotate by negative to straighten
                straight = img.rotate(-rotation, expand=True)
                buf = io.BytesIO()
                straight.save(buf, format="PNG", optimize=True)
                content = buf.getvalue()
            except Exception:
                # If not an image (e.g., PDF), skip rotation and continue
                pass

        # Try Google Cloud Vision API first
        try:
            print(f"Attempting Google Cloud Vision OCR for {file.filename}")
            vision_result = await asyncio.to_thread(google_cloud_vision_ocr, content)

            # Convert Vision result to expected format
            exported = {
                "pages": vision_result.get("pages", []),
                "source": "google-cloud-vision"
            }

            task_store.update_task(
                task_id,
                status="completed",
                details={
                    "page_count": len(exported.get("pages", [])),
                    "source": "google-cloud-vision",
                    "full_text_length": len(vision_result.get("full_text", ""))
                }
            )
            return {"result": exported, "task_id": task_id}

        except Exception as vision_error:
            print(f"Google Cloud Vision failed for {file.filename}: {vision_error}")
            print("Falling back to DocTR OCR...")

            # Fallback to DocTR
            exported = await ocr_queue.submit(_worker_process_full, content, file.filename or "uploaded", client_ip)

            task_store.update_task(
                task_id,
                status="completed",
                details={
                    "page_count": len(exported.get("pages", [])) if isinstance(exported, dict) else 0,
                    "source": "doctr_fallback",
                    "vision_error": str(vision_error)
                }
            )

        # Attach normalized per-word bbox to exported result (frontend can map to pixels)
        try:
            exported = _attach_normalized_bboxes(exported)
        except Exception:
            # non-fatal: continue returning original exported if augmentation fails
            pass

        return {"result": exported, "task_id": task_id}
    except Exception as e:
        task_store.update_task(task_id, status="error", details={"error": str(e)})
        raise
    finally:
        # ensure we always release the rate-limiter reservation for this client
        try:
            ocr_limiter.release_request(client_ip)
        except Exception:
            pass

@router.post("/extract-region")
async def extract_text_region(
    ocrreq: Request,
    file: UploadFile = File(...),
    rotation: float = Form(0.0),
    bbox_x: Optional[float] = Form(None),
    bbox_y: Optional[float] = Form(None),
    bbox_w: Optional[float] = Form(None),
    bbox_h: Optional[float] = Form(None),
):
    task_id = task_store.create_task(
        task_type="extract_region",
        filename=file.filename
    )


    client_ip = get_client_ip(ocrreq)
    allowed, reason, retry_after = ocr_limiter.check_rate_limit(client_ip)
    if not allowed:
        task_store.update_task(task_id, status="error", details={"error": reason})
        raise HTTPException(status_code=429, detail=reason, headers={"Retry-After": str(retry_after)})

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

        # ---------- Load image ----------
        import io
        from PIL import Image

        img = Image.open(io.BytesIO(content)).convert("RGB")
        
        
        print("Incoming bbox px:", bbox_x, bbox_y, bbox_w, bbox_h)
        print("Image size:", img.size)

        # ---------- STRAIGHTEN BEFORE cropping ----------
        rotation = int(round(rotation)) % 360
        if rotation in (90, 180, 270):
            img = img.rotate(-rotation, expand=True)

        # ---------- Build bbox (ORIGINAL IMAGE PIXELS ONLY) ----------
        bbox = None
        if (
            bbox_x is not None
            and bbox_y is not None
            and bbox_w is not None
            and bbox_h is not None
        ):
            x = int(round(bbox_x))
            y = int(round(bbox_y))
            w = int(round(bbox_w))
            h = int(round(bbox_h))

            bbox = (x, y, x + w, y + h)
            # Clamp bbox to image bounds (safety)
            iw, ih = img.size
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(x1, iw - 1))
            y1 = max(0, min(y1, ih - 1))
            x2 = max(x1 + 1, min(x2, iw))
            y2 = max(y1 + 1, min(y2, ih))
            bbox = (x1, y1, x2, y2)

            img = img.crop(bbox)

        # ---------- Try Google Cloud Vision API first ----------
        try:
            print(f"Attempting Google Cloud Vision OCR for region extraction on {file.filename}")
            
            # Convert cropped image to bytes for Vision API
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            img_bytes = buf.getvalue()
            
            # Use Google Cloud Vision
            vision_result = await asyncio.to_thread(google_cloud_vision_ocr, img_bytes)
            
            # Extract text and confidence from Vision result
            full_text = vision_result.get("full_text", "")
            
            # Calculate average confidence
            confidences = []
            for page in vision_result.get("pages", []):
                for block in page.get("blocks", []):
                    for line in block.get("lines", []):
                        for word in line.get("words", []):
                            if "confidence" in word and word["confidence"] is not None:
                                confidences.append(word["confidence"])
            
            avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

            task_store.update_task(
                task_id, 
                status="completed", 
                details={
                    "text_length": len(full_text),
                    "source": "google-cloud-vision",
                    "confidence": avg_confidence
                }
            )
            return {
                "text": full_text, 
                "confidence": avg_confidence, 
                "task_id": task_id,
                "source": "google-cloud-vision"
            }

        except Exception as vision_error:
            print(f"Google Cloud Vision failed for region extraction on {file.filename}: {vision_error}")
            print("Falling back to DocTR OCR...")

            # ---------- DocTR / worker path (fallback) ----------
            buf = io.BytesIO()
            img.save(buf, format="PNG")

            result = await ocr_queue.submit(
                _worker_process_region,
                buf.getvalue(),
                file.filename or "uploaded",
                client_ip,
                None  # bbox already applied
            )

            for page in result.get("pages", []):
                if not isinstance(page, dict):
                    continue
                for block in page.get("blocks", []) or []:
                    if not isinstance(block, dict):
                        continue
                    for line in block.get("lines", []) or []:
                        if not isinstance(line, dict):
                            continue
                        for word in line.get("words", []) or []:
                            if isinstance(word, dict):
                                wval = word.get("value")
                                geom = word.get("geometry")
                            else:
                                wval = None
                                geom = None
                            print(f"WORD: {wval!r} GEOM: {geom!r}")

            # If client provided a bbox we may have applied it; otherwise the client already sent a cropped image.
            if bbox is not None:
                print("Applied bbox (pixels):", bbox)
            else:
                print("No bbox provided; image already cropped by client.")

            print("DocTR pages:", result.get("pages", []))


            task_store.update_task(
                task_id,
                status="completed",
                details={
                    "text_length": len(result.get("text", "")), 
                    "confidence": result.get("confidence", 0.0),
                    "source": "doctr_fallback",
                    "vision_error": str(vision_error)
                }
            )

            return {
                "text": result.get("text", ""),
                "confidence": result.get("confidence", 0.0),
                "task_id": task_id,
                "source": "doctr_fallback"
            }

    except Exception as e:
        task_store.update_task(task_id, status="error", details={"error": str(e)})
        raise
    finally:
        # ensure we always release the rate-limiter reservation for this client
        try:
            ocr_limiter.release_request(client_ip)
        except Exception:
            pass

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

@router.post("/classify-document")
async def classify_document(request: Request, file: UploadFile = File(...)):
    """
    Classify the document type using Gemini with a local heuristic fallback.
    Returns: { task_id, type, confidence, source, ... }
    """
    task_id = task_store.create_task(task_type="classify_document", filename=file.filename)
    task_store.update_task(task_id, status="processing")
    try:
        content = await file.read()
        result = classify_document_type(content, filename=file.filename or "uploaded")
        task_store.update_task(task_id, status="completed", details={"doc_type": result.get("type")})
        return {"task_id": task_id, **result}
    except Exception as e:
        task_store.update_task(task_id, status="error", details={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/google-cloud-vision-ocr")
async def google_cloud_vision_ocr_endpoint(
    request: Request,
    file: UploadFile = File(...),
    include_confidence: bool = Query(True, description="Include confidence scores in response")
):
    """
    Perform OCR using Google Cloud Vision API with service account credentials.
    
    This endpoint uses your GCR (Google Cloud) service account credentials to perform
    high-quality OCR on uploaded images or documents.
    
    Args:
        file: Image or document file to OCR (JPEG, PNG, PDF, TIFF, WebP, etc.)
        include_confidence: Include confidence scores for each word
        
    Returns:
        {
            "task_id": int,
            "pages": [
                {
                    "text": "extracted text",
                    "blocks": [...],
                    "confidence": float,
                    "page_number": int
                }
            ],
            "full_text": "complete extracted text",
            "source": "google-cloud-vision"
        }
    """
    task_id = task_store.create_task(
        task_type="ocr_google_vision",
        filename=file.filename,
        details={"include_confidence": include_confidence}
    )
    
    client_ip = get_client_ip(request)
    
    try:
        # Check rate limit
        allowed, reason, retry_after = ocr_limiter.check_rate_limit(client_ip)
        if not allowed:
            task_store.update_task(task_id, status="error", details={"error": reason})
            raise HTTPException(
                status_code=429,
                detail=reason,
                headers={"Retry-After": str(retry_after)}
            )
        
        # Check file size
        size = getattr(file, "size", None)
        if size is not None:
            ok, msg = ocr_limiter.check_file_size(size)
            if not ok:
                task_store.update_task(task_id, status="error", details={"error": msg})
                ocr_limiter.release_request(client_ip)
                raise HTTPException(status_code=413, detail=msg)
        
        task_store.update_task(task_id, status="processing")
        content = await file.read()
        
        # If PDF, convert to image first (Google Vision can handle some PDFs, but images work better)
        if file.filename and file.filename.lower().endswith('.pdf'):
            try:
                pages_bytes = pdf_to_images_bytes(content, dpi=200)
                if pages_bytes:
                    content = pages_bytes[0]  # Use first page for now
            except Exception as e:
                print(f"PDF conversion failed, attempting direct processing: {e}")
        
        # Call Google Cloud Vision API
        result = await asyncio.to_thread(google_cloud_vision_ocr, content)
        
        # Filter out confidence if not requested
        if not include_confidence:
            for page in result.get("pages", []):
                for block in page.get("blocks", []):
                    for line in block.get("lines", []):
                        for word in line.get("words", []):
                            word.pop("confidence", None)
        
        # Update task with results
        pages_count = len(result.get("pages", []))
        full_text = result.get("full_text", "")
        
        task_store.update_task(
            task_id,
            status="completed",
            details={
                "page_count": pages_count,
                "text_length": len(full_text),
                "source": "google-cloud-vision"
            }
        )
        
        return {
            "task_id": task_id,
            "pages": result.get("pages", []),
            "full_text": full_text,
            "source": "google-cloud-vision",
            "filename": file.filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        print(f"Google Cloud Vision OCR error: {error_msg}")
        task_store.update_task(task_id, status="error", details={"error": error_msg})
        raise HTTPException(
            status_code=500,
            detail=f"Google Cloud Vision OCR failed: {error_msg}"
        )
    finally:
        try:
            ocr_limiter.release_request(client_ip)
        except Exception:
            pass

@router.get("/vision-health")
async def vision_health():
    """
    Health check endpoint for Google Cloud Vision API.
    Tests if API key is properly configured.
    
    Returns:
        {
            "status": "healthy" | "error",
            "credentials_configured": bool,
            "message": str
        }
    """
    import os
    api_key = os.getenv("OCR_KEY")
    if api_key:
        return {
            "status": "healthy",
            "credentials_configured": True,
            "message": "Google Cloud Vision API key is configured"
        }
    else:
        return {
            "status": "error",
            "credentials_configured": False,
            "message": "OCR_KEY environment variable not set"
        }
