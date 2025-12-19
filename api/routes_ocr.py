# app/routers/ocr_router.py
from fastapi import APIRouter, UploadFile, File, Query, Request, HTTPException, Depends, Form
from google.cloud import vision
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
    run_ocr_in_subprocess
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
    finally:
        # no local temp files here (cleanup handled elsewhere)
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

        # Read rotation and bbox from multipart form fields if present.
        try:
            # Prefer explicit Form params (handled by FastAPI). If not present, try parsing the raw form.
            def _read_from_parsed_form():
                nonlocal rotation, bbox_x, bbox_y, bbox_w, bbox_h
                # already set by Form defaults if provided; nothing to change
                return

            if bbox_x is None and bbox_y is None and bbox_w is None and bbox_h is None:
                # fallback: parse raw multipart form (compatible with older callers)
                form = await ocrreq.form()
                rotation_value = form.get("rotation", rotation)
                rotation = float(str(rotation_value)) if rotation_value else rotation

                def get_float(form, name: str) -> Optional[float]:
                    v = form.get(name)
                    if v is None:
                        return None
                    if isinstance(v, UploadFile):
                        return None
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return None

                if bbox_x is None: bbox_x = get_float(form, "bbox_x")
                if bbox_y is None: bbox_y = get_float(form, "bbox_y")
                if bbox_w is None: bbox_w = get_float(form, "bbox_w")
                if bbox_h is None: bbox_h = get_float(form, "bbox_h")

            # clamp helper
            def clamp01(x: float) -> float:
                return max(0.0, min(1.0, x))

            if None not in (bbox_x, bbox_y, bbox_w, bbox_h):
                bx_raw = cast(float, bbox_x)
                by_raw = cast(float, bbox_y)
                bw_raw = cast(float, bbox_w)
                bh_raw = cast(float, bbox_h)

                # If UI sent pixel coordinates (values > 1), normalize using the
                # rotated canvas dimensions that the frontend used (swap w/h for 90/270).
                bx = bx_raw
                by = by_raw
                bw = bw_raw
                bh = bh_raw
                try:
                    if max(bx_raw, by_raw, bw_raw, bh_raw) > 1.5:
                        import io as _io
                        from PIL import Image as PILImage
                        img = PILImage.open(_io.BytesIO(content))
                        img_w, img_h = img.size  # original (natural) image dims
                        rot = int(round(rotation)) % 360
                        if rot in (90, 270):
                            rot_w, rot_h = img_h, img_w
                        else:
                            rot_w, rot_h = img_w, img_h
                        if rot_w > 0 and rot_h > 0:
                            bx = bx_raw / rot_w
                            by = by_raw / rot_h
                            bw = bw_raw / rot_w
                            bh = bh_raw / rot_h
                except Exception:
                    # fallback: leave raw values (assume they were already normalized)
                    bx = bx_raw
                    by = by_raw
                    bw = bw_raw
                    bh = bh_raw

                bbox = {
                    "x": clamp01(bx),
                    "y": clamp01(by),
                    "width": clamp01(bw),
                    "height": clamp01(bh),
                }
            else:
                # NO bbox provided → treat upload as a client-side cropped image.
                # Do not re-orient or remap bytes; run OCR on entire uploaded image.
                bbox = None
        except Exception:
            rotation = rotation or 0.0
            bbox = None

        # Normalize rotation to canonical quarter-turn value (used only when remapping bbox)
        rotation = int(round(rotation)) % 360
        if rotation not in (0, 90, 180, 270):
            rotation = 0

        # If bbox is None we assume the client already cropped/straightened the image.
        # In that case do NOT call bbox_to_original and do not rotate the bytes on server.
        if bbox is None:
            bbox_orig = None
        else:
            # Convert incoming bbox (which is in UI rotation-space) back to original-image normalized coords
            bbox_orig = bbox_to_original(bbox, rotation)

        # NEW: If image is rotated and a CLOUD_VISION secret is configured, prefer Google Vision
        vision_key = getenv("CLOUD_VISION_API") or None
        if rotation != 0 and vision_key:
            try:
                # straighten image server-side, crop if bbox provided, then call Vision
                import io as _io
                from PIL import Image as PILImage

                img = PILImage.open(_io.BytesIO(content)).convert("RGB")
                straight = img.rotate(-rotation, expand=True) if rotation else img

                # If bbox provided, assume bbox is normalized (0..1) in UI rotation-space.
                if bbox is not None:
                    w, h = straight.size
                    sx = max(0, min(int(round(bbox["x"] * w)), w - 1))
                    sy = max(0, min(int(round(bbox["y"] * h)), h - 1))
                    sw = max(1, min(int(round(bbox["width"] * w)), w - sx))
                    sh = max(1, min(int(round(bbox["height"] * h)), h - sy))
                    crop = straight.crop((sx, sy, sx + sw, sy + sh))
                else:
                    crop = straight

                buf = _io.BytesIO()
                crop.save(buf, format="PNG", optimize=True)
                img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                
                client = vision.ImageAnnotatorClient()

                image = vision.Image(content=straight)

                response = client.annotate_image({
                    "image": image,
                    "features": [{"type_": vision.Feature.Type.TEXT_DETECTION}],
                })

                texts = response.text_annotations
                text = texts[0].description if texts else ""
                
                # try to compute a confidence if available in fullTextAnnotation pages/words
                conf = 0.0
                confidences = []
                for page in response.full_text_annotation.pages:
                    for block in page.blocks:
                        for paragraph in block.paragraphs:
                            for word in paragraph.words:
                                if word.confidence is not None:
                                    confidences.append(word.confidence)
                if confidences:
                    conf = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

                task_store.update_task(task_id, status="completed", details={"text_length": len(text)})
                return {"text": text or "", "confidence": round(conf, 2), "task_id": task_id}
            except Exception as e:
                # If Vision fails, fall back to existing DocTR worker path
                print("Google Vision path failed, falling back to DocTR:", e)

        # Submit to queue; worker returns text + confidence (existing flow)
        result = await ocr_queue.submit(_worker_process_region, content, file.filename or "uploaded", client_ip, bbox_orig)

        # Update task and return the extracted text with confidence
        task_store.update_task(
            task_id,
            status="completed",
            details={"text_length": len(text), "confidence": conf}
        )
        return {
            "text": text,
            "confidence": conf,
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
