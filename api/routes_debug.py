# app/routers/ai_router.py
from fastapi import APIRouter
import requests
from os import getenv
import gc
import importlib
from dotenv import load_dotenv
from services.ocr_service import get_memory_usage,create_doctr_ocr, dispose_doctr_ocr
import psutil
from utils.task_store import TaskStore
from utils.mem_bar import memory_bar
router = APIRouter()
load_dotenv()

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_API_KEY = getenv("GEMINI_API_KEY")


#Check available models
@router.get("/list-models")
async def list_models():
    url = f"{BASE_URL}/models"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print("Error listing models:", str(e))
        return {"error": "Failed to fetch models"}
    
    
@router.post("/unload-models")
async def unload_models():
    """
    Unload lazy transformer pipelines (TroCR / NER) to free memory.
    Call this before /debug/calibrate-doctr-memory or before heavy OCR.
    """
    removed = []
    try:
        main_mod = importlib.import_module("main")
        # Clear lru_cache wrappers if present
        for fn in ("get_trocr_printed", "get_trocr_handwritten",
                   "get_ner_pipeline_basic", "get_ner_pipeline_semantic", "get_ner_pipeline_general"):
            f = getattr(main_mod, fn, None)
            if f and hasattr(f, "cache_clear"):
                try:
                    f.cache_clear()
                    removed.append(fn + ":cache_cleared")
                except Exception:
                    removed.append(fn + ":cache_clear_failed")
        # Remove any stored app.state references if present
        app = getattr(main_mod, "app", None)
        if app is not None:
            for key in ("get_trocr_printed", "get_trocr_handwritten",
                        "get_ner_resume_pipeline_basic", "get_ner_resume_pipeline_semantic", "get_general_ner_pipeline"):
                if hasattr(app.state, key):
                    try:
                        delattr(app.state, key)
                        removed.append(f"app.state.{key}:deleted")
                    except Exception:
                        removed.append(f"app.state.{key}:delete_failed")
        gc.collect()
        return {"status": "ok", "actions": removed, "process_rss_mb": round(get_memory_usage(), 1)}
    except Exception as e:
        return {"status": "error", "error": str(e), "actions": removed}
    
@router.get("/memory")
async def memory_info():
    """Return current process RSS (MB) and host virtual memory summary."""
    proc_rss_mb = get_memory_usage()
    vm = psutil.virtual_memory()
    return {
        "process_rss_mb": round(proc_rss_mb, 1),
        "virtual_total_mb": vm.total // (1024 * 1024),
        "virtual_available_mb": vm.available // (1024 * 1024),
        "virtual_used_percent": vm.percent
    }
    
@router.get("/calibrate-doctr-memory")
async def calibrate_doctr_memory():
    """
    Measure actual memory usage of DocTR model.
    Call this once to determine DOCTR_REQUIRED_MB and adjust ocr_service.py.
    """
    mem_before = get_memory_usage()
    predictor = None
    
    try:
        predictor = create_doctr_ocr()
        mem_peak = get_memory_usage()
        delta = mem_peak - mem_before
        
        return {
            "status": "success",
            "memory_before_mb": round(mem_before, 1),
            "memory_after_loading_mb": round(mem_peak, 1),
            "delta_mb": round(delta, 1),
            "recommended_required_mb": round(delta + 50, 1),  # +50MB safety margin
            "note": f"Update DOCTR_REQUIRED_MB to {round(delta + 50, 1)} in services/ocr_service.py"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "memory_before_mb": round(mem_before, 1)
        }
    finally:
        if predictor:
            try:
                dispose_doctr_ocr(predictor)
            except Exception:
                pass
    
#Check server status
@router.get("/health")
async def health_check():
    return {"status": "ok", "message": "Server is running"}

#Check if environment variables are loaded and models are accessible
@router.get("/env-check")
async def env_check():
    return {
        "env_loaded": os.getenv("GEMINI_API_KEY") is not None,
        "models": ["ocr", "ner", "gemini"]
    }
    
    

@router.get("/memory-bar")
async def memory_bar_status():
    """
    Return the current MemoryBar accounting (cap, used, percent, items).
    """
    try:
        return memory_bar.get_usage()
    except Exception as e:
        return {"error": str(e)}

@router.get("/last-ocr-peak")
async def last_ocr_peak():
    """
    Return peak RSS (MB) recorded for the most recent OCR task.
    Falls back to current process RSS if no per-task peak is available.
    """
    ts = TaskStore()
    try:
        tasks = ts.list_tasks(limit=200)
    except Exception:
        tasks = []

    # Find last OCR-related task
    last_ocr = None
    for t in reversed(tasks):
        ttype = (t.get("task_type") or "").lower()
        if "ocr" in ttype or t.get("task_type", "").startswith("ocr"):
            last_ocr = t
            break

    if last_ocr:
        details = last_ocr.get("details", {}) or {}
        # common memory keys we might have recorded
        peak = details.get("mem_peak_mb") or details.get("mem_after_mb") or details.get("mem_before_mb")
        return {
            "task_id": last_ocr.get("id"),
            "task_type": last_ocr.get("task_type"),
            "recorded_details": details,
            "mem_peak_mb": peak or None,
            "process_rss_mb": round(get_memory_usage(), 1)
        }

    # fallback: no OCR task found or no mem info recorded
    return {
        "task_id": None,
        "task_type": None,
        "recorded_details": None,
        "mem_peak_mb": None,
        "process_rss_mb": round(get_memory_usage(), 1),
        "note": "No OCR task with memory snapshot found; returning current process RSS."
    }

@router.get("/last-ner-peak")
async def last_ner_peak():
    """
    Return peak RSS (MB) recorded for the most recent NER task.
    Falls back to current process RSS if no peak recorded.
    """
    ts = TaskStore()
    try:
        tasks = ts.list_tasks(limit=200)
    except Exception:
        tasks = []

    # Find last NER-related task
    last_ner = None
    for t in reversed(tasks):
        ttype = (t.get("task_type") or "").lower()
        if "ner" in ttype or "resume" in ttype:
            last_ner = t
            break

    if last_ner:
        details = last_ner.get("details", {}) or {}
        peak = details.get("mem_peak_mb") or details.get("mem_after_mb") or details.get("mem_before_mb")
        return {
            "task_id": last_ner.get("id"),
            "task_type": last_ner.get("task_type"),
            "recorded_details": details,
            "mem_peak_mb": peak or None,
            "process_rss_mb": round(get_memory_usage(), 1)
        }

    return {
        "task_id": None,
        "task_type": None,
        "recorded_details": None,
        "mem_peak_mb": None,
        "process_rss_mb": round(get_memory_usage(), 1),
        "note": "No NER task with memory snapshot found; returning current process RSS."
    }