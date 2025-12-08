# app/routers/ai_router.py
from fastapi import APIRouter
import requests
import os
from dotenv import load_dotenv
from services.ocr_service import get_memory_usage,create_doctr_ocr, dispose_doctr_ocr
import psutil
router = APIRouter()
load_dotenv()

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


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