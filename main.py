from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from api import routes_ocr, routes_ai, routes_debug, routes_parser
import uvicorn as uv
import os
from functools import lru_cache
import threading
import psutil
import signal
try:
    import certifi
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:
    pass
from fastapi import FastAPI
from dotenv import load_dotenv
load_dotenv()

MODEL_LOCK = threading.Lock()

# TrOCR models
@lru_cache(maxsize=1)
def get_trocr_printed():
    """Load TrOCR model for printed text"""
    from transformers import pipeline
    with MODEL_LOCK:
        print("Loading TrOCR model (printed)...")
        return pipeline(task="image-to-text", model="microsoft/trocr-large-printed")

@lru_cache(maxsize=1)
def get_trocr_handwritten():
    """Load TrOCR model for handwritten text"""
    from transformers import pipeline
    with MODEL_LOCK:
        print("Loading TrOCR model (handwritten)...")
        return pipeline(task="image-to-text", model="microsoft/trocr-large-handwritten")
    
@lru_cache(maxsize=1)
def get_resume_parser():
    """Load general NER model only when first requested"""
    from transformers import pipeline
    with MODEL_LOCK:
        print("Loading resume parser model (lazy)...")
        model_res_parser = "DeezNutz1337/Resume-Parser-BERT_Based"
        return pipeline(task="token-classification", model=model_res_parser, aggregation_strategy="simple")

def cleanup_resources():
    """Cleanup before shutdown"""
    import gc
    gc.collect()
    print("Resources cleaned up")
    
def signal_handler(sig, frame):
    cleanup_resources()
    exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("FastAPI startup - Models will load on first use (lazy loading)")
    print(f"Initial memory: {psutil.Process().memory_info().rss / 1024 / 1024:.1f}MB")
    
    yield
    
    print("Shutting down...")
    cleanup_resources()

app = FastAPI(title="APLSys Backend", lifespan=lifespan)
app.add_middleware(CORSMiddleware,
               allow_origins=["*"],
               allow_credentials=True,
               allow_methods=["*"],
               allow_headers=["*"]
               )

# Store model getters in app state
app.state.get_trocr_printed = get_trocr_printed
app.state.get_trocr_handwritten = get_trocr_handwritten
app.state.get_ner_resume_pipeline = get_resume_parser
# Note: OCR is no longer preloaded here. Doctr OCR will be created lazily
# only when an AI request arrives and will be torn down after that request.
# See services.ocr_service.create_doctr_ocr and api.routes_ai.get_doctr_dependency

# Register API routes
app.include_router(routes_ocr.router, prefix="/ocr", tags=["OCR"])
app.include_router(routes_ai.router, prefix="/ai", tags=["AI"])
app.include_router(routes_debug.router, prefix="/debug", tags=["DEBUG"])
app.include_router(routes_parser.router, prefix="/parser", tags=["PARSER"])

@app.get("/")
def root():
    return {"message": "Backend is running"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    #uv.run("main:app", host="0.0.0.0", port=port, reload=True)
