from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from api import routes_ocr, routes_ai, routes_debug, routes_parser
from transformers import pipeline
import uvicorn as uv
import os
from functools import lru_cache
import threading
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
    with MODEL_LOCK:
        print("Loading TrOCR model (printed)...")
        return pipeline(task="image-to-text", model="microsoft/trocr-large-printed")

@lru_cache(maxsize=1)
def get_trocr_handwritten():
    """Load TrOCR model for handwritten text"""
    with MODEL_LOCK:
        print("Loading TrOCR model (handwritten)...")
        return pipeline(task="image-to-text", model="microsoft/trocr-large-handwritten")

# NER models still load locally (they're smaller)
@lru_cache(maxsize=1)
def get_ner_pipeline_basic():
    """Load basic NER model only when first requested"""
    with MODEL_LOCK:
        print("Loading basic NER model (lazy)...")
        model_basic = "DeezNutz1337/Bert-Based-Resume-Profiler_BASIC"
        return pipeline(task="token-classification", model=model_basic, aggregation_strategy="simple")

@lru_cache(maxsize=1)
def get_ner_pipeline_semantic():
    """Load semantic NER model only when first requested"""
    with MODEL_LOCK:
        print("Loading semantic NER model (lazy)...")
        model_semantic = "DeezNutz1337/Bert-Based-Resume-Profiler_SEMANTIC"
        return pipeline(task="token-classification", model=model_semantic, aggregation_strategy="simple")

@lru_cache(maxsize=1)
def get_ner_pipeline_general():
    """Load general NER model only when first requested"""
    with MODEL_LOCK:
        print("Loading general NER model (lazy)...")
        model_general = "dbmdz/bert-large-cased-finetuned-conll03-english"
        return pipeline(task="token-classification", model=model_general, aggregation_strategy="simple")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("FastAPI startup - Models will load on first use (lazy loading)")
    print("OCR will use TrOCR (microsoft/trocr-large-printed and handwritten)")
    
    yield
    
    print("Shutting down...")

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
app.state.get_ner_resume_pipeline_basic = get_ner_pipeline_basic
app.state.get_ner_resume_pipeline_semantic = get_ner_pipeline_semantic
app.state.get_general_ner_pipeline = get_ner_pipeline_general

# Register API routes
app.include_router(routes_ocr.router, prefix="/ocr", tags=["OCR"])
app.include_router(routes_ai.router, prefix="/ai", tags=["AI"])
app.include_router(routes_debug.router, prefix="/debug", tags=["DEBUG"])
app.include_router(routes_parser.router, prefix="/parser", tags=["PARSER"])

@app.get("/")
def root():
    return {"message": "Backend is running"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uv.run("main:app", host="0.0.0.0", port=port, reload=True)
