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
from utils.mem_bar import memory_bar
load_dotenv()

# --- limit native BLAS/OMP threads to reduce memory/CPU pressure on startup ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

MODEL_LOCK = threading.Lock()

    
@lru_cache(maxsize=1)
def get_resume_parser():
    """Load general NER model only when first requested"""
    from transformers import pipeline
    with MODEL_LOCK:
        print("Loading resume parser model (lazy)...")
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024
        model_res_parser = "DeezNutz1337/Resume-Parser-BERT_Based"
        p = pipeline(task="token-classification", model=model_res_parser, aggregation_strategy="simple", device=-1)
        mem_after = psutil.Process().memory_info().rss / 1024 / 1024
        model_size = max(0.0, mem_after - mem_before)
        try:
            memory_bar.register("resume_ner", round(model_size, 1))
        except Exception:
            pass
        return p

def cleanup_resources():
    """Cleanup before shutdown"""
    import gc
    gc.collect()
    try:
        memory_bar.clear_all()
    except Exception:
        pass
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
    # Set PyTorch threading limits if torch exists
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        print("Torch threads limited to 1")
    except Exception:
        pass

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

app.state.get_ner_resume_pipeline = get_resume_parser

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
