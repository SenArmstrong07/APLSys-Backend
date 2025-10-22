from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from api import routes_ocr, routes_ai, routes_debug, routes_parser
from doctr.models import ocr_predictor
from transformers import pipeline
import uvicorn as uv
import os


# Load models once and store in app state
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading OCR model...")
    app.state.ocr_model = ocr_predictor(
        det_arch="db_resnet50", #Fastest detection model
        reco_arch="vitstr_base", #Most accurate recognition model
        pretrained=True,
        assume_straight_pages=False,  # Better for receipts
        straighten_pages=True,        # Automatically straighten skewed images
        detect_orientation=True       # Detect and correct orientation
    )

    print("Loading NER models...")
    # For Resume Parsing
    #tokenizer_resume = AutoTokenizer.from_pretrained("./model/resume-ner-model")
    #model_resume = "./model/resume-ner-model"
    #ner_resume_pipeline = pipeline(task ="token-classification", model=model_resume, aggregation_strategy="simple")
    
    #For General OCR usage
    #tokenizer_general = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    model_general = "dbmdz/bert-large-cased-finetuned-conll03-english"
    general_ner_pipeline = pipeline(task = "token-classification", model=model_general, aggregation_strategy="simple")

    #app.state.ner_resume_pipeline = ner_resume_pipeline
    app.state.general_ner_pipeline = general_ner_pipeline
    print("Models ready!")
    
    yield
    
    print("Shutting down...")

app = FastAPI(title="APLSys Backend", lifespan=lifespan)

app.add_middleware(CORSMiddleware,
               allow_origins=["*"],
               allow_credentials=True,
               allow_methods=["*"],
               allow_headers=["*"]
               )


# Register API routes
app.include_router(routes_ocr.router, prefix="/ocr", tags=["OCR"])
app.include_router(routes_ai.router, prefix="/ai", tags=["AI"])
app.include_router(routes_debug.router, prefix="/debug", tags=["DEBUG"])
app.include_router(routes_parser.router, prefix="/parser", tags=["PARSER"])

@app.get("/")
def root():
    return {"message": "Backend is running"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uv.run("main:app", host="127.0.0.1", port=8000, reload=True)