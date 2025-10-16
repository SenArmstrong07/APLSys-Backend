# app/routers/ai_router.py
from fastapi import APIRouter
import requests
import os
from dotenv import load_dotenv

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