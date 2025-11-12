from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException, Request, status
from model.request_schema import ClassifyRequest, ResumeAnalysisRequest
import requests
import os
import time
import threading
import re
from dotenv import load_dotenv
from model.request_schema import ResumeAnalysisRequest
from utils.img_to_b64 import image_to_base64
from typing import List, Optional
from PyPDF2 import PdfReader
import io
from model.request_schema import ResumeTextRequest, TextRequest
from services.ai_service import (
    gemini_extract_resume_profile,
    deepseek_extract_metadata_from_text,
    validate_resume_text
)
from utils.task_store import TaskStore
from utils.openrouter_client import openrouter, OPENROUTER_API_KEY, OPENROUTER_MODEL
router = APIRouter()
task_store = TaskStore()
load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


#- **Current Skills**: [List ALL skills the candidate demonstrates in their resume, categorized by type (technical, soft, domain-specific, etc.). Be comprehensive.]

# Add to both routes_parser.py and routes_ai.py
@router.get("/tasks")
async def get_tasks(status: Optional[str] = Query(None), limit: int = Query(100)):
    """Get status of parser/AI tasks"""
    return {"tasks": task_store.list_tasks(status=status, limit=limit)}

@router.get("/tasks/{task_id}")
async def get_task(task_id: int):
    """Get specific task status"""
    try:
        return {"task": task_store.get_task(task_id)}
    except KeyError:
        return {"error": "Task not found"}

def build_prompt(req: ResumeAnalysisRequest) -> str:
    """
    Build a concise resume analysis prompt.
    REQUIREMENTS:
    - Use exactly '##' (two hashes) for section headers. Do NOT use '###' or other markdown.
    - For each '##' section provide NO MORE THAN 2 sentences.
    - Keep lists short: max 3 items unless otherwise noted.
    - Be concise and focused; avoid extra commentary outside the sections.
    """
    base_prompt = f"""
Use ONLY '##' (two hashes) for section headers. Do NOT use '###', bold, italics, or any other markdown styles.
For each '##' section provide NO MORE THAN 2 sentences. Keep lists short (max 3 items). Do not add any commentary outside the sections.

## Overall Assessment
In up to 2 sentences, give a focused assessment of the resume's quality, strengths, and main weaknesses.

## Skills Analysis
- Skill Proficiency: In 1-2 sentences, summarize the candidate's apparent skill levels.
- Missing Skills: In 1-2 sentences, list the most critical missing skills (max 3 short phrases).

## Experience Analysis
In up to 2 sentences, evaluate how experience is presented (use of action verbs, metrics, relevance). End the section with a single-line score like: "Score: XX/100".

## Key Strengths
List up to 3 concise strengths (each 1 short phrase or sentence).

## Resume Score
Single-line: "Resume Score: XX/100"

"""
    base_prompt += f"\nResume Text:\n{req.resume}\n"
    if req.job_role:
        base_prompt += f"""
## Role Alignment Analysis
In up to 2 sentences, explain how the resume aligns with the role: {req.job_role} and give 1-2 focused recommendations.
"""
    if req.job_description:
        base_prompt += f"""
## Job Match Analysis
In up to 2 sentences, compare the resume to the job description and provide a job match percentage (0-100).

## Key Job Requirements Not Met
List up to 3 of the most critical missing requirements as short phrases and a 1-line suggestion for addressing them.
"""
    return base_prompt

# --- Simple in-memory rate limiter (per-IP) ---
RATE_LIMIT_LOCK = threading.Lock()
RATE_LIMIT_STORE = {}  # ip -> [timestamps]
RATE_LIMIT_MAX = 5     # requests
RATE_LIMIT_WINDOW = 60  # seconds

def check_rate_limit(client_ip: str):
    now = time.time()
    with RATE_LIMIT_LOCK:
        arr = RATE_LIMIT_STORE.get(client_ip, [])
        # drop old timestamps
        arr = [t for t in arr if now - t < RATE_LIMIT_WINDOW]
        if len(arr) >= RATE_LIMIT_MAX:
            # store back the cleaned arr (unchanged)
            RATE_LIMIT_STORE[client_ip] = arr
            return False, RATE_LIMIT_WINDOW - (now - arr[0])
        arr.append(now)
        RATE_LIMIT_STORE[client_ip] = arr
        return True, None

def clean_model_artifacts(text: str) -> str:
    """
    Remove Gemini/OpenRouter special artifacts such as:
      - angle-bracket markers: <...>
      - fullwidth-bar delimited tokens: ｜...｜
      - repeated underscore or low-line markers (▁, _)
    """
    if not text:
        return text
    # remove <...> tokens
    text = re.sub(r"<[^>]*>", "", text)
    # remove fullwidth-bar delimited tokens like '｜begin▁of▁sentence｜'
    text = re.sub(r"｜.*?｜", "", text)
    # replace repeated underscores / U+2581 with spaces
    text = re.sub(r"[_\u2581]+", " ", text)
    # collapse multiple spaces/newlines
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
    
@router.post("/gemini-extract-resume-profile")
async def gemini_extract_resume_profile_endpoint(req: ResumeTextRequest):
    """Extract a structured resume profile using Gemini fallback."""
    if not validate_resume_text(req.text):
        return {"error": "Text does not appear to be a resume"}
    
    task_id = task_store.create_task(
        task_type="gemini_resume_extract",
        details={"text_length": len(req.text)}
    )
    
    try:
        task_store.update_task(task_id, status="processing")
        result = gemini_extract_resume_profile(req.text)
        task_store.update_task(
            task_id, 
            status="completed",
            details={"profile_sections": len(result) if isinstance(result, dict) else 0}
        )
        print("DEBUG GEMINI RESULT:", result)
        return {**result, "task_id": task_id}
    except Exception as e:
        task_store.update_task(task_id, status="error", details={"error": str(e)})
        raise

@router.post("/batch-analyze-resumes")
async def batch_analyze_resumes(
    files: List[UploadFile] = File(...),
    job_role: Optional[str] = Form(None),
    job_description: Optional[str] = Form(None)
):
    batch_task_id = task_store.create_task(
        task_type="batch_resume_analysis",
        details={"file_count": len(files)}
    )
    task_store.update_task(batch_task_id, status="running")

    results = []
    for idx, file in enumerate(files, 1):
        file_task_id = task_store.create_task(
            task_type="resume_analysis",
            filename=file.filename,
            details={"batch_id": batch_task_id, "index": idx}
        )
        try:
            task_store.update_task(file_task_id, status="processing")
            # Read PDF text
            pdf_bytes = await file.read()
            reader = PdfReader(io.BytesIO(pdf_bytes))
            raw_text = "\n".join([page.extract_text() or "" for page in reader.pages])

            req = ResumeAnalysisRequest(
                resume=raw_text,
                job_role=job_role,
                job_description=job_description
            )
            prompt = build_prompt(req)

            # Call Gemini API (reuse your analyze logic)
            url = f"{BASE_URL}/models/{GEMINI_MODEL}:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": GEMINI_API_KEY
            }
            payload = {
                "contents": [
                    {
                        "parts": [
                            { "text": prompt }
                        ]
                    }
                ]
            }
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            text = (
                data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
            )
            # sanitize model artifacts before returning/storing
            text = clean_model_artifacts(text)
            
            task_store.update_task(
                file_task_id, 
                status="completed",
                progress=idx/len(files)
            )
            results.append({"name": file.filename, "result": text, "task_id": file_task_id})
        except Exception as e:
            task_store.update_task(file_task_id, status="error", details={"error": str(e)})
            results.append({"name": file.filename, "error": str(e), "task_id": file_task_id})

    task_store.update_task(batch_task_id, status="completed", progress=1.0)
    return {"results": results, "batch_task_id": batch_task_id}

@router.post("/analyze-resume")
async def analyze_resume(req: ResumeAnalysisRequest, request: Request):
    """
    Analyze resume using OpenRouter / DeepSeek chat completions.
    """
    client_ip = getattr(request.client, "host", "unknown")
    allowed, retry_after = check_rate_limit(client_ip)
    if not allowed:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail=f"Rate limit exceeded. Try again in {int(retry_after or 0)}s.")
    task_id = task_store.create_task(
        task_type="resume_analysis",
        details={
            "text_length": len(req.resume),
            "has_job_role": bool(req.job_role),
            "has_job_description": bool(req.job_description),
            "client_ip": client_ip
        }
    )

    if not OPENROUTER_API_KEY:
        task_store.update_task(task_id, status="error", details={"error": "OPENROUTER_API_KEY not set"})
        return {"error": "OPENROUTER_API_KEY not set in environment"}

    prompt = build_prompt(req)
    extra_headers = {
        "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "http://localhost:3000"),
        "X-Title": os.getenv("OPENROUTER_TITLE", "Resume Analyzer"),
    }

    try:
        task_store.update_task(task_id, status="processing")
        completion = openrouter.chat.completions.create(
            extra_headers=extra_headers,
            extra_body={},
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert resume analyzer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=800,
        )
        text = completion.choices[0].message.content
        # sanitize model artifacts before logging/returning
        text = clean_model_artifacts(text or "")
        task_store.update_task(
            task_id, 
            status="completed",
            details={"response_length": len(text) if text else 0}
        )
        print("DEBUG RESULT:", text)
        return {"result": text, "task_id": task_id}
    except Exception as e:
        task_store.update_task(task_id, status="error", details={"error": str(e)})
        return {"error": f"OpenRouter request failed: {str(e)}", "headers_sent": extra_headers}
        

    
@router.post("/classify")
async def classify_text(req: ClassifyRequest):
    prompt = (
        "Classify the following text into one of these tags: "
        "name, phone, email, education, address, skills, experience. "
        "Return only the tag.\n"
        f"Text: \"{req.text}\""
    )
    url = f"{BASE_URL}/models/{GEMINI_MODEL}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()
    tag = (
        data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
    )
    return {"tag": tag}
    
@router.post("/detect-table-layout")
def detect_table_layout(image_path):
    image_b64 = image_to_base64(image_path)
    url = f"{BASE_URL}/models/{GEMINI_MODEL}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }
    prompt = (
        "Analyze the following image and describe the table layout. "
        "List the number of tables, their positions (bounding boxes), and the number of rows and columns for each table. "
        "Respond in JSON format."
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",  # or "image/png"
                            "data": image_b64
                        }
                    }
                ]
            }
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

# @router.post("/gemini-extract-metadata")
# async def gemini_extract_metadata_endpoint(file: UploadFile = File(...)):
#     """
#     Extract and label structured metadata from any document image using Gemini.
#     Returns a JSON object with all detected fields, key-value pairs, and inferred structure.
#     """
#     # Save uploaded file to a temporary location
#     contents = await file.read()
#     temp_path = f"temp_{file.filename}"
#     with open(temp_path, "wb") as f:
#         f.write(contents)
#     try:
#         result = gemini_extract_metadata_from_image(temp_path)
#     finally:
#         import os
#         if os.path.exists(temp_path):
#             os.remove(temp_path)
#     return result

@router.post("/deepseek-label-extracted-text")
async def deepseek_label_extracted_text(req: TextRequest):
    """Use DeepSeek to label already extracted text with DEFAULT_TAGS."""
    task_id = task_store.create_task(
        task_type="deepseek_label",
        details={"text_length": len(req.text)}
    )
    
    try:
        task_store.update_task(task_id, status="processing")
        labeled = deepseek_extract_metadata_from_text(req.text)
        task_store.update_task(
            task_id, 
            status="completed",
            details={"label_count": len(labeled) if isinstance(labeled, dict) else 0}
        )
        print("DEBUG DEEPSEEK RESULT:", labeled)
        return {**labeled, "task_id": task_id}
    except Exception as e:
        task_store.update_task(task_id, status="error", details={"error": str(e)})
        raise

# Example usage:
if __name__ == "__main__":
    result = detect_table_layout("table3.jpg")
    print(result)
