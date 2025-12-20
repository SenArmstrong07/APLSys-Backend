from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException, Request, status, Response
from model.request_schema import ClassifyRequest, ResumeAnalysisRequest
import requests
from os import getenv
import time
import threading
import re
import asyncio
import random
from dotenv import load_dotenv
from model.request_schema import ResumeAnalysisRequest
from utils.img_to_b64 import image_to_base64
from typing import List, Optional
from PyPDF2 import PdfReader
import io
import json
from model.request_schema import ResumeTextRequest, TextRequest
from services.ai_service import (
    gemini_extract_resume_profile,
    deepseek_extract_metadata_from_text,
    validate_resume_text
)
from utils.task_store import TaskStore
from utils.openrouter_client import client, OPENROUTER_API_KEY, OPENROUTER_MODEL
router = APIRouter()
task_store = TaskStore()
load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_API_KEY = getenv("GEMINI_API_KEY")

GEMINI_RATE_LIMIT_LOCK = threading.Lock()
GEMINI_LAST_CALL_TIME = 0.0
GEMINI_MIN_INTERVAL = 3.0  # minimum seconds between Gemini calls (tweakable)


#- **Current Skills**: [List ALL skills the candidate demonstrates in their resume, categorized by type (technical, soft, domain-specific, etc.). Be comprehensive.]

def check_gemini_rate_limit() -> tuple:
    """
    Global limiter: ensures a minimum interval between Gemini requests.
    Returns (allowed: bool, wait_time_seconds: float|None)
    """
    global GEMINI_LAST_CALL_TIME
    now = time.time()
    with GEMINI_RATE_LIMIT_LOCK:
        elapsed = now - GEMINI_LAST_CALL_TIME
        if elapsed < GEMINI_MIN_INTERVAL:
            return False, GEMINI_MIN_INTERVAL - elapsed
        GEMINI_LAST_CALL_TIME = now
        return True, None
    
    
async def call_gemini_with_retries(text: str, attempts: int = 3, base_delay: float = 1.0, max_delay: float = 8.0) -> dict:
    """
    Call synchronous `gemini_extract_resume_profile` in a thread with retries,
    exponential backoff and jitter. Returns the dict result or an error dict.
    """
    last_exc = None
    for attempt in range(1, attempts + 1):
        # Global cooldown check
        allowed, wait = check_gemini_rate_limit()
        if not allowed:
            await asyncio.sleep(wait)

        try:
            # run blocking gemini call off the event loop
            result = await asyncio.to_thread(gemini_extract_resume_profile, text)
            # If result is dict and not an empty dict and not containing "error", treat as success
            if isinstance(result, dict) and result and "error" not in result:
                return result
            # If model returned an explicit error, propagate for retry
            last_exc = result if isinstance(result, dict) else {"error": "Unknown non-dict Gemini response"}
        except Exception as e:
            last_exc = {"error": str(e)}

        # If not last attempt, backoff with jitter
        if attempt < attempts:
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            jitter = random.uniform(0, delay * 0.5)
            await asyncio.sleep(delay + jitter)

    # Exhausted retries: return last error/result
    return last_exc or {"error": "gemini_failed_unknown"}



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

    base_prompt += f"""
## Overall Assessment
In up to 2 sentences, give a direct hiring recommendation based on the resume, the role (if provided), and the job description (if provided). Clearly state whether the candidate *should* or *should not* be hired and briefly justify why.
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
async def gemini_extract_resume_profile_endpoint(req: ResumeTextRequest, request: Request):
    """Extract a structured resume profile using Gemini fallback and OpenRouter as fallback."""
    if not validate_resume_text(req.text):
        return {"error": "Text does not appear to be a resume"}

    task_id = task_store.create_task(
        task_type="gemini_resume_extract",
        details={"text_length": len(req.text)}
    )

    try:
        task_store.update_task(task_id, status="processing")

        # 1) Primary: Gemini
        # try:
        #     result = gemini_extract_resume_profile(req.text)
        # except Exception as e:
        #     result = {"error": f"gemini_call_failed: {str(e)}"}
        
        result = await call_gemini_with_retries(req.text, attempts=3, base_delay=1.0, max_delay=8.0)

        # If Gemini returned an explicit error or empty dict -> fallback to OpenRouter
        if not isinstance(result, dict) or ("error" in result) or (isinstance(result, dict) and not result):
            # Build a compact JSON-only prompt for OpenRouter to extract a resume profile
            prompt = (
                "You are an expert resume extractor. Return EXACTLY one JSON object and NOTHING ELSE.\n"
                "Fields required (use these keys exactly):\n"
                "profile: {firstName, middleName, lastName, email, phone, location, summary},\n"
                "educations: [{school, degree, startDate, endDate, details}],\n"
                "work_experiences: [{company, title, startDate, endDate, descriptions}],\n"
                "skills: [strings],\n"
                "certifications: [strings]\n\n"
                f"Resume Text:\n{req.text}\n\n"
                "Return only the JSON object."
            )

            try:
                # Use OpenRouter client (same style as analyze-resume)
                extra_headers = {}
                if OPENROUTER_API_KEY is None:
                    raise RuntimeError("OPENROUTER_API_KEY not set")

                completion = client.chat.completions.create(
                    extra_headers=extra_headers,
                    extra_body={},
                    model=OPENROUTER_MODEL,
                    messages=[
                        {"role": "system", "content": "You are an expert resume extractor. RETURN JSON ONLY."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=800,
                )
                text = completion.choices[0].message.content or ""
                cleaned = clean_model_artifacts(text)
                # Try to extract JSON object from cleaned text
                start = cleaned.find("{")
                end = cleaned.rfind("}") + 1
                if start == -1 or end == 0:
                    # both Gemini and OpenRouter failed to return parseable JSON
                    task_store.update_task(task_id, status="error", details={"error": "No JSON from Gemini or OpenRouter", "gemini": result, "openrouter_raw": cleaned[:1000]})
                    return {"error": "Failed to extract structured resume (no JSON returned)", "task_id": task_id, "details": {"gemini": result, "openrouter_raw": cleaned[:1000]}}
                parsed = json.loads(cleaned[start:end])
                task_store.update_task(
                    task_id,
                    status="completed",
                    details={"profile_sections": len(parsed) if isinstance(parsed, dict) else 0, "source": "openrouter_fallback"}
                )
                print("DEBUG GEMINI RESULT (fallback -> openrouter):", parsed)
                return {**parsed, "task_id": task_id, "source": "openrouter_fallback"}
            except Exception as e:
                task_store.update_task(task_id, status="error", details={"error": str(e)})
                raise
        else:
            # Gemini succeeded
            task_store.update_task(
                task_id,
                status="completed",
                details={"profile_sections": len(result) if isinstance(result, dict) else 0, "source": "gemini"}
            )
            print("DEBUG GEMINI RESULT:", result)
            return {**result, "task_id": task_id, "source": "gemini"}
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
        retry = int(retry_after or 0)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {retry}s.",
            headers={"Retry-After": str(retry)})
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
        "HTTP-Referer": getenv("OPENROUTER_REFERER", "http://localhost:3000"),
        "X-Title": getenv("OPENROUTER_TITLE", "Resume Analyzer"),
    }

    try:
        task_store.update_task(task_id, status="processing")
        completion = client.chat.completions.create(
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
        

    
# Default tag set (frontend DEFAULT_TAGS can override by sending tags in request)
DEFAULT_TAGS = [
  'name','full_name','first_name','last_name','middle_name',
  'date_of_birth','gender','age','nationality',
  'address','location','city','country','postal_code',
  'phone','mobile_number','email',
  'id_number','passport_number','license_number',
  'skills','experience','years_of_experience',
  'education','degree','field_of_study',
  'certifications','organization','position','job_title',
  'achievements','projects','languages','references',
  'invoice_number','receipt_number','transaction_id',
  'purchase_order','vendor_name','customer_name','company_name','business_name','tax_id',
  'subtotal','total_amount','amount_due','amount_paid','discount','tax','vat_number','currency','payment_method','issue_date','due_date',
  'account_number','bank_name','branch_code','iban','swift_code','balance','statement_period','policy_number','contract_number','signature','authorization','terms_and_conditions',
  'date','time','document_type','reference_number','barcode','qrcode','website','url','notes','remarks','misc'
]

def reduce_tokens(text: str, max_chars: int = 2000) -> Optional[str]:
    """
    Best-effort reduce token payload by collapsing whitespace and keeping head+tail.
    Returns reduced text or None if reduction still too large.
    """
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    head_len = int(max_chars * 0.6)
    tail_len = max_chars - head_len
    candidate = cleaned[:head_len].rstrip() + " " + cleaned[-tail_len:].lstrip()
    if len(candidate) <= max_chars:
        return candidate
    # final aggressive truncation
    trunc = cleaned[:max_chars]
    return trunc if len(trunc) <= max_chars else None

@router.post("/classify")
async def classify_text(req: ClassifyRequest):
    """
    Classify text into one of the provided tags (or DEFAULT_TAGS).
    Attempts to reduce token size; if reduction fails or the external call fails, respond 204 No Content.
    """
    tags = req.tags if req.tags else DEFAULT_TAGS
    try:
        max_chars = int(getenv("CLASSIFY_MAX_CHARS", "2000"))
    except Exception:
        max_chars = 2000

    reduced = reduce_tokens(req.text or "", max_chars=max_chars)
    if reduced is None:
        # couldn't reduce tokens to acceptable size -> return nothing
        return Response(status_code=204)

    prompt = (
        "Classify the following text into one of these tags: "
        + ", ".join(tags)
        + ". Return only the tag.\n"
        f"Text: \"{reduced}\""
    )

    url = f"{BASE_URL}/models/{GEMINI_MODEL}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
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
    except Exception:
        # External call failed -> return nothing per requirement
        return Response(status_code=204)

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
