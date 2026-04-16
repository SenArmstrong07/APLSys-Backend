import requests
import os
import json
from dotenv import load_dotenv
from google import genai
from google.auth import default
from google.auth.transport.requests import Request
from utils.img_to_b64 import image_to_base64
from utils.openrouter_client import client, OPENROUTER_MODEL, OPENROUTER_API_KEY
import time
import re

# Load environment variables
load_dotenv()

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_AUTH_SCOPE = os.getenv("GEMINI_AUTH_SCOPE", "https://www.googleapis.com/auth/cloud-platform")


def get_gcp_access_token(scopes=None) -> str:
    """Return a Google OAuth access token from ADC / Cloud Run service account."""
    scopes = scopes or [GEMINI_AUTH_SCOPE]
    credentials, _ = default(scopes=scopes)
    if not credentials.valid or credentials.expired:
        credentials.refresh(Request())
    return credentials.token


def get_gemini_headers() -> dict:
    """Build Gemini request headers using API key if provided, otherwise ADC bearer token."""
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        headers["x-goog-api-key"] = api_key
    else:
        token = get_gcp_access_token()
        headers["Authorization"] = f"Bearer {token}"
    return headers


def get_gemini_client():
    """Create a Gemini client using API key or ADC credentials."""
    credentials, _ = default()
    if not credentials.valid or credentials.expired:
        credentials.refresh(Request())
    return genai.Client(credentials=credentials)


def validate_resume_text(text: str) -> bool:
    """Basic validation to check if text looks like a resume"""
    resume_indicators = [
        "experience",
        "education",
        "skills",
        "objective",
        "qualifications"
    ]
    text_lower = text.lower()
    return any(indicator in text_lower for indicator in resume_indicators)

def gemini_extract_resume_profile(full_text: str, model_name=GEMINI_MODEL) -> dict:
    """
    Send a minimal / truncated payload to Gemini to extract a structured resume JSON.
    Implements retries on 503 and falls back to a smaller model if needed.
    """
    load_dotenv()
    model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"{BASE_URL}/models/{model}:generateContent"
    headers = get_gemini_headers()

    # Minimalization strategy:
    # - keep top-of-resume header (first 8 non-empty lines)
    # - keep the first 3000 chars afterwards (avoid sending entire file)
    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
    header = "\n".join(lines[:8]) if lines else full_text[:200]
    tail = full_text[:3000] if len(full_text) > 3000 else full_text
    minimized = f"{header}\n\n{tail}"

    prompt = (
        "Extract the resume into a compact JSON object with fields: profile (firstName, middleName, lastName, "
        "email, phone, location, summary), educations (school, degree, gpa, date), workExperiences (company, jobTitle, date, descriptions), "
        "skills (list). Return ONLY the JSON object, no explanation.\n\n"
        "Resume Text:\n"
        f"{minimized}\n\nJSON:"
    )

    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    # Try a few retries on 503 with exponential backoff
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            # if we got a response object, check status
            if resp.status_code == 503:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            data = resp.json()
            text = (
                data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
            )
            # Try to extract JSON blob from the response text
            try:
                start = text.find('{')
                end = text.rfind('}') + 1
                profile_json = json.loads(text[start:end])
                return profile_json
            except Exception:
                return {"error": "Failed to parse Gemini response", "raw": text}
        except requests.exceptions.HTTPError as e:
            # If a 503 was returned without body, try again; otherwise re-raise
            if resp is not None and resp.status_code == 503:
                time.sleep(2 ** attempt)
                continue
            raise e
        except requests.exceptions.RequestException:
            # network/timeout: backoff and retry
            time.sleep(2 ** attempt)
            continue

    # Fallback to a smaller model to reduce load / likelihood of 503
    fallback_model = "gemini-2.5-flash"
    try:
        url_fb = f"{BASE_URL}/models/{fallback_model}:generateContent"
        resp = requests.post(url_fb, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        text = (
            data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
        )
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            profile_json = json.loads(text[start:end])
            return profile_json
        except Exception:
            return {"error": "Failed to parse fallback Gemini response", "raw": text}
    except Exception:
        return {"error": "Gemini service unavailable. Please try again later."}
    
def deepseek_extract_metadata_from_text(extracted_text: str) -> dict:
    """
    Use OpenRouter / Deepseek (via OpenRouter API) to extract and label structured metadata
    from OCR-extracted text, using only the allowed tags from DEFAULT_TAGS.
    Returns a JSON object with all detected fields, key-value pairs, and inferred structure.
    This version sanitizes model artifacts and returns a clear error when no fields are detected.
    """
    allowed_tags = [
        'name', 'full_name', 'first_name', 'last_name', 'middle_name',
        'date_of_birth', 'gender', 'age', 'nationality',
        'address', 'location', 'city', 'country', 'postal_code',
        'phone', 'mobile_number', 'email',
        'id_number', 'passport_number', 'license_number',
        'skills', 'experience', 'years_of_experience',
        'education', 'degree', 'field_of_study',
        'certifications', 'organization', 'position', 'job_title',
        'achievements', 'projects', 'languages', 'references',
        'invoice_number', 'receipt_number', 'transaction_id',
        'purchase_order', 'vendor_name', 'customer_name',
        'company_name', 'business_name', 'tax_id',
        'subtotal', 'total_amount', 'amount_due', 'amount_paid',
        'discount', 'tax', 'vat_number', 'currency', 'payment_method',
        'issue_date', 'due_date',
        'account_number', 'bank_name', 'branch_code', 'iban', 'swift_code',
        'balance', 'statement_period', 'policy_number', 'contract_number',
        'signature', 'authorization', 'terms_and_conditions',
        'date', 'time', 'document_type', 'reference_number',
        'barcode', 'qrcode', 'website', 'url',
        'notes', 'remarks', 'misc'
    ]
    tags_str = ", ".join([f'"{tag}"' for tag in allowed_tags])
    prompt = (
        "You are an expert document parser. Given the following extracted text, "
        "identify and label any matching information using ONLY the allowed tags listed below. "
        "Return ONLY the tags and values that you actually find in the text - do not include empty fields. "
        "Return your answer as a minimal JSON object with only the detected fields.\n\n"
        f"Allowed tags:\n{tags_str}\n\n"
        f"Extracted Text:\n{extracted_text}\n\n"
        "Return only a JSON object with the detected fields. Do not include explanations or empty fields."
    )

    extra_headers = {}
    referer = os.getenv("OPENROUTER_REFERER")
    title = os.getenv("OPENROUTER_TITLE")
    if referer:
        extra_headers["HTTP-Referer"] = referer
    if title:
        extra_headers["X-Title"] = title

    if not OPENROUTER_API_KEY:
        return {"error": "OPENROUTER_API_KEY not set in environment"}

    try:
        completion = client.chat.completions.create(
            extra_headers=extra_headers,
            extra_body={},
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert document parser. Respond only with a JSON object using the allowed tags."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1500,
        )
        text = completion.choices[0].message.content
    except Exception as e:
        return {"error": f"OpenRouter request failed: {str(e)}"}

    # --- sanitize model artifacts (e.g. <...>, ｜...｜, U+2581 fragments) ---
    def _clean_model_text(t: str) -> str:
        if not isinstance(t, str):
            return ""
        t = re.sub(r"<[^>]+>", "", t)            # remove <...>
        t = re.sub(r"｜.*?｜", "", t)            # remove fullwidth-bar delimited tokens
        t = re.sub(r"[_\u2581]+", " ", t)       # replace underscores/U+2581 with spaces
        t = re.sub(r"\s{2,}", " ", t)           # collapse multiple spaces
        return t.strip()

    cleaned = _clean_model_text(text or " ")

    # parse JSON object from cleaned text
    try:
        if not cleaned:
            return {"error": "Empty response from model", "raw": text}
        start = cleaned.find('{')
        end = cleaned.rfind('}') + 1
        if start == -1 or end == 0:
            # return raw for debugging instead of silent empty dict
            return {"error": "No JSON object found in model response", "raw": cleaned[:1000]}
        parsed = json.loads(cleaned[start:end])

        # If parsed result is empty dict, return explicit message (helps frontend)
        if isinstance(parsed, dict) and not parsed:
            return {"error": "No fields detected", "raw": cleaned[:1500]}

        return parsed
    except Exception:
        return {"error": "Failed to parse OpenRouter response", "raw": cleaned[:1500]}

def gemini_classify_document(text: str, attempts: int = 3, base_delay: float = 1.0) -> dict:
    """
    Ask Gemini to classify document type. Returns a dict (expected keys: 'type', 'confidence', ...).
    On failure returns {'error': ...}
    """
    if not text:
        return {"error": "No text provided"}

    url = f"{BASE_URL}/models/{GEMINI_MODEL}:generateContent"
    headers = get_gemini_headers()

    prompt = (
        "Classify the type of the following document. "
        "Return EXACTLY one JSON object and NOTHING ELSE with keys: "
        "\"type\" (one of: invoice, receipt, resume, contract, bank_statement, passport, id_card, letter, bill, other), "
        "\"confidence\" (0.0-1.0), and optional \"labels\" (list).\n\n"
        f"Document Text:\n{text[:4000]}\n\nJSON:"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    last_exc = None
    for attempt in range(attempts):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            if resp.status_code == 503:
                time.sleep(base_delay * (2 ** attempt))
                continue
            resp.raise_for_status()
            data = resp.json()
            text_resp = (
                data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
            )
            start = text_resp.find("{")
            end = text_resp.rfind("}") + 1
            if start == -1 or end == 0:
                return {"error": "No JSON returned from Gemini", "raw": text_resp[:1000]}
            parsed = json.loads(text_resp[start:end])
            return parsed
        except Exception as e:
            last_exc = e
            time.sleep(base_delay * (2 ** attempt))
    return {"error": f"gemini_failed: {str(last_exc)}"}


