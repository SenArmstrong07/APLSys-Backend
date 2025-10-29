import requests
import os
import json
from dotenv import load_dotenv
from utils.img_to_b64 import image_to_base64
from utils.openrouter_client import openrouter, OPENROUTER_MODEL, OPENROUTER_API_KEY
# Load environment variables
load_dotenv()

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODEL = "gemini-2.5-pro"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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

def gemini_extract_resume_profile(full_text: str) -> dict:
    """
    Use Gemini to extract a structured resume profile from raw resume text.
    """
    prompt = (
        "Given the following resume text, extract all details into a JSON object with this structure:\n"
        "{\n"
        "  \"profile\": {\n"
        "    \"firstName\": \"\",\n"
        "    \"middleName\": \"\",\n"
        "    \"lastName\": \"\",\n"
        "    \"age\": \"\",\n"
        "    \"gender\": \"\",\n"
        "    \"email\": \"\",\n"
        "    \"phone\": \"\",\n"
        "    \"location\": \"\",\n"
        "    \"url\": \"\",\n"
        "    \"summary\": \"\"\n"
        "  },\n"
        "  \"educations\": [\n"
        "    {\"school\": \"\", \"degree\": \"\", \"gpa\": \"\", \"date\": \"\", \"descriptions\": \"\"}\n"
        "  ],\n"
        "  \"workExperiences\": [\n"
        "    {\"company\": \"\", \"jobTitle\": \"\", \"date\": \"\", \"descriptions\": \"\"}\n"
        "  ],\n"
        "  \"projects\": [\n"
        "    {\"project\": \"\", \"date\": \"\", \"descriptions\": \"\"}\n"
        "  ],\n"
        "  \"skills\": {\n"
        "    \"descriptions\": \"\",\n"
        "    \"featuredSkills\": [{\"skill\": \"\"}]\n"
        "  }\n"
        "}\n"
        "Fill in as much as possible from the resume. Use empty strings for missing fields. "
        "Resume Text:\n"
        f"{full_text}\n"
        "JSON:"
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
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        response.raise_for_status()
        data = response.json()
        text = (
            data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
        )
    except requests.exceptions.HTTPError as e:
        if response.status_code == 503:
            return {"error": "Gemini service unavailable. Please try again later."}
        raise e
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    import json
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        profile_json = json.loads(text[start:end])
        return profile_json
    except Exception:
        return {"error": "Failed to parse Gemini response", "raw": text}
    
def deepseek_extract_metadata_from_text(extracted_text: str) -> dict:
    """
    Use OpenRouter / Deepseek (via OpenRouter API) to extract and label structured metadata
    from OCR-extracted text, using only the allowed tags from DEFAULT_TAGS.
    Returns a JSON object with all detected fields, key-value pairs, and inferred structure.
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
        completion = openrouter.chat.completions.create(
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

    try:
        if not isinstance(text, str) or not text:
            return {"error": "Empty or non-text response from model", "raw": text}
        start = text.find('{')
        end = text.rfind('}') + 1
        if start == -1 or end == 0:
            return {"error": "No JSON object found in model response", "raw": text}
        return json.loads(text[start:end])
    except Exception:
        return {"error": "Failed to parse OpenRouter response", "raw": text}


