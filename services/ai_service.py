import requests
import os
from dotenv import load_dotenv
from utils.img_to_b64 import image_to_base64

# Load environment variables
load_dotenv()

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODEL = "gemini-2.5-pro"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


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
    
def gemini_extract_metadata_from_text(extracted_text: str) -> dict:
    """
    Use Gemini to extract and label structured metadata from OCR-extracted text,
    using only the allowed tags from DEFAULT_TAGS.
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
        "You are an expert document parser. Given the following extracted text from a document, "
        "extract all relevant metadata, key-value pairs, and any structured information you can infer. "
        "Only use the following tags as keys in your JSON output:\n"
        f"{tags_str}\n"
        "Do not invent new keys or use keys outside this list. "
        "Do not use any other keys (such as ORG, MISC, PER, LOC, etc). "
        "If a value is not present, use an empty string. "
        "Return your answer as a flat JSON object with only these allowed tags as keys.\n"
        f"Extracted Text:\n{extracted_text}\nJSON:"
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
    text = (
        data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
    )
    import json
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        metadata_json = json.loads(text[start:end])
        return metadata_json
    except Exception:
        return {"error": "Failed to parse Gemini response", "raw": text}
    

