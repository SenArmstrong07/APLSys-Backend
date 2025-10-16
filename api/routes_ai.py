from fastapi import APIRouter, UploadFile, File, Form
from model.request_schema import ClassifyRequest, ResumeAnalysisRequest
import requests
import os
from dotenv import load_dotenv
from model.request_schema import ResumeAnalysisRequest
from utils.img_to_b64 import image_to_base64
from typing import List, Optional
from PyPDF2 import PdfReader
import io
from model.request_schema import ResumeTextRequest, TextRequest
from services.ai_service import (
    gemini_extract_resume_profile,
    gemini_extract_metadata_from_text
)
router = APIRouter()
load_dotenv()

GEMINI_MODEL = "gemini-2.5-pro"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


#- **Current Skills**: [List ALL skills the candidate demonstrates in their resume, categorized by type (technical, soft, domain-specific, etc.). Be comprehensive.]

def build_prompt(req: ResumeAnalysisRequest) -> str:
    base_prompt = f"""
Please use only "##" for section headers and avoid using "###" or Markdown bold/italic formatting in your output.
And for each section, make your outputs concise but in 3 sentences max.
## Overall Assessment
[Provide a detailed & summarized assessment of the resume's overall quality, effectiveness, and alignment with industry standards. Include specific observations about formatting, content organization, and general impression. Be thorough and specific.]

## Skills Analysis

- **Skill Proficiency**: [Assess the apparent level of expertise in key skills based on how they're presented in the resume]
- **Missing Skills**: [List important skills that would improve the resume for their target role. Be specific and explain why each skill matters.]

## Experience Analysis
[Provide detailed feedback on how well the candidate has presented their experience. Analyze the use of action verbs, quantifiable achievements, and relevance to their target role. Suggest specific improvements. Afterwards, provide a score from 0-100 based on how well the experience section is presented: Resume Score: XX/100. Use this format exactly, where XX is the numerical score.]

## Key Strengths
[List 2-5 specific strengths of the resume with detailed explanations of why these are effective]

## Resume Score
[Provide a score from 0-100 based on the overall quality of the resume. Use this format exactly: "Resume Score: XX/100"]

Resume Data:
{req.resume}
"""
    if req.job_role:
        base_prompt += f"""
The candidate is targeting a role as: {req.job_role}

## Role Alignment Analysis
[Analyze how well the resume aligns with the target role of {req.job_role}. Provide specific recommendations to better align the resume with this role.]
"""

    if req.job_description:
        base_prompt += f"""
Additionally, compare this resume to the following job description:

Job Description:
{req.job_description}

## Job Match Analysis
[Provide a detailed analysis of how well the resume matches the job description, with a match percentage and specific areas of alignment and misalignment]

## Key Job Requirements Not Met
[List specific requirements from the job description that are not addressed in the resume, with recommendations on how to address each gap]
"""
    return base_prompt

@router.post("/gemini-extract-resume-profile")
async def gemini_extract_resume_profile_endpoint(req: ResumeTextRequest):
    """
    Extract a structured resume profile using Gemini fallback.
    """
    result = gemini_extract_resume_profile(req.text)
    return result

@router.post("/batch-analyze-resumes")
async def batch_analyze_resumes(
    files: List[UploadFile] = File(...),
    job_role: Optional[str] = Form(None),
    job_description: Optional[str] = Form(None)
):
    results = []
    for file in files:
        # Read PDF text
        pdf_bytes = await file.read()
        reader = PdfReader(io.BytesIO(pdf_bytes))
        raw_text = "\n".join([page.extract_text() or "" for page in reader.pages])

        # Build prompt (reuse your existing logic)
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
            results.append({"name": file.filename, "result": text})
        except Exception as e:
            results.append({"name": file.filename, "error": str(e)})
    return {"results": results}


@router.post("/analyze-resume")
async def analyze_resume(req: ResumeAnalysisRequest):
    url = f"{BASE_URL}/models/{GEMINI_MODEL}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }
    
    #build prompt
    prompt = build_prompt(req)
    
    payload = {
        "contents": [
            {
                "parts": [
                    { "text": prompt }
                ]
            }
        ]
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        # Parse response
        text = (
            data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
        )
        
        return {"result": text}
    except Exception as e:
        print("Gemini server error:", str(e), response.text if 'response' in locals() else "")
        return {"error": "Failed to analyze resume"}
    
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

@router.post("/gemini-label-extracted-text")
async def gemini_label_extracted_text(req: TextRequest):
    """
    Use Gemini to label already extracted text (from ocr.space) with DEFAULT_TAGS.
    """
    labeled = gemini_extract_metadata_from_text(req.text)
    return labeled

# Example usage:
if __name__ == "__main__":
    result = detect_table_layout("table3.jpg")
    print(result)
