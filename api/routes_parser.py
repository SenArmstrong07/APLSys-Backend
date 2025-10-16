from fastapi import APIRouter, UploadFile, File, Query, Request
import pandas as pd
from doctr.io import DocumentFile
import matplotlib.pyplot as plt
from services.parsing_service import (
    extract_tables_from_pdf,
    extract_tables_with_camelot,
    export_tables_to_csv,
    parse_document_text,
    # map_entities_to_profile,
    # merge_ner_entities
)
from model.request_schema import DocumentParseRequest, ResumeParseRequest
from typing import Optional

router = APIRouter()
    
@router.post("/parse-document")
async def parse_document(request: DocumentParseRequest, fastapi_req: Request):
    ner_pipeline = fastapi_req.app.state.general_ner_pipeline
    return parse_document_text(request.text, ner_pipeline)

# @router.post("/label-tokens-resume")
# async def label_tokens_resume(request: ResumeParseRequest, fastapi_req: Request):
#     ner_pipeline = fastapi_req.app.state.ner_resume_pipeline
#     raw_entities = ner_pipeline(request.text)
#     for entity in raw_entities:
#         if "score" in entity:
#             entity["score"] = float(entity["score"])
#     merged_entities = merge_ner_entities(raw_entities)
#     profile = map_entities_to_profile(merged_entities)
#     return {"profile": profile, "entities": merged_entities}

@router.post("/tabula_extract")
async def tabula_extract(file: UploadFile = File(...), pages: Optional[str] = Query("all")):
    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    tables = extract_tables_from_pdf(temp_path, pages=pages if pages is not None else "all")
    # Convert tables to JSON for API response
    tables_json = []
    for table in tables:
        if isinstance(table, pd.DataFrame):
            tables_json.append(table.to_dict(orient="records"))
        else:
            tables_json.append(table)  # fallback if not a DataFrame
    return {"tables": tables_json}

@router.post("/camelot_extract")
async def camelot_extract(file: UploadFile = File(...), pages: Optional[str] = Query("all")):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    tables = extract_tables_with_camelot(temp_path, pages=pages if pages is not None else "all")
    tables_json = [table.to_dict(orient="records") for table in tables]
    return {"tables": tables_json}

@router.post("/export_csv")
async def export_csv(file: UploadFile = File(...), method: str = Query("camelot"), base_filename: Optional[str] = Query("table"), pages: Optional[str] = Query("all")):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    if method == "tabula":
        tables = extract_tables_from_pdf(temp_path, pages=pages if pages is not None else "all")
    else:
        tables = extract_tables_with_camelot(temp_path, pages=pages if pages is not None else "all")
    safe_base_filename = base_filename if base_filename is not None else "table"
    export_tables_to_csv(tables, base_filename=safe_base_filename)
    return {"message": f"Exported {len(tables)} tables to CSV with base filename '{safe_base_filename}'."}


@router.post("/extract-resume-text")
async def extract_resume_txt(request: Request, file: UploadFile = File(...)):
    print("File received. Waiting for extraction...")
    plain_text: str = ""
    # Access your DocTR model
    model = request.app.state.ocr_model

    # Read uploaded file into bytes
    file_bytes = await file.read()

    # Load PDF or image into DocTR
    doc = DocumentFile.from_pdf(file_bytes)
    # Run OCR prediction
    result = model(doc)
    
    exported= result.export()
    
    for page in exported["pages"]:
        for block in page["blocks"]:
            for line in block["lines"]:
                line_text = " ".join(word["value"] for word in line["words"])
                plain_text += line_text + " \n"
    
    #Visualize result
    # result.show()
    # plt.show()
    # Return extracted text or structure
    return plain_text