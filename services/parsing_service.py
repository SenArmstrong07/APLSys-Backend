import layoutparser as lp
from camelot.io import read_pdf as camelot_read_pdf
from tabula.io import read_pdf
from docx import Document
import pandas as pd
import re

def extract_email(text):
    match = re.search(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", text)
    return match.group() if match else None

def extract_phone(text):
    match = re.search(r"\+?\d[\d\s\-\(\)]{7,}\d", text)
    return match.group() if match else None

def merge_ner_entities(entities):
    merged = []
    prev = None
    for ent in entities:
        label = ent.get("entity_group") or ent.get("entity")
        word = ent["word"]
        # Remove BERT subword prefix
        if word.startswith("##") or word.startswith("###"):
            word = word[2:]
            if prev and prev["entity"] == label:
                prev["word"] += word
                prev["end"] = ent["end"]
                prev["score"] = max(prev["score"], ent["score"])
                continue
        # Merge if same entity and adjacent
        if prev and prev["entity"] == label and ent["start"] == prev["end"]:
            prev["word"] += word
            prev["end"] = ent["end"]
            prev["score"] = max(prev["score"], ent["score"])
        else:
            if prev:
                merged.append(prev)
            prev = {
                "entity": label,
                "word": word,
                "start": ent["start"],
                "end": ent["end"],
                "score": ent["score"],
            }
    if prev:
        merged.append(prev)
    return merged

def map_entities_to_profile(entities, full_text=""):
    """
    Map flat NER entities to structured resume fields.
    """
    profile = {
        "first_name": "",
        "middle_name": "",
        "last_name": "",
        "age": "",
        "gender": "",
        "email": "",
        "phone": "",
        "location": "",
    }

    # --- Improved Name Parsing ---
    name_entities = [e["word"] for e in entities if e["entity"] == "NAME"]
    if name_entities:
        # Join all name entities and split
        name_str = " ".join(name_entities).replace("  ", " ").strip()
        name_parts = name_str.split()
        if len(name_parts) == 1:
            profile["first_name"] = name_parts[0]
        elif len(name_parts) == 2:
            profile["first_name"], profile["last_name"] = name_parts
        elif len(name_parts) > 2:
            profile["first_name"] = name_parts[0]
            profile["middle_name"] = " ".join(name_parts[1:-1])
            profile["last_name"] = name_parts[-1]

    # --- Email ---
    email_entity = next((e for e in entities if e["entity"] == "EMAIL"), None)
    if email_entity:
        profile["email"] = email_entity["word"]

    # --- Phone ---
    phone_entity = next((e for e in entities if e["entity"] == "PHONE"), None)
    if phone_entity:
        profile["phone"] = phone_entity["word"]

    # --- Address/Location ---
    address_entity = next((e for e in entities if e["entity"] in ("ADDRESS", "LOCATION")), None)
    if address_entity:
        profile["location"] = address_entity["word"]
    else:
        # Fallback: regex search for 'Brgy.' line in full_text
        import re
        match = re.search(r"(Brgy\.[^\n]+)", full_text, re.IGNORECASE)
        if match:
            profile["location"] = match.group(1).strip()
    return profile


# NER labeling for resumes
def lbl_resume_text(text: str, ner_pipeline):
    result = ner_pipeline(text)
    # Convert numpy.float32 to float for JSON serialization
    for entity in result:
        if "score" in entity:
            entity["score"] = float(entity["score"])
    merged_entities = merge_ner_entities(result)
    return {"entities": merged_entities}

#General use, digital documents
def parse_document_text(text: str, ner_pipeline):
    """
    Parse general digital document text using a NER pipeline.
    Returns extracted entities with their entity_group for debugging.
    """
    result = ner_pipeline(text)
    entities = []
    for entity in result:
        if "score" in entity:
            entity["score"] = float(entity["score"])
        # Add entity_group and word for debugging
        print(entity.get("entity_group"))
        entities.append({
            "entity": entity.get("word", ""),
            "entity_group": entity.get("entity_group", ""),
            "score": entity.get("score", 0)
        })
    return {"entities": entities}

def extract_tables_from_pdf(pdf_path, pages="all"):
    """
    Extract tables from a PDF using tabula-py.
    Returns a list of pandas DataFrames, one for each table found.
    """
    # Read tables from PDF
    tables = read_pdf(pdf_path, pages=pages, multiple_tables=True)
    return tables

def extract_tables_from_pdf_with_camelot(pdf_path, pages="all"):
    """
    Extract tables from a PDF using Camelot.
    Returns a list of pandas DataFrames, one for each table found.
    """
    tables = camelot_read_pdf(pdf_path, pages=pages)
    dataframes = [table.df for table in tables]
    return dataframes

def extract_tables_from_docx_with_camelot(docx_path):
    """
    Extract tables from a .docx file using python-docx.
    Returns a list of pandas DataFrames, one for each table found.
    """
    doc = Document(docx_path)
    dataframes = []
    for table in doc.tables:
        rows = []
        for row in table.rows:
            # get text for each cell, strip whitespace
            cells = [cell.text.strip() for cell in row.cells]
            rows.append(cells)
        if not rows:
            continue
        # normalize row lengths (pad shorter rows with empty strings)
        max_cols = max(len(r) for r in rows)
        normalized = [r + [""] * (max_cols - len(r)) for r in rows]
        df = pd.DataFrame(normalized)
        dataframes.append(df)
    return dataframes
# ...existing code...

def export_tables_to_csv(tables, base_filename="table"):
    """
    Export a list of pandas DataFrames to CSV files.
    Each table will be saved as base_filename_{idx+1}.csv
    """
    for idx, table in enumerate(tables):
        filename = f"{base_filename}_{idx+1}.csv"
        table.to_csv(filename, index=False)
        print(f"Exported: {filename}")
        
def export_tables_to_excel(tables, base_filename="table"):
    """
    Export a list of pandas DataFrames to CSV files.
    Each table will be saved as base_filename_{idx+1}.excel
    """
    for idx, table in enumerate(tables):
        filename = f"{base_filename}_{idx+1}.xlsx"
        table.to_excel(filename, index=False)
        print(f"Exported: {filename}")
    