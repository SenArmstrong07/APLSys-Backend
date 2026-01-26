#from camelot.io import read_pdf as camelot_read_pdf
# from tabula.io import read_pdf
from docx import Document
import re
from typing import Dict, List, Any, cast
import os
import fitz
from doctr.io import DocumentFile
from typing import Optional
import asyncio
import tempfile
import json
import os
import sys
import psutil
from utils.json_encoder import convert_numpy_types

# try to import python-docx; keep optional to avoid hard failure at import time
try:
    from docx import Document as DocxDocument
except Exception:
    DocxDocument = None

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
def parse_document_text(path: str, ocr_model) -> str:
    """
    Extract raw text from a document file at `path`.
    Supported: .pdf (PyMuPDF), .docx (python-docx), image files (DocTR via ocr_model).
    Returns plain text (no further parsing).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    _, ext = os.path.splitext(path)
    ext = (ext or "").lower()

    def _clean_text(s: str) -> str:
        # join hyphenated line-breaks and normalize whitespace
        s = re.sub(r"-\s*\n\s*", "", s)
        s = "\n".join([ln.rstrip() for ln in s.splitlines()])
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()

    # PDF
    if ext == ".pdf":
        txt_parts = []
        pdf = fitz.open(path)
        for p in pdf:
            txt_parts.append(p.get_text("text"))
        pdf.close()
        return _clean_text("\n".join(str(p) for p in txt_parts))

    # DOCX
    if ext == ".docx":
        if DocxDocument is None:
            raise RuntimeError("python-docx not installed; cannot extract .docx text")
        try:
            doc = DocxDocument(path)
            paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
            if not paragraphs:
                # Try to extract from tables as fallback
                for table in doc.tables:
                    for row in table.rows:
                        for cell in row.cells:
                            if cell.text.strip():
                                paragraphs.append(cell.text)
            result = _clean_text("\n".join(paragraphs))
            if not result:
                raise ValueError("No text content found in DOCX document")
            return result
        except Exception as e:
            raise RuntimeError(f"DOCX parsing error: {e}")

    # Images (common raster image extensions)
    if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"):
        if ocr_model is None:
            raise RuntimeError("ocr_model must be provided for image text extraction (DocTR)")
        # Use DocTR DocumentFile.from_images and the provided predictor
        doc = DocumentFile.from_images([path])
        result = ocr_model(doc)
        exported = result.export()
        parts = []
        for page in exported.get("pages", []):
            for block in page.get("blocks", []):
                for line in block.get("lines", []):
                    # sort words left-to-right if geometry exists
                    words = line.get("words", []) or []
                    try:
                        words = sorted(words, key=lambda w: (min([pt[0] for pt in w.get("geometry", [])]) if w.get("geometry") else 0.0))
                    except Exception:
                        pass
                    parts.append(" ".join(w.get("value", "") for w in words).strip())
        return _clean_text("\n".join([p for p in parts if p]))

    # Unknown extension: attempt best-effort PDF/open fallback by trying PyMuPDF open
    try:
        pdf = fitz.open(path)
        txt_parts = [p.get_text("text") for p in pdf]
        pdf.close()
        return _clean_text("\n".join(str(p) for p in txt_parts))
    except Exception:
        raise RuntimeError(f"Unsupported file type or extraction failed for: {path}")

# def extract_tables_from_pdf(pdf_path, pages="all"):
#     """
#     Extract tables from a PDF using tabula-py.
#     Returns a list of pandas DataFrames, one for each table found.
#     """
#     # Read tables from PDF
#     tables = read_pdf(pdf_path, pages=pages, multiple_tables=True)
#     return tables

# def extract_tables_from_pdf_with_camelot(pdf_path, pages="all"):
#     """
#     Extract tables from a PDF using Camelot.
#     Returns a list of pandas DataFrames, one for each table found.
#     """
#     tables = camelot_read_pdf(pdf_path, pages=pages)
#     dataframes = [table.df for table in tables]
#     return dataframes

# def extract_tables_from_docx_with_camelot(docx_path):
#     """
#     Extract tables from a .docx file using python-docx.
#     Returns a list of pandas DataFrames, one for each table found.
#     """
#     doc = Document(docx_path)
#     dataframes = []
#     for table in doc.tables:
#         rows = []
#         for row in table.rows:
#             # get text for each cell, strip whitespace
#             cells = [cell.text.strip() for cell in row.cells]
#             rows.append(cells)
#         if not rows:
#             continue
#         # normalize row lengths (pad shorter rows with empty strings)
#         max_cols = max(len(r) for r in rows)
#         normalized = [r + [""] * (max_cols - len(r)) for r in rows]
#         df = pd.DataFrame(normalized)
#         dataframes.append(df)
#     return dataframes

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

def clean_and_structure_resume_text(text: str) -> Dict[str, str]:
    """
    Clean and structure resume text extracted from PDF.
    Handles layout noise, normalizes sections, and returns organized dict.
    """
    # Normalize newlines and collapse excessive whitespace
    text = re.sub(r"-\s*\n\s*", "", text)  # join hyphenated line breaks
    text = "\n".join([ln.strip() for ln in text.splitlines() if ln.strip()])
    
    # Common resume section headers (case-insensitive)
    section_patterns = {
        "contact": r"(?:contact|phone|email|address|location)",
        "about": r"(?:about|objective|summary|profile)",
        "education": r"(?:education|academic|schooling|degree)",
        "experience": r"(?:experience|work|employment|professional)",
        "skills": r"(?:skills|technical|competencies|expertise)",
        "projects": r"(?:projects|portfolio|work samples)",
        "certifications": r"(?:certifications?|licenses?|achievements?)",
        "languages": r"(?:languages?|linguistic)",
    }
    
    structured = {section: "" for section in section_patterns.keys()}
    structured["other"] = ""
    
    current_section = "other"
    lines = text.split("\n")
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Check if line is a section header
        matched = False
        for section, pattern in section_patterns.items():
            if re.search(pattern, line_lower) and len(line) < 50:  # likely a header
                current_section = section
                matched = True
                break
        
        if not matched:
            # Add line to current section
            structured[current_section] += line + "\n"
    
    # Clean up: strip trailing newlines and remove empty sections
    for key in structured:
        structured[key] = structured[key].strip()
    
    structured = {k: v for k, v in structured.items() if v}
    
    return structured

def format_structured_resume(structured: Dict[str, str]) -> str:
    """
    Format structured resume dict back into readable text for NER input.
    """
    formatted = []
    for section, content in structured.items():
        if content:
            formatted.append(f"\n## {section.upper()}\n{content}")
    return "\n".join(formatted)

def _ner_subprocess_worker(input_file: str, output_file: str, model_id: str):
    """
    Child process: load transformers pipeline, run NER on text, write result + child peak RSS.
    """
    try:
        import importlib, json, psutil
        from transformers import pipeline

        # Read input
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()

        mem_before = psutil.Process().memory_info().rss / 1024 / 1024
        p = pipeline(task="token-classification", model=model_id, aggregation_strategy="simple", device=-1)
        res = cast(list[dict[str, Any]], p(text))
        mem_after = psutil.Process().memory_info().rss / 1024 / 1024
        mem_peak = max(mem_before, mem_after)

        # Ensure result is JSON-serializable (convert numpy/torch types)
        try:
            serializable = convert_numpy_types(res)
        except Exception:
            # Best-effort: convert 'score' fields and fallback to stringifying unknowns
            try:
                for item in res:
                    if isinstance(item, dict) and "score" in item:
                        item["score"] = float(item["score"])
                serializable = res
            except Exception:
                serializable = json.loads(json.dumps(res, default=str))

        with open(output_file, "w", encoding="utf-8") as out:
            json.dump({"result": serializable, "mem_peak_mb": round(mem_peak, 1)}, out, ensure_ascii=False)
        return 0
    except Exception as e:
        with open(output_file, "w", encoding="utf-8") as out:
            json.dump({"error": str(e)}, out)
        return 1

async def run_ner_in_subprocess(text: str, model_id: str = "DeezNutz1337/Resume-Parser-BERT_Based") -> dict:
    """
    Run NER in isolated subprocess to ensure memory is reclaimed when the child exits.
    Returns dict: {"result": [...], "mem_peak_mb": X} or raises RuntimeError.
    """
    loop = asyncio.get_event_loop()
    # prepare temp files
    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8", suffix=".txt") as inp:
        inp.write(text)
        input_file = inp.name
    output_file = tempfile.mktemp(suffix=".json")
    try:
        cwd = os.getcwd()
        code = f"""
import sys
sys.path.insert(0, {repr(cwd)})
from services.parsing_service import _ner_subprocess_worker
exit(_ner_subprocess_worker({repr(input_file)}, {repr(output_file)}, {repr(model_id)}))
"""
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        # If child failed, prefer to read the structured JSON it may have written
        if proc.returncode != 0:
            # try to read JSON first
            if os.path.exists(output_file):
                try:
                    with open(output_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        if "error" in data:
                            raise RuntimeError(f"NER subprocess failed: {data.get('error')}")
                        if "result" in data:
                            # child produced a result despite non-zero exit; return it
                            return data
                except json.JSONDecodeError:
                    # fall back to stderr below
                    pass
                except Exception as e:
                    # if reading JSON raised, include stderr for context
                    err_txt = stderr.decode("utf-8", errors="ignore").strip()
                    raise RuntimeError(f"NER subprocess failed (reading child JSON): {e} | stderr: {err_txt or 'none'}")
            # fallback to stderr/stdout content
            err = stderr.decode("utf-8", errors="ignore").strip() or stdout.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"NER subprocess failed: {err or 'unknown error'}")

        # Child succeeded — read its output JSON
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise RuntimeError("NER subprocess succeeded but output file is missing")

        if "error" in data:
            raise RuntimeError(data["error"])
        return data
    finally:
        try:
            os.unlink(input_file)
        except Exception:
            pass
        try:
            if os.path.exists(output_file):
                os.unlink(output_file)
        except Exception:
            pass