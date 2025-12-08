from pdf2image import convert_from_bytes
from PIL import Image, ImageEnhance
import cv2
import os
from datetime import datetime
from collections import OrderedDict
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pathlib import Path
from io import BytesIO
import io
import json
import requests
import time
import base64
from docx import Document as DocxDocument
from utils.json_encoder import convert_numpy_types
from typing import Dict, Any, Union, List, Optional
import asyncio
import numpy as np
import importlib
import gc
import psutil
import sys

load_dotenv()


GEMINI_MODEL = "gemini-2.5-pro"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Keep track of processing status
processing_status: OrderedDict = OrderedDict()

# Type aliases for better readability
OCRWord = Dict[str, Union[str, float, List[List[float]]]]
OCRLine = Dict[str, List[OCRWord]]
OCRBlock = Dict[str, List[OCRLine]]
OCRPage = Dict[str, List[OCRBlock]]
OCRResult = Dict[str, List[OCRPage]]

# Add memory tracking constants
MAX_MEMORY_MB = 1024  # Hard limit per request
WARN_MEMORY_MB = 900  # Warning threshold
DOCTR_DPI = 200  # Reduced from 300 for memory savings

# Adjust thresholds for Doctr loading
DOCTR_REQUIRED_MB = 150  # Memory needed to safely load DocTR (adjust based on your model)
DOCTR_BUFFER_MB = 50    # Safety buffer after loading

def update_processing_status(filename: Optional[str], status: str, error: Optional[str] = None):
    """Update processing status for a file"""
    processing_status[filename] = {
        "filename": filename,
        "status": status,  # "processing", "completed", "error"
        "error": error,
        "timestamp": datetime.now().isoformat(),
    }
    # Keep only last 100 files
    while len(processing_status) > 100:
        processing_status.popitem(last=False)

def get_processing_status() -> Dict[str, Any]:
    """Get current processing status"""
    # Get list of files currently being processed
    active_files = [
        item["filename"] 
        for item in processing_status.values() 
        if item["status"] == "processing"
    ]
    
    return {
        "processing": len(active_files) > 0,
        "activeFiles": active_files,
        # Keep the detailed status for other uses
        "files": list(processing_status.values())
    }

def get_memory_usage() -> float:
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def check_memory_available(required_mb: int = 100) -> bool:
    """Check if enough memory is available"""
    current = get_memory_usage()
    available = MAX_MEMORY_MB - current
    return available >= required_mb

def _preprocess_image_for_trocr(img_array: np.ndarray) -> Image.Image:
    """Preprocess image for TrOCR: denoise, contrast enhancement, convert to PIL"""
    # 1. Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # 2. Denoise
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # 3. Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    
    # 4. Convert back to PIL Image (RGB for model compatibility)
    pil_img = Image.fromarray(contrast).convert('RGB')
    return pil_img

# Optimize image preprocessing for lower DPI
def _preprocess_image_for_doctr(img_array: np.ndarray, target_dpi: int = 200) -> Image.Image:
    """
    Minimal preprocessing for DocTR with lower DPI for memory efficiency.
    """
    # Resize if image is too large
    h, w = img_array.shape[:2]
    max_dim = 1500  # Reduce from default
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_array = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    pil_img = Image.fromarray(img_array).convert('RGB')
    return pil_img

async def run_ocr(trocr_printed, file) -> Dict[str, Any]:
    """Run OCR using TrOCR model"""
    filename = getattr(file, 'filename', 'unknown')
    update_processing_status(filename, "processing")
    
    try:
        content = await file.read()
        
        # Load and preprocess image
        image = Image.open(io.BytesIO(content)).convert('RGB')
        img_array = np.array(image)
        
        preprocessed = _preprocess_image_for_doctr(img_array)
        
        # Run TrOCR inference
        result = await asyncio.to_thread(trocr_printed, preprocessed)
        
        # Extract text from result (TrOCR returns list of dicts with 'generated_text')
        extracted_text = result[0]['generated_text'] if result else ""
        
        # Convert to compatible format with "pages" structure
        exported = {
            "pages": [{
                "text": extracted_text,
                "blocks": [{
                    "lines": [{
                        "words": [{"value": w, "confidence": None, "geometry": []} for w in extracted_text.split()]
                    }]
                }]
            }]
        }
        
        update_processing_status(filename, "completed")
        return exported
        
    except Exception as e:
        update_processing_status(filename, "error", str(e))
        raise

# Optimize PDF conversion
def pdf_to_images_bytes(pdf_bytes: bytes, dpi: int = 200) -> List[bytes]:  # Reduced from 300
    """Convert PDF to images with lower DPI for memory efficiency"""
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, fmt='ppm')  # Use PPM instead of PNG
    out = []
    for page in pages:
        # Reduce quality
        buf = BytesIO()
        page.save(buf, format="JPEG", quality=75, optimize=True)
        out.append(buf.getvalue())
        del page  # Explicit cleanup
    return out

def save_pdf_images(
    pdf_bytes: bytes,
    output_root: Path | None = None,
    base_name: str = "document",
    dpi: int = 300
) -> Dict[str, Any]:
    """
    Save PDF pages as PNG files into a folder named <base_name>_images under output_root.
    Returns metadata: { original_name, dir, pages: [{page, filename, path}] }
    """
    output_root = output_root or Path(os.getenv("DOCUMENTS_FOLDER", "./documents"))
    output_root.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(time.time())
    base_name = base_name or f"document_{timestamp}"
    folder_name = f"{Path(base_name).stem}_images_{timestamp}"
    out_dir = output_root / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pages_bytes = pdf_to_images_bytes(pdf_bytes, dpi=dpi)
    pages_meta = []
    for i, img_bytes in enumerate(pages_bytes, start=1):
        img_name = f"{Path(base_name).stem}_page_{i}.png"
        img_path = out_dir / img_name
        with open(img_path, "wb") as f:
            f.write(img_bytes)
        pages_meta.append({
            "page": i,
            "filename": img_name,
            "path": str(img_path)
        })

    return {
        "original_name": base_name,
        "dir": str(out_dir),
        "pages": pages_meta
    }

def pdf_to_images_b64(pdf_bytes: bytes, dpi: int = 150) -> List[Dict[str, Any]]:
    """Return base64-encoded images for quick preview (does not save)"""
    pages_bytes = pdf_to_images_bytes(pdf_bytes, dpi=dpi)
    out = []
    for i, b in enumerate(pages_bytes, start=1):
        out.append({"page": i, "b64": base64.b64encode(b).decode("utf-8")})
    return out

def ocr_space_ocr(image_bytes, api_key=None, language="eng"):
    """Run OCR using ocr.space API and return structured results."""
    if api_key is None:
        api_key = os.getenv("OCR_SPACE_API_KEY") or os.getenv("FREEOCR")
    url = "https://api.ocr.space/parse/image"
    files = {'file': ('image.jpg', image_bytes)}
    data = {
        'apikey': api_key,
        'language': language,
        'isTable': True,
        'OCREngine': 2
    }
    response = requests.post(url, files=files, data=data)
    response.raise_for_status()
    result = response.json()
    parsed = result.get("ParsedResults", [{}])[0]
    page_text = parsed.get("ParsedText", "")
    lines = []
    for line in page_text.split("\n"):
        words = [{"value": w, "confidence": None, "geometry": []} for w in line.split() if w]
        if words:
            lines.append({"words": words})
    # Group all lines into a single block
    blocks = [{"lines": lines}] if lines else []
    return {"pages": [{"blocks": blocks}]}

def extract_on_document(file, model):
    """
    Accepts:
      - bytes (uploaded content)
      - file path string (pdf, docx, image)
    Returns:
      - (doc, exported) where doc is PIL Image for DocTR, or None for text
      - exported is a dict with "pages": [...] structure
    """
    # If file is bytes (uploaded file content)
    if isinstance(file, bytes):
        # Try docx first
        try:
            docx_doc = DocxDocument(io.BytesIO(file))
            paragraphs = [p.text for p in docx_doc.paragraphs if p.text.strip()]
            table_texts = []
            for table in docx_doc.tables:
                rows = []
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    rows.append("\t".join(cells))
                if rows:
                    table_texts.append("\n".join(rows))
            full_text = "\n\n".join(filter(None, paragraphs + table_texts))
            return None, {"pages": [{"text": full_text}]}
        except Exception:
            pass

        # Try to read as PDF — convert to images for OCR
        try:
            # Use pdf2image to convert PDF pages to images for Doctr OCR
            pages_images = convert_from_bytes(file, dpi=300)
            if pages_images and len(pages_images) > 0:
                # Return first page as PIL Image for OCR
                first_page = pages_images[0]
                if isinstance(first_page, Image.Image):
                    return first_page, {"pages": []}
        except Exception as e:
            print(f"PDF conversion failed: {e}")
            pass

        # Not a PDF or docx, treat as image
        try:
            image = Image.open(io.BytesIO(file)).convert('RGB')
            img_array = np.array(image)
            preprocessed = _preprocess_image_for_doctr(img_array)
            return preprocessed, {"pages": []}
        except Exception as e:
            raise ValueError(f"Unsupported file type or failed to process document: {e}")

    # If file is a path string
    elif isinstance(file, str):
        lower = file.lower()
        if lower.endswith(".docx"):
            docx_doc = DocxDocument(file)
            paragraphs = [p.text for p in docx_doc.paragraphs if p.text.strip()]
            table_texts = []
            for table in docx_doc.tables:
                rows = []
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    rows.append("\t".join(cells))
                if rows:
                    table_texts.append("\n".join(rows))
            full_text = "\n\n".join(filter(None, paragraphs + table_texts))
            return None, {"pages": [{"text": full_text}]}
        elif lower.endswith(".pdf"):
            try:
                # Try PDF image conversion first
                pages_images = convert_from_bytes(open(file, 'rb').read(), dpi=300)
                if pages_images and len(pages_images) > 0 and isinstance(pages_images[0], Image.Image):
                    return pages_images[0], {"pages": []}
            except Exception:
                pass
            
            # Fallback to text extraction
            reader = PdfReader(file)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            return None, {"pages": [{"text": text}]}
        else:
            # File path to image
            image = Image.open(file).convert('RGB')
            img_array = np.array(image)
            preprocessed = _preprocess_image_for_doctr(img_array)
            return preprocessed, {"pages": []}

    # Fallback
    try:
        image = Image.open(file).convert('RGB')
        img_array = np.array(image)
        preprocessed = _preprocess_image_for_doctr(img_array)
        return preprocessed, {"pages": []}
    except Exception as e:
        raise ValueError(f"Unsupported file type or failed to process document: {e}")

def search_word(exported_doc, query: str):
    """Search for words in OCR output and return matches"""
    matches = []
    for page_idx, page in enumerate(exported_doc.get("pages", [])):
        page_text = page.get("text", "")
        if query.lower() in page_text.lower():
            matches.append({
                "page": page_idx,
                "query": query,
                "found": True
            })
    return matches

def extract_context(text: str, query: str, context_chars: int = 100) -> str:
    """Helper function to extract text context around the match."""
    query_pos = text.lower().find(query.lower())
    if query_pos == -1:
        return ""
        
    start = max(0, query_pos - context_chars)
    end = min(len(text), query_pos + len(query) + context_chars)
    
    context = text[start:end]
    if start > 0:
        context = f"...{context}"
    if end < len(text):
        context = f"{context}..."
        
    return context

def collect_all_pages(file, model):
    """Collect all pages from document"""
    doc, exported = extract_on_document(file, model)
    pages = []
    
    for page_idx, page_dict in enumerate(exported.get("pages", [])):
        page_text = page_dict.get("text", "")
        pages.append({
            "page_number": page_idx + 1,
            "text": page_text,
            "image": None,
            "ocr_data": page_dict
        })
    return pages

def save_ocr_layer(exported, output_path):
    """Save the structured OCR output as a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(exported, f, ensure_ascii=False, indent=2)

def load_ocr_layer(json_path):
    """Load the OCR layer from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Modify create_doctr_ocr to be memory-aware
def create_doctr_ocr(device: str = "cpu"):
    """
    Create Doctr OCR with memory optimization.
    Use CPU by default (GPU not available on free tier anyway).
    """
    
    current_mem = get_memory_usage()
    available_mem = MAX_MEMORY_MB - current_mem
    
    # Check if we have enough memory to load DocTR
    if available_mem < DOCTR_REQUIRED_MB:
        raise MemoryError(
            f"Insufficient memory to load DocTR. "
            f"Current: {current_mem:.1f}MB, Required: {DOCTR_REQUIRED_MB}MB, "
            f"Available: {available_mem:.1f}MB / {MAX_MEMORY_MB}MB"
        )
    
    if not check_memory_available(150):
        raise MemoryError(f"Insufficient memory: {get_memory_usage():.1f}MB / {MAX_MEMORY_MB}MB")
    
    print(f"Loading Doctr OCR (device={device}, current memory={get_memory_usage():.1f}MB)...")
    
    try:
        doctr_models = importlib.import_module("doctr.models")
        
        # Use smaller, faster models
        detector = doctr_models.fast_small(pretrained=True)  # Smaller than large
        recognizer = doctr_models.crnn_mobilenet_v3_small(pretrained=True)  # Smaller model
        
        predictor = doctr_models.ocr_predictor(
            det_arch=detector,
            reco_arch=recognizer,
            pretrained=True,
            assume_straight_pages=True,
        )
        
        if device == "cpu":
            predictor = predictor.to("cpu")
            
        mem_after = get_memory_usage()
        print(f"Doctr loaded. Memory: {mem_after:.1f}MB (delta: +{mem_after - current_mem:.1f}MB)")
        return predictor
    except Exception as e:
        print(f"Failed to load Doctr: {e}")
        raise

def dispose_doctr_ocr(predictor):
    """Aggressively dispose of predictor and free memory"""
    try:
        if hasattr(predictor, "model"):
            del predictor.model
        if hasattr(predictor, "det_predictor"):
            del predictor.det_predictor
        if hasattr(predictor, "reco_predictor"):
            del predictor.reco_predictor
        del predictor
    except Exception:
        pass
    
    gc.collect()
    import ctypes
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)  # Linux-specific memory trim
    except (OSError, AttributeError):
        pass
    
    print(f"Memory after cleanup: {get_memory_usage():.1f}MB")