from pdf2image import convert_from_bytes
from doctr.io import DocumentFile
from doctr.utils.visualization import visualize_page
import matplotlib.pyplot as plt
import layoutparser as lp
import numpy as np
from doctr.io import DocumentFile
from doctr.utils import geometry
from PIL import Image, ImageEnhance
import cv2
import os
from datetime import datetime
from collections import OrderedDict
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from google.cloud import vision
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

# Update run_ocr to use status tracking
async def run_ocr(model, file) -> OCRResult:
    """Run OCR with status tracking"""
    filename = getattr(file, 'filename', 'unknown')
    update_processing_status(filename, "processing")
    
    try:
        content = await file.read()
        
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(content))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Preprocessing pipeline
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # 1. Denoise
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # 2. Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(denoised)
        
        # 3. Thresholding
        thresh = cv2.adaptiveThreshold(
            contrast, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # 4. Dilation to enhance text
        kernel = np.ones((1,1), np.uint8)
        processed = cv2.dilate(thresh, kernel, iterations=1)
        
        # Convert back to bytes
        is_success, buffer = cv2.imencode(".png", processed)
        processed_bytes = io.BytesIO(buffer).getvalue()
        
        # Run OCR on processed image
        doc = DocumentFile.from_images([processed_bytes])
        result = model(doc)
        exported = result.export()
        # Convert numpy types and ensure dict structure
        converted: OCRResult = convert_numpy_types(exported)
        if not isinstance(converted, dict) or "pages" not in converted:
            converted = {"pages": []}
        
        update_processing_status(filename, "completed")
        return converted
    except Exception as e:
        update_processing_status(filename, "error", str(e))
        raise

# New: convert PDF bytes to list of PNG bytes (in-memory)
def pdf_to_images_bytes(pdf_bytes: bytes, dpi: int = 300) -> List[bytes]:
    """
    Convert PDF bytes to a list of PNG image bytes (one per page).
    Returns list[bytes].
    """
    pages = convert_from_bytes(pdf_bytes, dpi=dpi)
    out = []
    for page in pages:
        buf = BytesIO()
        page.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out

# New: save PDF pages as images and return mapping associated with the original PDF
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

# Optional helper: return base64-encoded images for quick preview (does not save)
def pdf_to_images_b64(pdf_bytes: bytes, dpi: int = 150) -> List[Dict[str, Any]]:
    pages_bytes = pdf_to_images_bytes(pdf_bytes, dpi=dpi)
    out = []
    for i, b in enumerate(pages_bytes, start=1):
        out.append({"page": i, "b64": base64.b64encode(b).decode("utf-8")})
    return out


async def process_multipage_document(content: bytes, ocr_function):
    """
    Process a multipage document (PDF/TIFF) using the specified OCR function.
    Returns combined results from all pages.
    """
    try:
        # Convert PDF pages to images
        pages = convert_from_bytes(content)
        results = []
        
        for i, page in enumerate(pages):
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            page.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Process each page with OCR
            page_result = ocr_function(img_byte_arr)
            if isinstance(page_result, dict) and "pages" in page_result:
                # Add page number to result
                for page_data in page_result["pages"]:
                    page_data["page_number"] = i + 1
                results.extend(page_result["pages"])
        
        return {"pages": results}
    except Exception as e:
        return {"error": f"Failed to process document: {str(e)}"}

def google_vision_ocr(image_bytes):
    """
    Run OCR using Google Vision API and return structured results.
    """
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.document_text_detection(image=image) # type: ignore[attr-defined]
    annotations = response.full_text_annotation

    # Structure the result similar to doctr's export()
    result = {
        "pages": []
    }
    for page in annotations.pages:
        page_dict = {"blocks": []}
        for block in page.blocks:
            block_dict = {"lines": []}
            for paragraph in block.paragraphs:
                line_dict = {"words": []}
                for word in paragraph.words:
                    word_text = "".join([symbol.text for symbol in word.symbols])
                    line_dict["words"].append({
                        "value": word_text,
                        "confidence": word.confidence,
                        "geometry": [
                            [v.x / page.width, v.y / page.height] for v in word.bounding_box.vertices
                        ] if word.bounding_box.vertices else []
                    })
                block_dict["lines"].append(line_dict)
            page_dict["blocks"].append(block_dict)
        result["pages"].append(page_dict)
    return result

def ocr_space_ocr(image_bytes, api_key=None, language="eng"):
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
    print("Text:", blocks)
    return {"pages": [{"blocks": blocks}]}

# def ocr_space_ocr(image_bytes, api_key=None, language="eng"):
#     """
#     Run OCR using ocr.space API and return structured results.
#     """
#     if api_key is None:
#         api_key = os.getenv("OCR_SPACE_API_KEY") or os.getenv("FREEOCR")
#     url = "https://api.ocr.space/parse/image"
#     files = {'file': ('image.jpg', image_bytes)}
#     data = {
#         'apikey': api_key,
#         'language': language,
#         'isTable': True,
#         'OCREngine': 2
#     }
#     response = requests.post(url, files=files, data=data)
#     response.raise_for_status()
#     result = response.json()
#     # Structure the result similar to previous format
#     pages = []
#     parsed_results = result.get("ParsedResults", [])
#     for parsed in parsed_results:
#         page_text = parsed.get("ParsedText", "")
#         # Split into lines and words for compatibility
#         blocks = []
#         for line in page_text.split("\n"):
#             words = [{"value": w, "confidence": None, "geometry": []} for w in line.split() if w]
#             if words:
#                 blocks.append({"lines": [{"words": words}]})
#         pages.append({"blocks": blocks})
#     return {"pages": pages}

# def extract_tables_from_image(image_path):
#     """
#     Detect tables in an image using LayoutParser's PubLayNet model.
#     Returns a list of table bounding boxes and cropped table images.
#     """
#     # Load image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise FileNotFoundError(f"Image at path '{image_path}' could not be loaded.")
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     model = Detectron2LayoutModel(
#         config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
#         label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
#         extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8]
#     )

#     # Detect layout
#     layout = model.detect(image_rgb)

#     # Filter for tables
#     tables = [b for b in layout if b.type == "Table"]

#     table_results = []
#     for idx, table in enumerate(tables):
#         x1, y1, x2, y2 = map(int, table.coordinates)
#         table_img = image_rgb[y1:y2, x1:x2]
#         table_results.append({
#             "bbox": [x1, y1, x2, y2],
#             "image": table_img
#         })
#         # Optionally, save or display the cropped table image
#         # Image.fromarray(table_img).save(f"table_{idx+1}.png")

#     return table_results

def extract_on_document(file, model):
    """
    Accepts:
      - bytes (uploaded content)
      - file path string (pdf, docx, image)
    Returns:
      - (doc, exported) where doc is a DocumentFile (for images) or None (for text/docx/pdf)
      - exported is a dict with "pages": [{"text": ...}] (compatible with other functions)
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

        # Try to read as PDF
        try:
            reader = PdfReader(io.BytesIO(file))
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            return None, {"pages": [{"text": text}]}
        except Exception:
            # Not a PDF or docx, treat as image
            doc = DocumentFile.from_images([file])
            result = model(doc)
            exported = result.export()
            return doc, exported

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
            reader = PdfReader(file)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            return None, {"pages": [{"text": text}]}
        else:
            # File path to image
            doc = DocumentFile.from_images(file)
            result = model(doc)
            exported = result.export()
            return doc, exported

    # Fallback: if unknown type, try to treat as image bytes/path
    try:
        doc = DocumentFile.from_images(file)
        result = model(doc)
        exported = result.export()
        return doc, exported
    except Exception as e:
        raise ValueError(f"Unsupported file type or failed to process document: {e}")

async def search_word(exported_doc, query: str):
    """Search for words in OCR output and return matches with their boxes"""
    matches = []
    for page_idx, page in enumerate(exported_doc["pages"]):
        for block in page["blocks"]:
            for line in block["lines"]:
                for word in line["words"]:
                    if query.lower() in word["value"].lower():
                        matches.append({
                            "page": page_idx,
                            "word": word["value"],
                            "box": word["geometry"]  # normalized (x_min, y_min, x_max, y_max)
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
    doc, exported = extract_on_document(file, model)
    pages = []
    if doc is not None:
        # OCR result (images)
        for page_idx, (page_dict, image) in enumerate(zip(exported["pages"], doc)):
            lines = []
            for block in page_dict.get("blocks", []):
                if isinstance(block, dict):
                    for line in block.get("lines", []):
                        line_text = " ".join([word["value"] for word in line.get("words", [])])
                        lines.append(line_text)
            page_text = "\n".join(lines)
            pages.append({
                "page_number": page_idx + 1,
                "text": page_text,
                "image": image,
                "ocr_data": page_dict
            })
    else:
        # Digital PDF (text extraction)
        for page_idx, page_dict in enumerate(exported.get("pages", [])):
            page_text = page_dict.get("text", "")
            pages.append({
                "page_number": page_idx + 1,
                "text": page_text,
                "image": None,
                "ocr_data": page_dict
            })
    return pages

def ocr_and_visualize(model,file, search_query=None):
    doc, exported = extract_on_document(file, model)
    if doc is not None and hasattr(doc, "__iter__"):
        # OCR result (images)
        for page_idx, (page_dict, image) in enumerate(zip(exported["pages"], doc)):
            print(f"---- PAGE {page_idx+1} ----")
            for block in page_dict.get("blocks", []):
                if isinstance(block, dict):
                    for line in block.get("lines", []):
                        line_text = " ".join([word["value"] for word in line.get("words", [])])
                        print(line_text)
            if search_query:
                search_results = search_word(exported, search_query)
                print("Search Results:", search_results)
            visualize_page(page_dict, image)
            plt.show()
    else:
        # Digital PDF (text extraction)
        for page_idx, page_dict in enumerate(exported.get("pages", [])):
            print(f"---- PAGE {page_idx+1} ----")
            print(page_dict.get("text", ""))


def save_ocr_layer(exported, output_path):
    """
    Save the structured OCR output (exported) as a JSON file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(exported, f, ensure_ascii=False, indent=2)

def load_ocr_layer(json_path):
    """
    Load the OCR layer from a JSON file.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def search_word_in_ocr_layer(exported, query):
    """
    Search for words in the OCR layer and return their bounding boxes.
    """
    matches = []
    for page_idx, page in enumerate(exported["pages"]):
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                for word in line.get("words", []):
                    if query.lower() in word["value"].lower():
                        matches.append({
                            "page": page_idx,
                            "word": word["value"],
                            "box": word["geometry"]  # (x_min, y_min, x_max, y_max)
                        })
    return matches

def visualize_word_boxes(image, matches, color=(255, 0, 0), thickness=2):
    """
    Draw bounding boxes for matched words on the image.
    image: numpy array (RGB)
    matches: list of {"box": [x_min, y_min, x_max, y_max], ...}
    """
    import cv2
    h, w = image.shape[:2]
    img_vis = image.copy()
    for match in matches:
        x_min, y_min, x_max, y_max = match["box"]
        # Geometry is normalized, so scale to image size
        pt1 = (int(x_min * w), int(y_min * h))
        pt2 = (int(x_max * w), int(y_max * h))
        cv2.rectangle(img_vis, pt1, pt2, color, thickness)
        cv2.putText(img_vis, match["word"], (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img_vis


# # Example usage:
# if __name__ == "__main__":
#     # 1. Run OCR and save layer
#     doc = DocumentFile.from_images("samp2.jpg")
#     result = ocr_model(doc)
#     exported = result.export()
#     save_ocr_layer(exported, "samp_ocr_layer.json")

#     # 2. Load OCR layer and search for a word
#     ocr_layer = load_ocr_layer("samp_ocr_layer.json")
#     matches = search_word_in_ocr_layer(ocr_layer, "CAMPUS")
#     print("Matches:", matches)

#     # 3. Visualize
#     image = doc[0]  # PIL Image or numpy array
#     if isinstance(image, Image.Image):
#         image = np.array(image.convert("RGB"))
#     vis_img = visualize_word_boxes(image, matches)
#     plt.imshow(vis_img)
#     plt.axis("off")
#     plt.show()

if __name__ == "__main__":
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    print("Client loaded:", type(client))
    print("Has method:", hasattr(client, "document_text_detection"))