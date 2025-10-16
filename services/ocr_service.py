from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from doctr.utils.visualization import visualize_page
import matplotlib.pyplot as plt
import layoutparser as lp
from layoutparser.models import Detectron2LayoutModel
import cv2
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from google.cloud import vision
import io
import json
import requests
load_dotenv()


GEMINI_MODEL = "gemini-2.5-pro"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

async def run_ocr(model, file):
    # This runs inference on each new document
    content = await file.read()
    doc = DocumentFile.from_images([content])
    result = model(doc).export() #export() for structured JSON script, render() for raw text
    
    #print extracted text page by page
    for page in result["pages"]:
        print("---- PAGE ----")
        for block in page["blocks"]:
            for line in block["lines"]:
                line_text = " ".join([word["value"] for word in line["words"]])
                print(line_text)
    
    return result

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

def extract_tables_from_image(image_path):
    """
    Detect tables in an image using LayoutParser's PubLayNet model.
    Returns a list of table bounding boxes and cropped table images.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path '{image_path}' could not be loaded.")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    model = Detectron2LayoutModel(
        config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8]
    )

    # Detect layout
    layout = model.detect(image_rgb)

    # Filter for tables
    tables = [b for b in layout if b.type == "Table"]

    table_results = []
    for idx, table in enumerate(tables):
        x1, y1, x2, y2 = map(int, table.coordinates)
        table_img = image_rgb[y1:y2, x1:x2]
        table_results.append({
            "bbox": [x1, y1, x2, y2],
            "image": table_img
        })
        # Optionally, save or display the cropped table image
        # Image.fromarray(table_img).save(f"table_{idx+1}.png")

    return table_results

def extract_on_document(file, model):
    # If file is bytes (uploaded file content)
    if isinstance(file, bytes):
        # Try to read as PDF first
        try:
            reader = PdfReader(io.BytesIO(file))
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            return None, {"pages": [{"text": text}]}
        except Exception:
            # Not a PDF, treat as image
            doc = DocumentFile.from_images([file])
            result = model(doc)
            exported = result.export()
            return doc, exported
    elif isinstance(file, str) and file.lower().endswith(".pdf"):
        # File path to PDF
        reader = PdfReader(file)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        return None, {"pages": [{"text": text}]}
    else:
        # File path to image or image bytes
        doc = DocumentFile.from_images(file)
        result = model(doc)
        exported = result.export()
        return doc, exported

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

# ...existing code...


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