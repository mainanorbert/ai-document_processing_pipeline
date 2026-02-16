from typing import Any, Dict, List
from django.conf import settings

from paddleocr import PaddleOCR
from langchain.tools import tool
from PIL import Image
import pytesseract
import requests

# Lazy initialization
_paddle_ocr_instance = None


def get_paddle_instance():
    global _paddle_ocr_instance
    if _paddle_ocr_instance is None:
        _paddle_ocr_instance = PaddleOCR(lang="en")
    return _paddle_ocr_instance


def extract_with_paddle(image_path: str) -> List[Dict[str, Any]]:
    ocr = get_paddle_instance()
    result = ocr.predict(image_path)
    page = result[0]

    texts = page["rec_texts"]
    boxes = page["dt_polys"]
    scores = page.get("rec_scores", [None] * len(texts))

    extracted_items: List[Dict[str, Any]] = []

    for text, box, score in zip(texts, boxes, scores):
        x_coords = [point[0] for point in box]
        y_coords = [point[1] for point in box]

        item: Dict[str, Any] = {
            "text": text,
            "bbox": [
                min(x_coords),
                min(y_coords),
                max(x_coords),
                max(y_coords),
            ],
        }

        if score is not None:
            item["confidence"] = score

        extracted_items.append(item)

    return extracted_items


def extract_with_tesseract(image_path: str) -> List[Dict[str, Any]]:
    text = pytesseract.image_to_string(Image.open(image_path))

    # Normalize output to same structure as Paddle
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    return [{"text": line, "bbox": None} for line in lines]




def extract_with_api(image_path: str):
    url = "https://api.ocr.space/parse/image"
    api_key = getattr(settings, "OCR_SPACE_API_KEY", None)

    if not api_key:
        raise ValueError("OCR_SPACE_API_KEY not configured")

    with open(image_path, "rb") as f:
        response = requests.post(
            url,
            data={
                "apikey": api_key,
                "language": "eng",
                "OCREngine": 2,
            },
            files={"file": f},
            timeout=30,
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"OCR API HTTP error {response.status_code}: {response.text}"
        )

    try:
        result = response.json()
    except ValueError:
        raise RuntimeError(f"OCR API returned non-JSON: {response.text}")

    if result.get("OCRExitCode") != 1:
        raise RuntimeError(
            f"OCR API error: {result.get('ErrorMessage')} | Full response: {result}"
        )

    text = result["ParsedResults"][0]["ParsedText"]
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    return [{"text": line, "bbox": None} for line in lines]




# 🔥 Unified OCR tool (THIS is what LLM uses)
@tool
def ocr_read_document(image_path: str) -> List[Dict[str, Any]]:
    """
    Unified OCR tool. Engine selected via Django settings.
    """
    try:
        engine = getattr(settings, "OCR_ENGINE", "paddle")

        if engine == "tesseract":
            print("Using Tesseract OCR backend...")
            return extract_with_tesseract(image_path)
        if engine == "api":
            print("Using OCR API backend...")
            return extract_with_api(image_path)           

        if engine == "paddle":
            print("Using PaddleOCR backend...")
            return extract_with_paddle(image_path)
        else:
            raise ValueError(f"Unsupported OCR_ENGINE: {engine}")

    except Exception as exc:
        return [{"error": f"OCR failed: {exc}"}]


def ocr_plain_text(image_path: str) -> str:
    """
    Returns plain text regardless of backend.
    """
    try:
        results = ocr_read_document.invoke(image_path)
        return "\n".join(
            item["text"] for item in results if "text" in item
        )
    except Exception as exc:
        return f"OCR failed: {exc}"
