from typing import Any, Dict, List
from pathlib import Path
from threading import Lock

from langchain.tools import tool

_ocr_instance = None
_ocr_lock = Lock()


def _get_ocr():
    """
    Lazily initialize PaddleOCR once per process in a thread-safe way.
    """
    global _ocr_instance

    if _ocr_instance is not None:
        return _ocr_instance

    with _ocr_lock:
        if _ocr_instance is None:
            from paddleocr import PaddleOCR

            print("Initializing PaddleOCR (predict-based, stable mode)...")
            _ocr_instance = PaddleOCR(lang="en")
            print("PaddleOCR ready.")

    return _ocr_instance


@tool
def paddle_ocr_read_document(image_path: str) -> List[Dict[str, Any]]:
    """
    Extract text with bounding boxes and confidence scores from an image.
    """
    try:
        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        ocr = _get_ocr()
        result = ocr.predict(str(image_file))

        if not result or not result[0]:
            return [{"warning": "No text detected"}]

        page = result[0]
        texts = page.get("rec_texts", [])
        boxes = page.get("dt_polys", [])
        scores = page.get("rec_scores", [None] * len(texts))

        extracted: List[Dict[str, Any]] = []

        for text, box, score in zip(texts, boxes, scores):
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]

            item: Dict[str, Any] = {
                "text": text,
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
            }

            if score is not None:
                item["confidence"] = round(float(score), 4)

            extracted.append(item)

        return extracted

    except Exception as exc:
        return [{"error": f"OCR failed: {exc}"}]