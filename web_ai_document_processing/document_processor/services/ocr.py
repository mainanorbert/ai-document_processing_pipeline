from typing import Any, Dict, List
from pathlib import Path
from threading import Lock

from langchain.tools import tool
from paddlex import create_pipeline

_ocr_pipeline = None
_ocr_lock = Lock()


def _get_ocr_pipeline():
    """
    Lazily initialize PaddleX OCR pipeline once per process.
    """
    global _ocr_pipeline

    if _ocr_pipeline is not None:
        return _ocr_pipeline

    with _ocr_lock:
        if _ocr_pipeline is None:
            print("Initializing PaddleX OCR pipeline...")
            _ocr_pipeline = create_pipeline("OCR")
            print("PaddleX OCR ready.")

    return _ocr_pipeline


@tool
def paddle_ocr_read_document(image_path: str) -> List[Dict[str, Any]]:
    """
    Extract text with bounding boxes and confidence scores using PaddleX OCR.
    """
    try:
        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        pipeline = _get_ocr_pipeline()
        result = pipeline(str(image_file))

        blocks = result.get("result", [])
        extracted: List[Dict[str, Any]] = []

        for block in blocks:
            text = block.get("text", "").strip()
            bbox = block.get("box", [])

            if not text or not bbox:
                continue

            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]

            extracted.append(
                {
                    "text": text,
                    "bbox": [min(xs), min(ys), max(xs), max(ys)],
                }
            )

        if not extracted:
            return [{"warning": "No text detected"}]

        return extracted

    except Exception as exc:
        return [{"error": f"OCR failed: {exc}"}]