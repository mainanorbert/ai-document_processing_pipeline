from typing import Any, Dict, List

from paddleocr import PaddleOCR
from langchain.tools import tool

ocr = PaddleOCR(lang="en")

@tool
def paddle_ocr_read_document(image_path: str) -> List[Dict[str, Any]]:
    """
    Extract text with bounding boxes from an image using PaddleOCR.
    """
    try:
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
        # return "Nothing was extracted"

    except Exception as exc:
        return [{"error": f"OCR failed: {exc}"}]


def paddle_ocr_text(image_path: str) -> str:
    """
    Return plain OCR text from image.
    """
    try:
        result = ocr.predict(image_path)
        page = result[0]
        return "\n".join(page["rec_texts"])
    except Exception as exc:
        return f"OCR failed: {exc}"
