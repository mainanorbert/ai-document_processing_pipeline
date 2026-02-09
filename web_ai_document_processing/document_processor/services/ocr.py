# from typing import Any, Dict, List

# from langchain.tools import tool

# # We do NOT import PaddleOCR at the top anymore
# # This prevents model loading + PDX initialization at import time

# _ocr_instance: 'PaddleOCR | None' = None   # ← use string literal for forward reference


# def _get_ocr():
#     """
#     Lazily initialize and return the global PaddleOCR instance.
#     Model loading + PaddleX initialization only happens on first real OCR call.
#     """
#     global _ocr_instance
#     if _ocr_instance is None:
#         # Import happens here — only once, on first use
#         from paddleocr import PaddleOCR
#         print("Initializing PaddleOCR (may take 5–30 seconds the first time)...")
#         _ocr_instance = PaddleOCR(
#             use_angle_cls=True,    # recommended for better rotation handling
#             lang="en",
#             # Optional: tune these if needed
#             # use_gpu=False,
#             # show_log=False,
#             # det_limit_side_len=960,
#         )
#         print("PaddleOCR ready.")
#     return _ocr_instance


# @tool
# def paddle_ocr_read_document(image_path: str) -> List[Dict[str, Any]]:
#     """
#     Extract text lines with bounding boxes and confidence from an image using PaddleOCR.

#     Returns:
#         List of dicts with keys: "text" (str), "bbox" (list[int]), optional "confidence" (float)
#         On error/failure: list with a single {"error": "..."} dict
#     """
#     try:
#         ocr = _get_ocr()
#         # Modern PaddleOCR uses .ocr() — not .predict()
#         result = ocr.ocr(image_path)

#         if not result or len(result) == 0 or len(result[0]) == 0:
#             return [{"warning": "No text detected in the image"}]

#         page = result[0]  # usually result[0] for single image

#         extracted_items: List[Dict[str, Any]] = []

#         for line in page:
#             if len(line) != 2:
#                 continue
#             box_points, (text, confidence) = line

#             x_coords = [int(p[0]) for p in box_points]
#             y_coords = [int(p[1]) for p in box_points]

#             item: Dict[str, Any] = {
#                 "text": text,
#                 "bbox": [
#                     min(x_coords),
#                     min(y_coords),
#                     max(x_coords),
#                     max(y_coords),
#                 ],
#             }
#             if confidence is not None:
#                 item["confidence"] = round(float(confidence), 4)

#             extracted_items.append(item)

#         return extracted_items or [{"warning": "No readable text found"}]

#     except Exception as exc:
#         return [{"error": f"OCR failed: {str(exc)}"}]


# def paddle_ocr_text(image_path: str) -> str:
#     """
#     Extract plain text (lines joined by \n) from an image.
#     Returns error message string on failure.
#     """
#     try:
#         ocr = _get_ocr()
#         result = ocr.ocr(image_path)

#         if not result or len(result) == 0 or len(result[0]) == 0:
#             return ""

#         page = result[0]
#         lines = [line[1][0].strip() for line in page if len(line) == 2 and line[1][0].strip()]

#         return "\n".join(lines)

#     except Exception as exc:
#         return f"OCR failed: {str(exc)}"


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
