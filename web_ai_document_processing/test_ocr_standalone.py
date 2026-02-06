from typing import List, Dict, Any, Optional

# Global singleton (lazy initialized)
_ocr_instance = None


def _get_ocr():
    """
    Lazily initialize PaddleOCR only once.
    """
    global _ocr_instance
    if _ocr_instance is None:
        print("Initializing PaddleOCR (predict-based, stable mode)...")
        from paddleocr import PaddleOCR  # ✅ lazy import
        _ocr_instance = PaddleOCR(lang="en")
        print("PaddleOCR ready.")
    return _ocr_instance


def paddle_ocr_text(image_path: str) -> str:
    """
    Extract plain OCR text from image.
    Returns newline-joined text.
    """
    try:
        ocr = _get_ocr()
        result = ocr.predict(image_path)

        if not result or not result[0]:
            return ""

        page = result[0]
        return "\n".join(page.get("rec_texts", []))

    except Exception as exc:
        return f"OCR failed: {exc}"


def paddle_ocr_read_document(image_path: str) -> List[Dict[str, Any]]:
    """
    Extract text with bounding boxes and confidence scores.
    """
    try:
        ocr = _get_ocr()
        result = ocr.predict(image_path)

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


# ---- TEST ----
path = "/home/nober/ai/ai-document_processing_pipeline/web_ai_document_processing/uploads/uploads/invoice.png"

print(paddle_ocr_text(path))
print("llm output:", paddle_ocr_read_document(path))