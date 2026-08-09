import pymupdf as fitz
import easyocr
import numpy as np
from PIL import Image
import io
from typing import Tuple

class MultiModalExtractor:
    def __init__(self, use_gpu: bool = True):
        # Uses PyTorch (built into Colab) for fast GPU OCR on Python 3.12
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)

    def process_pdf_page(self, page: fitz.Page, page_num: int, min_text_chars: int = 50) -> Tuple[str, str, Image.Image]:
        native_text = page.get_text("text").strip()
        pix = page.get_pixmap(dpi=150)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

        if len(native_text) >= min_text_chars:
            return "digital", native_text, img
        
        img_np = np.array(img)
        ocr_results = self.reader.readtext(img_np)
        
        extracted_lines = []
        for item in ocr_results:
            text_content = item[1]
            confidence = item[2]
            if confidence > 0.4:
                extracted_lines.append(text_content)
                    
        ocr_text = "\n".join(extracted_lines)
        return "scanned", ocr_text, img