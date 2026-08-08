import fitz  # PyMuPDF
from paddleocr import PaddleOCR
import numpy as np
from PIL import Image
import io
from typing import Tuple

class MultiModalExtractor:
    def __init__(self, use_gpu: bool = True):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu, show_log=False)

    def process_pdf_page(self, page: fitz.Page, page_num: int, min_text_chars: int = 50) -> Tuple[str, str, Image.Image]:
        native_text = page.get_text("text").strip()
        pix = page.get_pixmap(dpi=150)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

        if len(native_text) >= min_text_chars:
            return "digital", native_text, img
        
        img_np = np.array(img)
        ocr_result = self.ocr.ocr(img_np, cls=True)
        
        extracted_lines = []
        if ocr_result and ocr_result[0]:
            for line in ocr_result[0]:
                text_content = line[1][0]
                confidence = line[1][1]
                if confidence > 0.4:
                    extracted_lines.append(text_content)
                    
        ocr_text = "\n".join(extracted_lines)
        return "scanned", ocr_text, img