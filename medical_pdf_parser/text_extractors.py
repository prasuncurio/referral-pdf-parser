from typing import Optional
import fitz  # PyMuPDF
import pdfplumber
import PyPDF2
from .logger import setup_logger
from .ocr import OCRManager
from .config import ParserConfig


class TextExtractor:
    def __init__(self, config: ParserConfig):
        self.config = config
        self.logger = setup_logger(config)
        self.ocr = OCRManager(config)

    def _compute_page_range(self, pdf_path: str) -> Optional[tuple]:
        """
        Compute a safe (start, end) 1-based inclusive page range from config and PDF length.
        Returns (start, end) where either can be None meaning "use whole document".
        """
        start = self.config.page_start
        end = self.config.page_end
        limit = self.config.page_limit

        if start is None and end is None and limit is None:
            return None

        total_pages = None
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()
        except Exception:
            total_pages = None

        if start is None:
            start = 1

        if limit is not None and (end is None):
            end = start + max(0, limit - 1)

        if end is None and total_pages is not None:
            end = total_pages

        if start is not None and start < 1:
            start = 1
        if total_pages is not None and end is not None and end > total_pages:
            end = total_pages

        return start, end

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF (tries native, then OCR if necessary). Respects page range in config.
        """
        text = ""
        page_range = self._compute_page_range(pdf_path)
        page_start = page_end = None
        if page_range:
            page_start, page_end = page_range

        # Try PyMuPDF
        try:
            doc = fitz.open(pdf_path)
            if page_start is None and page_end is None:
                pages_iter = range(len(doc))
            else:
                start_idx = (page_start - 1) if page_start is not None else 0
                end_idx = (page_end - 1) if page_end is not None else len(doc) - 1
                start_idx = max(0, start_idx)
                end_idx = min(len(doc) - 1, end_idx)
                pages_iter = range(start_idx, end_idx + 1)
            for i in pages_iter:
                page = doc[i]
                try:
                    txt = page.get_text()
                    if txt:
                        text += txt + "\n"
                except Exception:
                    self.logger.debug(f"PyMuPDF get_text failed for page {i}")
            doc.close()
            if text.strip():
                self.logger.info("Extracted text using PyMuPDF.")
                return text
        except Exception as e:
            self.logger.warning(f"PyMuPDF failed: {e}")

        # Try pdfplumber
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_start is None and page_end is None:
                    pages_to_use = pdf.pages
                else:
                    s = (page_start - 1) if page_start is not None else 0
                    e = (page_end) if page_end is not None else len(pdf.pages)
                    pages_to_use = pdf.pages[s:e]
                for page in pages_to_use:
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception:
                        self.logger.debug("pdfplumber failed extracting a page")
            if text.strip():
                self.logger.info("Extracted text using pdfplumber.")
                return text
        except Exception as e:
            self.logger.warning(f"pdfplumber failed: {e}")

        # Try PyPDF2
        try:
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                total = len(pdf_reader.pages)
                if page_start is None and page_end is None:
                    start_idx = 0
                    end_idx = total - 1
                else:
                    start_idx = (page_start - 1) if page_start is not None else 0
                    end_idx = (page_end - 1) if page_end is not None else total - 1
                    start_idx = max(0, start_idx)
                    end_idx = min(total - 1, end_idx)
                for i in range(start_idx, end_idx + 1):
                    page = pdf_reader.pages[i]
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception:
                        self.logger.debug(f"PyPDF2 failed extracting page {i}")
            if text.strip():
                self.logger.info("Extracted text using PyPDF2.")
                return text
        except Exception as e:
            self.logger.warning(f"PyPDF2 failed: {e}")

        # Fallback to OCR
        self.logger.info("All direct extraction failed; trying OCR...")
        pages = self.ocr.pdf_to_images(pdf_path, dpi=self.config.dpi, page_start=page_start, page_end=page_end)
        if not pages:
            self.logger.error("No images generated from PDF; cannot perform OCR.")
            return ""

        for method in self.config.ocr_order:
            if method == "easyocr":
                text = self.ocr.extract_with_easyocr(pages)
                if text.strip():
                    self.logger.info("Extracted text using EasyOCR.")
                    return text
            elif method == "tesseract":
                text = self.ocr.extract_with_tesseract(pages, tesseract_config=self.config.tesseract_config)
                if text.strip():
                    self.logger.info("Extracted text using pytesseract.")
                    return text
            elif method == "trocr":
                text = self.ocr.extract_with_trocr(pages)
                if text.strip():
                    self.logger.info("Extracted text using TrOCR.")
                    return text
            else:
                self.logger.debug(f"Unknown OCR method in config: {method}")

        self.logger.error("All extraction methods failed.")
        return ""