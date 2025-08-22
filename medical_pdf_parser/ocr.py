from typing import List, Optional, Tuple
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import concurrent.futures
import torch
import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import pytesseract
import pdf2image
from .config import ParserConfig
from .logger import setup_logger

# Lazy import note: heavy objects loaded on demand


class OCRManager:
    def __init__(self, config: ParserConfig):
        self.config = config
        self.logger = setup_logger(config)
        self._trocr_processor = None
        self._trocr_model = None
        self._easyocr_reader = None

    # ---------------------------
    # Helpers
    # ---------------------------
    def pdf_to_images(self, pdf_path: str, dpi: Optional[int] = None,
                      page_start: Optional[int] = None, page_end: Optional[int] = None) -> List[Image.Image]:
        dpi_val = dpi or self.config.dpi
        try:
            kwargs = {"dpi": dpi_val}
            if page_start is not None:
                kwargs["first_page"] = page_start
            if page_end is not None:
                kwargs["last_page"] = page_end
            pages = pdf2image.convert_from_path(pdf_path, **kwargs)
            self.logger.info(f"Converted PDF to {len(pages)} images (dpi={dpi_val}, pages={page_start}-{page_end}).")
            return pages
        except Exception as e:
            self.logger.error(f"PDF to image conversion failed: {e}")
            return []

    def _preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        if not self.config.preprocess_images:
            return image
        try:
            gray = image.convert("L")
            filtered = gray.filter(ImageFilter.MedianFilter(size=3))
            enhanced = ImageOps.autocontrast(filtered)
            return enhanced.convert("RGB")
        except Exception as e:
            self.logger.debug(f"Preprocessing failed, returning original image: {e}")
            return image

    # ---------------------------
    # TrOCR
    # ---------------------------
    def _ensure_trocr(self) -> Tuple[Optional[TrOCRProcessor], Optional[VisionEncoderDecoderModel]]:
        if self._trocr_processor is None or self._trocr_model is None:
            self.logger.info("Loading TrOCR model and processor...")
            try:
                self._trocr_processor = TrOCRProcessor.from_pretrained(self.config.trocr_model_name)
                self._trocr_model = VisionEncoderDecoderModel.from_pretrained(self.config.trocr_model_name)
                if self.config.device == "cuda" and torch.cuda.is_available():
                    try:
                        self._trocr_model.to("cuda")
                        self.logger.info("Moved TrOCR model to CUDA device.")
                    except Exception as e:
                        self.logger.warning(f"Could not move TrOCR model to CUDA: {e}")
            except Exception as e:
                self.logger.error(f"Failed to load TrOCR model/processor: {e}")
                self._trocr_processor = None
                self._trocr_model = None
        return self._trocr_processor, self._trocr_model

    def extract_with_trocr(self, pages: List[Image.Image]) -> str:
        processor, model = self._ensure_trocr()
        if processor is None or model is None:
            self.logger.error("TrOCR model or processor not available.")
            return ""

        def _trocr_page(idx_page):
            i, page = idx_page
            try:
                img = self._preprocess_image_for_ocr(page)
                pixel_values = processor(img, return_tensors="pt").pixel_values
                if self.config.device == "cuda" and torch.cuda.is_available():
                    try:
                        pixel_values = pixel_values.to("cuda")
                    except Exception:
                        pass
                gen_kwargs = {"max_length": self.config.trocr_gen_max_length}
                generated_ids = model.generate(pixel_values, **gen_kwargs)
                page_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                self.logger.debug(f"TrOCR finished page {i+1}")
                return page_text
            except Exception as e:
                self.logger.error(f"TrOCR failed on page {i+1}: {e}")
                return ""

        pages_enum = list(enumerate(pages))
        if self.config.ocr_parallel and len(pages_enum) > 1:
            workers = min(self.config.max_workers, max(1, len(pages_enum)))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                results = executor.map(_trocr_page, pages_enum)
                texts = list(results)
        else:
            texts = [_trocr_page(x) for x in pages_enum]
        return "\n".join(texts).strip() + ("\n" if texts else "")

    # ---------------------------
    # EasyOCR
    # ---------------------------
    def _ensure_easyocr(self):
        if self._easyocr_reader is None:
            use_gpu = False
            if self.config.easyocr_gpu is not None:
                use_gpu = self.config.easyocr_gpu
            else:
                use_gpu = (self.config.device == "cuda" and torch.cuda.is_available())
            try:
                self.logger.info(f"Initializing EasyOCR reader (gpu={use_gpu})...")
                self._easyocr_reader = easyocr.Reader(['en'], gpu=use_gpu)
            except Exception as e:
                self.logger.error(f"EasyOCR initialization failed: {e}")
                self._easyocr_reader = None
        return self._easyocr_reader

    def extract_with_easyocr(self, pages: List[Image.Image], min_confidence: Optional[float] = None) -> str:
        reader = self._ensure_easyocr()
        if reader is None:
            self.logger.error("EasyOCR reader not available.")
            return ""
        min_conf = min_confidence if min_confidence is not None else self.config.easyocr_min_confidence

        def _easyocr_page(idx_page):
            i, page = idx_page
            try:
                img = self._preprocess_image_for_ocr(page)
                page_array = np.array(img)
                results = reader.readtext(page_array)
                page_text = " ".join([t for (_, t, conf) in results if conf is None or conf > min_conf])
                self.logger.debug(f"EasyOCR finished page {i+1} with {len(results)} results")
                return page_text
            except Exception as e:
                self.logger.error(f"EasyOCR failed on page {i+1}: {e}")
                return ""

        pages_enum = list(enumerate(pages))
        if self.config.ocr_parallel and len(pages_enum) > 1:
            workers = min(self.config.max_workers, max(1, len(pages_enum)))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                results = executor.map(_easyocr_page, pages_enum)
                texts = list(results)
        else:
            texts = [_easyocr_page(x) for x in pages_enum]
        return "\n".join(texts).strip() + ("\n" if texts else "")

    # ---------------------------
    # Tesseract
    # ---------------------------
    def extract_with_tesseract(self, pages: List[Image.Image], tesseract_config: Optional[str] = None) -> str:
        config = tesseract_config or self.config.tesseract_config

        def _ocr_page(idx_page):
            i, page = idx_page
            try:
                img = self._preprocess_image_for_ocr(page)
                page_text = pytesseract.image_to_string(img, config=config)
                self.logger.debug(f"Tesseract finished page {i+1}")
                return page_text
            except Exception as e:
                self.logger.error(f"Tesseract OCR failed on page {i+1}: {e}")
                return ""

        pages_enum = list(enumerate(pages))
        if self.config.ocr_parallel and len(pages_enum) > 1:
            workers = min(self.config.max_workers, max(1, len(pages_enum)))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                results = executor.map(_ocr_page, pages_enum)
                texts = list(results)
        else:
            texts = [_ocr_page(x) for x in pages_enum]

        return "\n".join(texts).strip() + ("\n" if texts else "")