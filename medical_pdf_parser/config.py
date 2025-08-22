from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ParserConfig:
    """
    Centralized configuration. Defaults preserve original behavior.

    New flags to enable optional improvements:
    - ocr_parallel: use ThreadPoolExecutor to OCR pages in parallel (default: False to preserve behavior).
    - allow_numeric_member_id: allow purely-numeric insurance member IDs (default: False to preserve original filtering).
    - patient_section_only: when True, many fields are extracted from the Patient Information section only (default True).
    - easyocr_gpu: if None, auto-detects based on torch.cuda; set explicitly to True/False to override.
    - trocr_gen_max_length: safeguard for TrOCR generation length (default 128).
    """
    dpi: int = 300
    ocr_order: tuple = ("easyocr", "tesseract", "trocr")
    tesseract_config: str = r"--oem 2 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()- :/@#"
    easyocr_min_confidence: float = 0.5
    trocr_model_name: str = "microsoft/trocr-base-printed"
    device: str = "cpu"  # "cpu" or "cuda"
    max_workers: int = 4
    log_to_file: bool = False
    log_file_path: str = "parser.log"
    preprocess_images: bool = False

    # Page selection controls (None means "use whole document")
    page_start: Optional[int] = None   # 1-based index
    page_end: Optional[int] = None     # inclusive, 1-based
    page_limit: Optional[int] = None   # if set and page_start is None, use first `page_limit` pages

    # New tuning flags
    ocr_parallel: bool = False
    allow_numeric_member_id: bool = False
    patient_section_only: bool = True
    easyocr_gpu: Optional[bool] = None
    trocr_gen_max_length: int = 128