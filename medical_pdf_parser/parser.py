from typing import Dict, Optional, List
from .config import ParserConfig
from .logger import setup_logger
from .text_extractors import TextExtractor
from .fields import (
    extract_patient_name,
    extract_email,
    extract_phone_number,
    extract_zip_code,
    extract_epds_score,
    extract_insurance_member_id,
    extract_secondaryinsurance_member_id
)


class MedicalPDFParser:
    """
    Extracts structured information from medical PDFs using text extraction and OCR.

    Backwards-compatible with the original public API:
      - ParserConfig stays the same name and default values that preserve original behavior.
      - parse_pdf(pdf_path) returns the same keys as before.
    """

    def __init__(self, config: Optional[ParserConfig] = None):
        self.config = config or ParserConfig()
        self.logger = setup_logger(self.config)
        self.text_extractor = TextExtractor(self.config)

    def _empty_result(self) -> Dict[str, Optional[str]]:
        """Return empty result dictionary."""
        return {
            "first_name": None,
            "last_name": None,
            "email": None,
            "phone_number": None,
            "zip_code": None,
            "epds_score": None,
            "insurance_member_id": None,
            "secondaryinsurance_member_id": None
        }

    def parse_pdf(self, pdf_path: str) -> Dict[str, Optional[str]]:
        """
        Parse a PDF and extract structured information.
        Public API preserved exactly (same keys returned).
        """
        text = self.text_extractor.extract_text_from_pdf(pdf_path)
        if not text.strip():
            self.logger.error("No text extracted from PDF.")
            return self._empty_result()
        self.logger.info(f"Extracted text preview: {text[:500]}...")

        # Attempt to extract sections.
        patient_section_start = text.find("Patient Information")
        patient_section_end = text.find("Electronically Signed")
        patient_text = text[patient_section_start:patient_section_end] if patient_section_start != -1 else ""
        if not self.config.patient_section_only:
            patient_text = text

        referral_order_info_start = text.find("Referral Order Information")
        referral_order_info_end = text.find("Patient Information")
        referral_order_info_text = text[referral_order_info_start:referral_order_info_end] if referral_order_info_start != -1 else ""

        name_info = extract_patient_name(patient_text)

        epds_score = extract_epds_score(referral_order_info_text)
        if epds_score is None:
            epds_score = extract_epds_score(text)

        result = {
            "first_name": name_info["first_name"],
            "last_name": name_info["last_name"],
            "email": extract_email(patient_text),
            "phone_number": extract_phone_number(patient_text),
            "zip_code": extract_zip_code(patient_text),
            "epds_score": epds_score,
            "insurance_member_id": extract_insurance_member_id(patient_text, allow_numeric_member_id=self.config.allow_numeric_member_id),
            "secondaryinsurance_member_id": extract_secondaryinsurance_member_id(patient_text, allow_numeric_member_id=self.config.allow_numeric_member_id)
        }
        return result

    def parse_multiple_pdfs(self, pdf_paths: List[str]) -> List[Dict[str, Optional[str]]]:
        results = []
        for pdf_path in pdf_paths:
            self.logger.info(f"Parsing {pdf_path}")
            result = self.parse_pdf(pdf_path)
            result["file_path"] = pdf_path
            results.append(result)
        return results