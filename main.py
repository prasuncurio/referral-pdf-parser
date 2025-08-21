import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import pdf2image
import re
from typing import Dict, Optional, List
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import logging
import os
import easyocr
import numpy as np

class MedicalPDFParser:
    def __init__(self):
        self.setup_logging()
        
    def extract_text_with_ocr(self, pdf_path: str) -> str:
        """Extract text from scanned PDF using OCR"""
        text = ""
        
        try:
            # Convert PDF pages to images
            self.logger.info("Converting PDF to images for OCR...")
            pages = pdf2image.convert_from_path(pdf_path, dpi=300)
            
            for i, page in enumerate(pages):
                self.logger.info(f"Processing page {i+1} with OCR...")
                
                # Use Tesseract to extract text from image
                # Configure Tesseract for better medical document recognition
                custom_config = r'--oem 2 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()- :/@#'
                page_text = pytesseract.image_to_string(page, config=custom_config)
                text += page_text + "\n"
            
        except Exception as e:
            self.logger.error(f"OCR processing failed: {e}")
            # Fallback: Try basic OCR without custom config
            try:
                pages = pdf2image.convert_from_path(pdf_path)
                for page in pages:
                    page_text = pytesseract.image_to_string(page)
                    text += page_text + "\n"
            except Exception as e2:
                self.logger.error(f"Fallback OCR also failed: {e2}")
        
        return text
    
    def extract_text_with_trocr(self, pdf_path: str) -> str:
        """Extract text from scanned PDF using TrOCR"""
        text = ""
        
        try:
            # Load TrOCR model and processor
            self.logger.info("Loading TrOCR model (this may take a moment on first run)...")
            processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
            model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
            
            # Convert PDF pages to images (reuse your existing method)
            self.logger.info("Converting PDF to images for TrOCR...")
            pages = pdf2image.convert_from_path(pdf_path, dpi=300)
            
            for i, page in enumerate(pages):
                self.logger.info(f"Processing page {i+1} with TrOCR...")
                
                # TrOCR works better with smaller image chunks for documents
                # You can process the full page or split it into sections
                pixel_values = processor(page, return_tensors="pt").pixel_values
                generated_ids = model.generate(pixel_values)
                page_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                text += page_text + "\n"
                
        except Exception as e:
            self.logger.error(f"TrOCR processing failed: {e}")
        
        return text
    
    def extract_text_with_easyocr(self, pdf_path: str) -> str:
        """Extract text from scanned PDF using EasyOCR"""
        text = ""
        
        try:
            # Initialize EasyOCR reader (downloads models on first run)
            self.logger.info("Initializing EasyOCR (this may take a moment on first run)...")
            reader = easyocr.Reader(['en'])  # English only
            
            # Convert PDF pages to images
            self.logger.info("Converting PDF to images for EasyOCR...")
            pages = pdf2image.convert_from_path(pdf_path, dpi=300)
            
            for i, page in enumerate(pages):
                self.logger.info(f"Processing page {i+1} with EasyOCR...")

                page_array = np.array(page)
                
                # EasyOCR can work directly with PIL images
                results = reader.readtext(page_array)
                
                # Extract text from results
                page_text = ""
                for (bbox, text_content, confidence) in results:
                    # Only include text with reasonable confidence
                    if confidence > 0.5:  # Adjust threshold as needed
                        page_text += text_content + " "
                
                text += page_text + "\n"
                
        except Exception as e:
            self.logger.error(f"EasyOCR processing failed: {e}")
        
        return text

    def setup_logging(self):
        """Setup logging for debugging purposes"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file using multiple methods including OCR"""
        text = ""
        
        # Method 1: Try PyMuPDF (fitz) - often works better
        try:
            doc = fitz.open(pdf_path)
            self.logger.info(f'Num pages: {len(doc)}')
            for i, page in enumerate(doc):
                self.logger.info(f'Page {i+1} text length: {len(page.get_text())}')
                self.logger.info(f'Page {i+1} has images: {len(page.get_images())}')
                text += page.get_text() + "\n"
            doc.close()
            if text.strip():
                self.logger.info("Successfully extracted text using PyMuPDF")
                return text
        except Exception as e:
            self.logger.warning(f"PyMuPDF extraction failed: {e}")
        
        # Method 2: Try pdfplumber - good for structured data
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                self.logger.info("Successfully extracted text using pdfplumber")
                return text
        except Exception as e:
            self.logger.warning(f"pdfplumber extraction failed: {e}")
        
        # Method 3: Try PyPDF2 as fallback
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                self.logger.info("Successfully extracted text using PyPDF2")
                return text
        except Exception as e:
            self.logger.warning(f"PyPDF2 extraction failed: {e}")
        
        # Method 4: OCR for scanned PDFs
        self.logger.info("Standard PDF extraction failed. Attempting OCR...")
        try:
            text = self.extract_text_with_easyocr(pdf_path)
            if text.strip():
                self.logger.info("Successfully extracted text using OCR")
                return text
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
        
        # If all methods fail
        if not text.strip():
            self.logger.error("All PDF extraction methods failed including OCR.")
            
        return text
    
    def extract_patient_name(self, text: str) -> Dict[str, Optional[str]]:
        """Extract first and last name from patient information"""
        name_info = {"first_name": None, "last_name": None}
        
        # Pattern for "Patient Name" field
        patient_name_patterns = [
            r"Patient Name\s*[:\s]*([A-Z]+),\s*([A-Z]+)",  # LASTNAME, FIRSTNAME
            r"Patient Name\s*[:\s]*([A-Za-z]+)\s+([A-Za-z]+)",  # FIRSTNAME LASTNAME
        ]
        
        for pattern in patient_name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if "," in pattern:  # Last name first format
                    name_info["last_name"] = match.group(1).title()
                    name_info["first_name"] = match.group(2).title()
                else:  # First name first format
                    name_info["first_name"] = match.group(1).title()
                    name_info["last_name"] = match.group(2).title()
                break
        
        return name_info
    
    def extract_email(self, text: str) -> Optional[str]:
        """Extract email address from text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        match = re.search(email_pattern, text)
        return match.group(0) if match else None
    
    def extract_phone_number(self, text: str) -> Optional[str]:
        """Extract phone number from text"""
        # Look for phone numbers in various formats
        phone_patterns = [
            r'Phone[:\s]*\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})',
            r'Ph[:\s]*\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})',
            r'H[:\s]*\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})',
            r'M[:\s]*\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})',
        ]
        
        for pattern in phone_patterns:
            match = re.search(pattern, text)
            if match:
                return f"({match.group(1)}) {match.group(2)}-{match.group(3)}"
        
        return None
    
    def extract_zip_code(self, text: str) -> Optional[str]:
        """Extract zip code from address information"""
        # Look for 5-digit zip codes, optionally followed by 4-digit extension
        zip_patterns = [
            r'\b(\d{5}(?:-\d{4})?)\b',
            r'[A-Z]{2}\s+(\d{5}(?:-\d{4})?)',  # State abbreviation followed by zip
        ]
        
        for pattern in zip_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Filter out obvious non-zip codes (like years, IDs, etc.)
                zip_code = match if isinstance(match, str) else match
                if self._is_valid_zip(zip_code):
                    return zip_code
        
        return None
    
    def _is_valid_zip(self, zip_code: str) -> bool:
        """Validate if a string looks like a valid US zip code"""
        # Basic validation for US zip codes
        if re.match(r'^\d{5}(-\d{4})?$', zip_code):
            # Exclude obvious non-zip codes
            if not zip_code.startswith(('19', '20')):  # Exclude years
                return True
        return False
    
    def extract_epds_score(self, text: str) -> Optional[str]:
        """Extract EPDS score from text"""
        epds_patterns = [
            r'EPDS\s+score\s*[:\s]*(\d+)',
            r'EPDS[:\s]*(\d+)',
            r'score\s*[:\s]*(\d+)',  # Generic score pattern near EPDS
        ]
        
        for pattern in epds_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                # EPDS scores are typically 0-30
                if 0 <= score <= 30:
                    return str(score)
        
        return None
    
    def extract_insurance_member_id(self, text: str) -> Optional[str]:
        """Extract insurance member ID from text"""
        # Look for various insurance-related ID patterns
        insurance_patterns = [
            r'Member\s*ID[:\s]*([A-Z0-9]+)',
            r'Insurance\s*ID[:\s]*([A-Z0-9]+)',
            r'Policy\s*Number[:\s]*([A-Z0-9]+)',
            r'Member\s*Number[:\s]*([A-Z0-9]+)',
            r'ID[:\s]*([A-Z0-9]{6,})',  # Generic ID pattern with minimum length
        ]
        
        for pattern in insurance_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                member_id = match.group(1)
                # Filter out obvious non-insurance IDs
                if len(member_id) >= 4 and not member_id.isdigit():
                    return member_id
        
        return None
    
    def parse_pdf(self, pdf_path: str) -> Dict[str, Optional[str]]:
        """Main method to parse PDF and extract all required information"""
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            self.logger.error("No text extracted from PDF")
            return self._empty_result()
        
        # Debug: Print first 500 characters of extracted text
        self.logger.info(f"Extracted text preview: {text[:500]}...")

        # extract patient information section
        patient_section_start = text.find("Patient Information")
        patient_section_end = text.find("Electronically Signed")
        patient_text = text[patient_section_start:patient_section_end] if patient_section_start != -1 else ""
        self.logger.info(f'Patient Info text: {patient_text}')

        # Referral Order Section
        referral_order_info_start = text.find("Referral Order Information")
        referral_order_info_end = text.find("Patient information")
        referral_order_info_text = text[referral_order_info_start:referral_order_info_end] if referral_order_info_start != -1 else ""
        self.logger.info(f'Referral Order Info text: {referral_order_info_text}')
        
        # Extract name information
        name_info = self.extract_patient_name(patient_text)
        
        result = {
            "first_name": name_info["first_name"],
            "last_name": name_info["last_name"],
            "email": self.extract_email(patient_text),
            "phone_number": self.extract_phone_number(patient_text),
            "zip_code": self.extract_zip_code(patient_text),
            "epds_score": self.extract_epds_score(referral_order_info_text),
            "insurance_member_id": self.extract_insurance_member_id(patient_text)
        }
        
        return result
    
    def _empty_result(self) -> Dict[str, Optional[str]]:
        """Return empty result dictionary"""
        return {
            "first_name": None,
            "last_name": None,
            "email": None,
            "phone_number": None,
            "zip_code": None,
            "epds_score": None,
            "insurance_member_id": None
        }
    
    def parse_multiple_pdfs(self, pdf_paths: List[str]) -> List[Dict[str, Optional[str]]]:
        """Parse multiple PDFs and return list of results"""
        results = []
        for pdf_path in pdf_paths:
            self.logger.info(f"Parsing {pdf_path}")
            result = self.parse_pdf(pdf_path)
            result["file_path"] = pdf_path
            results.append(result)
        return results


# Example usage and testing
def main():
    parser = MedicalPDFParser()
    
    # Example usage for single PDF
    pdf_path = "./data/test1.pdf"
    result = parser.parse_pdf(pdf_path)
    
    print("Extracted Information:")
    print(f"First Name: {result['first_name']}")
    print(f"Last Name: {result['last_name']}")
    print(f"Email: {result['email']}")
    print(f"Phone Number: {result['phone_number']}")
    print(f"Zip Code: {result['zip_code']}")
    print(f"EPDS Score: {result['epds_score']}")
    print(f"Insurance Member ID: {result['insurance_member_id']}")
    
    # Example usage for multiple PDFs
    # pdf_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
    # results = parser.parse_multiple_pdfs(pdf_files)
    
    # for i, result in enumerate(results):
    #     print(f"\nDocument {i+1} ({result['file_path']}):")
    #     for key, value in result.items():
    #         if key != 'file_path':
    #             print(f"  {key}: {value}")


if __name__ == "__main__":
    main()