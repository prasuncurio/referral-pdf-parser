import sys
import os

# Ensure project root is on sys.path so local package imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- rest of the file ---
from medical_pdf_parser import ParserConfig, MedicalPDFParser

def main():
    # Adjust configuration as needed; this preserves original defaults unless you change flags.
    config = ParserConfig(page_limit=2, preprocess_images=True, ocr_parallel=True)
    parser = MedicalPDFParser(config=config)
    pdf_path = "./data/test2.pdf"  # change to your PDF path
    result = parser.parse_pdf(pdf_path)

    print("Extracted Information:")
    print(f"First Name: {result['first_name']}")
    print(f"Last Name: {result['last_name']}")
    print(f"Email: {result['email']}")
    print(f"Phone Number: {result['phone_number']}")
    print(f"Zip Code: {result['zip_code']}")
    print(f"EPDS Score: {result['epds_score']}")
    print(f"Insurance Member ID: {result['insurance_member_id']}")
    print(f"Secondary Insurance Member ID: {result['secondaryinsurance_member_id']}")

if __name__ == "__main__":
    main()