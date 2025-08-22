import re
from typing import Dict, Optional


def extract_patient_name(text: str) -> Dict[str, Optional[str]]:
    """Extract patient first and last name with additional OCR-tolerant patterns."""
    name_info = {"first_name": None, "last_name": None}
    if not text:
        return name_info

    patterns = [
        r"Patient Name\s*[:\s]*([A-Z]+),\s*([A-Z]+)",     # LASTNAME, FIRSTNAME uppercase
        r"Patient Name\s*[:\s]*([A-Za-z]+)\s+([A-Za-z]+)",# FIRST LAST
        r"Patient\s*[:\s]*Name\s*[:\s]*([A-Za-z]+)\s+([A-Za-z]+)",
        r"Ratient Name\s*[:\s]*([A-Z]+),\s*([A-Z]+)",    # common OCR typo
        r"Ratient Name\s*[:\s]*([A-Za-z]+)\s+([A-Za-z]+)",
        r"Name\s*[:\s]*([A-Za-z]+)\s+([A-Za-z]+)",       # generic "Name" fallback
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            matched_substr = match.group(0)
            if "," in matched_substr:
                name_info["last_name"] = match.group(1).title()
                name_info["first_name"] = match.group(2).title()
            else:
                name_info["first_name"] = match.group(1).title()
                name_info["last_name"] = match.group(2).title()
            break
    return name_info


def extract_email(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', text)
    return match.group(0) if match else None


def extract_phone_number(text: str) -> Optional[str]:
    if not text:
        return None

    patterns = [
        r'(?:Phone|Ph|Tel|T)[\s:]*\+?1?[\s\(.-]*([2-9]\d{2})[\)\s.-]*([2-9]\d{2})[\s.-]*([0-9]{4})',
        r'(\+?\d{1,3})[\s-]\(?(\d{1,4})\)?[\s-](\d{1,4})[\s-](\d{1,9})',
        r'\(?([2-9]\d{2})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            groups = match.groups()
            if len(groups) >= 3 and all(g and g.isdigit() for g in groups[:3]) and len(groups[0]) == 3:
                return f"({groups[0]}) {groups[1]}-{groups[2]}"
            else:
                return " ".join([g for g in groups if g])
    return None


def _is_valid_zip(zip_code: str) -> bool:
    if re.match(r'^\d{5}(-\d{4})?$', zip_code):
        if not zip_code.startswith(('19', '20')):
            return True
    return False


def extract_zip_code(text: str) -> Optional[str]:
    if not text:
        return None
    patterns = [
        r'\b(\d{5}(?:-\d{4})?)\b',
        r'[A-Z]{2}\s+(\d{5}(?:-\d{4})?)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if _is_valid_zip(match):
                return match
    return None


def extract_epds_score(text: str) -> Optional[str]:
    if not text:
        return None

    for m in re.finditer(r'(E\s*P\s*D\s*S|EPDS)', text, re.IGNORECASE):
        start = m.start()
        window = text[start:start + 140]
        num_match = re.search(r'(?:score)?\s*[:=\-\s]{0,4}\s*(\d{1,2})(?:\s*/\s*30)?', window, re.IGNORECASE)
        if num_match:
            try:
                score = int(num_match.group(1))
                if 0 <= score <= 30:
                    return str(score)
            except ValueError:
                pass

    patterns = [
        r'EPDS\s*[:\-]?\s*(?:score\s*)?[:\-]?\s*(\d{1,2})(?:\s*/\s*30)?',
        r'E\s*P\s*D\s*S\s*[:\-]?\s*(\d{1,2})',
        r'EPDS[^\d]{0,6}(\d{1,2})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                score = int(match.group(1))
                if 0 <= score <= 30:
                    return str(score)
            except ValueError:
                continue

    match = re.search(r'(\d{1,2})\s*/\s*30', text)
    if match:
        try:
            score = int(match.group(1))
            if 0 <= score <= 30:
                return str(score)
        except ValueError:
            pass

    return None


def extract_insurance_member_id(text: str, allow_numeric_member_id: bool = False) -> Optional[str]:
    if not text:
        return None
    patterns = [
        r'Member\s*ID[:\s]*([A-Z0-9-]+)',
        r'Insurance\s*ID[:\s]*([A-Z0-9-]+)',
        r'Policy\s*Number[:\s]*([A-Z0-9-]+)',
        r'Member\s*Number[:\s]*([A-Z0-9-]+)',
        r'Primary\s*Insurance[:\s]*([A-Z0-9-]+)',
        r'ID[:\s]*([A-Z0-9-]{4,})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            member_id = match.group(1).strip()
            if allow_numeric_member_id or (not member_id.isdigit()):
                return member_id
    return None


def extract_secondaryinsurance_member_id(text: str, allow_numeric_member_id: bool = False) -> Optional[str]:
    if not text:
        return None
    patterns = [
        r'Secondary\s*Insurance[:\s]*([A-Z0-9-]+)',
        r'Secondary\s*ID[:\s]*([A-Z0-9-]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            member_id = match.group(1).strip()
            if allow_numeric_member_id or (not member_id.isdigit()):
                return member_id
    return None