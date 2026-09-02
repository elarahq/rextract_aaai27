import re
from datetime import datetime
from typing import Optional

# YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD (ISO and ISO-like — already normalized, pass through)
_DATE_ISO = re.compile(r"^(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})$")
# DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
_DATE_NUMERIC = re.compile(r"^(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})$")
# DD/MM/YY, DD-MM-YY (2-digit year: 00-29 → 2000-2029, 30-99 → 1930-1999)
_DATE_NUMERIC_2Y = re.compile(r"^(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2})$")
# Ordinal suffix stripper: "30th" → "30", "1st" → "1"
_ORDINAL_RE = re.compile(r"(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)
_WRITTEN_FORMATS = [
    "%d %B %Y",   # 30 September 2023
    "%d %b %Y",   # 30 Sep 2023
    "%B %d %Y",   # September 30 2023
    "%b %d %Y",   # Sep 30 2023
    "%d/%b/%Y",   # 24/FEB/2023
    "%d-%b-%Y",   # 24-FEB-2023
    "%d/%b-%Y",   # 24/FEB-2023
]


def _try_parse_date(value: str) -> Optional[str]:
    """Return YYYY-MM-DD if value is a recognisable date string, else None."""
    if not value or not isinstance(value, str):
        return None
    v = value.strip()

    # ISO: YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD (VLM sometimes outputs these directly)
    iso = _DATE_ISO.match(v)
    if iso:
        yyyy, mm, dd = iso.group(1), iso.group(2).zfill(2), iso.group(3).zfill(2)
        return f"{yyyy}-{mm}-{dd}"

    # Numeric: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
    m = _DATE_NUMERIC.match(v)
    if m:
        dd, mm, yyyy = m.group(1).zfill(2), m.group(2).zfill(2), m.group(3)
        return f"{yyyy}-{mm}-{dd}"

    # Numeric: DD/MM/YY, DD-MM-YY (2-digit year)
    m2 = _DATE_NUMERIC_2Y.match(v)
    if m2:
        dd, mm, yy = m2.group(1).zfill(2), m2.group(2).zfill(2), m2.group(3)
        yyyy = f"20{yy}" if int(yy) <= 29 else f"19{yy}"
        return f"{yyyy}-{mm}-{dd}"

    # Written: strip ordinal suffixes and commas, then try strptime
    cleaned = " ".join(_ORDINAL_RE.sub(r"\1", v).replace(",", " ").split())
    for fmt in _WRITTEN_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    return None


def _normalize_dates(fields: dict) -> dict:
    """Normalize all date-like string values in extracted fields to YYYY-MM-DD."""
    result = {}
    for key, value in fields.items():
        if isinstance(value, str):
            parsed = _try_parse_date(value)
            if parsed:
                result[key] = parsed
                continue
        result[key] = value
    return result
