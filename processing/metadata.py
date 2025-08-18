from typing import Dict, Literal
import re

from utils.typing import (
    SECDocument,
    FormType
)


# TODO: Get more metadata such as the various Q&A
def get_sec_metadata(document: SECDocument, form_type: FormType) -> Dict[str, str]:
    """
    Extract SEC metadata. Dynamically changed based on the form type
    """

    # Date patterns
    if form_type == "10K":
        pattern = r"For the fiscal year ended\s+([A-Za-z]+\s+\d{1,2}\s*,\s*\d{4})"
    elif form_type == "10Q":
        pattern = r"For the quarterly period ended\s+([A-Za-z]+\s+\d{1,2}\s*,\s*\d{4})"
    else:
        pattern = ""

    dates = re.findall(pattern, document.text, re.IGNORECASE) if pattern else []
    period_end = dates[1] if len(dates) > 1 else (dates[0] if dates else None)

    # Extract commission file number
    comm_match = re.search(
        r"Commission\s+[Ff]ile\s+(?:[Nn]umber|[Nn]o\.?):?\s*([0-9]{1,3}-[0-9]{5})",
        document.text,
        re.IGNORECASE,
    )
    commission = comm_match.group(1) if comm_match else None

    document.commission_number = commission
    document.period_end = period_end

    return {"period_end": period_end, "commission_number": commission}
