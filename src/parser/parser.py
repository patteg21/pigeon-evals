from typing import Dict, Optional
import re

from evals.src.utils.types.sec_files import SECDocument, SECMetadata

class SECDataParser:

    def __init__(self):
        pass

    def process(self, document: SECDocument) -> None:
        """Extract comprehensive SEC metadata from 10-K/10-Q documents."""
        
        metadata = {}
        text = document.text.split("[PAGE BREAK]")[0] # Just uses the first page
        form_type = document.form_type
        
        # Extract company name
        company_match = re.search(
            r"^([A-Z][A-Za-z\s\.,&]+(?:Inc\.|Corp\.|Corporation|Company|Co\.|LLC|Ltd\.|Limited)?)$",
            text,
            re.MULTILINE
        )
        metadata["company_name"] = company_match.group(1).strip() if company_match else None
        
        # Extract fiscal year end date
        if form_type == "10K":
            date_pattern = r"For the fiscal year ended\s+([A-Za-z]+\s+\d{1,2}\s*,\s*\d{4})"
        elif form_type == "10Q":
            date_pattern = r"For the quarterly period ended\s+([A-Za-z]+\s+\d{1,2}\s*,\s*\d{4})"
        else:
            date_pattern = r"For the (?:fiscal year|quarterly period) ended\s+([A-Za-z]+\s+\d{1,2}\s*,\s*\d{4})"

        date_matches = re.findall(date_pattern, text, re.IGNORECASE)
        metadata["period_end"] = date_matches[0] if date_matches else None
        
        # Extract commission file number
        commission_match = re.search(
            r"Commission\s+[Ff]ile\s+(?:[Nn]umber|[Nn]o\.?):?\s*([0-9]{1,3}-[0-9]{5})",
            text,
            re.IGNORECASE
        )
        metadata["commission_number"] = commission_match.group(1) if commission_match else None
        
        # Extract state of incorporation
        state_match = re.search(
            r"\|\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\|\s*\d{2}-\d{7}",
            text
        )
        metadata["state_of_incorporation"] = state_match.group(1) if state_match else None
        
        # Extract EIN (Employer Identification Number)
        ein_match = re.search(
            r"([0-9]{2}-[0-9]{7})",
            text
        )
        metadata["ein"] = ein_match.group(1) if ein_match else None
        
        # Extract principal executive offices address
        address_pattern = r"([^|]+)\s*\|\s*([A-Za-z\s]+)\s*,?\s*([A-Za-z]+)\s*\|\s*(\d{5})"
        address_match = re.search(address_pattern, text)
        if address_match:
            metadata["address"] = address_match.group(1).strip()
            metadata["city"] = address_match.group(2).strip()
            metadata["state"] = address_match.group(3).strip()
            metadata["zip_code"] = address_match.group(4).strip()
        else:
            metadata["address"] = None
            metadata["city"] = None
            metadata["state"] = None
            metadata["zip_code"] = None
        
        # Extract phone number
        phone_match = re.search(
            r"\(\s*(\d{3})\s*\)\s*(\d{3})-(\d{4})",
            text
        )
        metadata["phone"] = f"({phone_match.group(1)}) {phone_match.group(2)}-{phone_match.group(3)}" if phone_match else None
    
        
        # Extract market value of non-voting stock
        market_value_match = re.search(
            r"approximately\s*\$\s*([\d,]+,[\d,]+,[\d,]+)",
            text
        )
        metadata["market_value"] = market_value_match.group(1) if market_value_match else None
        
        # Extract shares outstanding
        shares_match = re.search(
            r"([0-9,]+,[\d,]+,[\d,]+)\s+shares of common stock were issued and outstanding",
            text
        )
        metadata["shares_outstanding"] = shares_match.group(1) if shares_match else None
        
        # Extract outstanding shares date
        shares_date_match = re.search(
            r"as of ([A-Za-z]+ \d{1,2}, \d{4})\.",
            text
        )
        metadata["shares_outstanding_date"] = shares_date_match.group(1) if shares_date_match else None
        
        # Extract filer status
        filer_status = None
        if "Large accelerated filer" in text and "☒" in text:
            filer_status = "Large accelerated filer"
        elif "Accelerated filer" in text and "☒" in text:
            filer_status = "Accelerated filer"
        elif "Non-accelerated filer" in text and "☒" in text:
            filer_status = "Non-accelerated filer"
        elif "Smaller reporting company" in text and "☒" in text:
            filer_status = "Smaller reporting company"
            
        metadata["filer_status"] = filer_status
        
        # Create SECMetadata object
        sec_metadata = SECMetadata(
            company_name=metadata.get("company_name"),
            period_end=metadata.get("period_end"),
            commission_number=metadata.get("commission_number"),
            state_of_incorporation=metadata.get("state_of_incorporation"),
            ein=metadata.get("ein"),
            address=metadata.get("address"),
            city=metadata.get("city"),
            state=metadata.get("state"),
            zip_code=metadata.get("zip_code"),
            phone=metadata.get("phone"),
            trading_symbol=metadata.get("trading_symbol"),
            market_value=metadata.get("market_value"),
            shares_outstanding=metadata.get("shares_outstanding"),
            shares_outstanding_date=metadata.get("shares_outstanding_date"),
            filer_status=metadata.get("filer_status")
        )
        
        # Update document with metadata
        document.sec_metadata = sec_metadata
        
        # Update legacy document attributes for backward compatibility
        if metadata.get("company_name"):
            document.company = metadata["company_name"]
