# src/pipeline/extractors/email_extractor.py
import re
from typing import Dict, List, Any
from src.models.document import Document

class EmailExtractor:
    """Specialized extractor for email-type PDFs."""
    
    def extract(self, pdf, base_document: Document) -> Document:
        """Extract text and metadata optimized for email-type PDFs."""
        text_content = ""
        
        # Extract text from all pages
        for page in pdf:
            text_content += page.get_text() + "\n"
            
        # Parse email headers and body
        parsed_email = self._parse_email(text_content)
        
        # Update document with parsed email
        base_document.text_content = parsed_email["full_text"]
        
        # Update metadata with email-specific fields
        base_document.metadata.update({
            "email_from": parsed_email.get("from", ""),
            "email_to": parsed_email.get("to", ""),
            "email_cc": parsed_email.get("cc", ""),
            "email_subject": parsed_email.get("subject", ""),
            "email_date": parsed_email.get("date", "")
        })
        
        # If we have a sender email, use it as author
        if parsed_email.get("from") and "@" in parsed_email.get("from"):
            base_document.author = parsed_email.get("from")
            
        # If we have an email date, use it for created_at
        if parsed_email.get("date"):
            from dateutil import parser as date_parser
            try:
                base_document.created_at = date_parser.parse(parsed_email.get("date"))
            except:
                pass
                
        return base_document
    
    def _parse_email(self, text: str) -> Dict[str, Any]:
        """Parse email headers and body from text."""
        result = {
            "full_text": text,
            "body": ""
        }
        
        # Extract common email headers
        header_patterns = {
            "from": r'(?:From|From:)\s+(.*?)(?:\n|\r\n)',
            "to": r'(?:To|To:)\s+(.*?)(?:\n|\r\n)',
            "cc": r'(?:CC|Cc|CC:|Cc:)\s+(.*?)(?:\n|\r\n)',
            "subject": r'(?:Subject|Subject:)\s+(.*?)(?:\n|\r\n)',
            "date": r'(?:Date|Sent|Date:|Sent:)\s+(.*?)(?:\n|\r\n)'
        }
        
        for key, pattern in header_patterns.items():
            match = re.search(pattern, text)
            if match:
                result[key] = match.group(1).strip()
                
        # Simple heuristic to find email body - look for blank line after headers
        header_section_end = 0
        for header in ["from", "to", "subject", "date"]:
            if header in result:
                pattern = header_patterns[header]
                match = re.search(pattern, text)
                if match:
                    end_pos = match.end()
                    if end_pos > header_section_end:
                        header_section_end = end_pos
        
        if header_section_end > 0:
            # Find next blank line after headers
            blank_line_match = re.search(r'\n\s*\n', text[header_section_end:])
            if blank_line_match:
                body_start = header_section_end + blank_line_match.end()
                result["body"] = text[body_start:].strip()
            else:
                result["body"] = text[header_section_end:].strip()
        
        # Look for email thread markers
        thread_markers = [
            r'----+ ?Original Message ?----+',
            r'----+ ?Forwarded Message ?----+',
            r'On .* wrote:',
            r'From:.*Sent:.*To:.*Subject:'
        ]
        
        thread_parts = [result["body"]]
        for marker in thread_markers:
            new_parts = []
            for part in thread_parts:
                splits = re.split(marker, part)
                new_parts.extend(splits)
            thread_parts = new_parts
            
        if len(thread_parts) > 1:
            result["thread_parts"] = thread_parts
            
        return result