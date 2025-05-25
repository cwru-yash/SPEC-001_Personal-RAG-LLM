# services/processor/src/pipeline/extractors/text_extractor.py
import os
import uuid
import csv
import json
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path

from ...models.document import Document

class TextExtractor:
    """Extractor for text-based files including CSV, JSON, TXT, etc."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with configuration."""
        self.config = config or {}
        self.encoding_fallbacks = self.config.get("encoding_fallbacks", 
                                                  ["utf-8", "latin-1", "cp1252", "ascii"])
        self.csv_config = self.config.get("csv", {})
        self.json_config = self.config.get("json", {})
    
    def extract(self, file_path: str) -> Document:
        """Extract text from various text file formats."""
        file_ext = Path(file_path).suffix[1:].lower()
        
        if file_ext == "csv":
            return self._extract_csv(file_path)
        elif file_ext == "json":
            return self._extract_json(file_path)
        elif file_ext in ["txt", "md", "log"]:
            return self._extract_plain_text(file_path)
        elif file_ext in ["xml", "html", "htm"]:
            return self._extract_markup(file_path)
        else:
            # Default to plain text extraction
            return self._extract_plain_text(file_path)
    
    def _read_file_with_fallback(self, file_path: str) -> str:
        """Read file with encoding fallback."""
        for encoding in self.encoding_fallbacks:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, read with error replacement
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    
    def _extract_csv(self, file_path: str) -> Document:
        """Extract content from CSV files."""
        doc_id = str(uuid.uuid4())
        file_name = os.path.basename(file_path)
        
        try:
            content = self._read_file_with_fallback(file_path)
            
            # Parse CSV
            csv_reader = csv.DictReader(content.splitlines())
            headers = csv_reader.fieldnames or []
            
            # Extract preview and statistics
            rows = list(csv_reader)
            row_count = len(rows)
            
            max_preview_rows = self.csv_config.get("max_preview_rows", 100)
            preview_rows = rows[:max_preview_rows]
            
            # Create structured text representation
            text_parts = [f"CSV File: {file_name}"]
            text_parts.append(f"Headers: {', '.join(headers)}")
            text_parts.append(f"Total Rows: {row_count}")
            text_parts.append("\n=== DATA PREVIEW ===")
            
            # Add header row
            if headers:
                text_parts.append(" | ".join(headers))
                text_parts.append("-" * (len(" | ".join(headers))))
            
            # Add preview rows
            for row_dict in preview_rows:
                row_values = [str(row_dict.get(h, "")) for h in headers]
                text_parts.append(" | ".join(row_values))
            
            if row_count > max_preview_rows:
                text_parts.append(f"\n... ({row_count - max_preview_rows} more rows)")
            
            # Calculate column statistics
            column_stats = {}
            for header in headers:
                non_empty = sum(1 for row in rows if row.get(header, "").strip())
                column_stats[header] = {
                    "non_empty_count": non_empty,
                    "empty_count": row_count - non_empty,
                    "fill_rate": non_empty / row_count if row_count > 0 else 0
                }
            
            return Document(
                doc_id=doc_id,
                file_name=file_name,
                file_extension="csv",
                content_type=["spreadsheet", "csv", "tabular"],
                created_at=datetime.now(),
                text_content="\n".join(text_parts),
                metadata={
                    "row_count": row_count,
                    "column_count": len(headers),
                    "headers": headers,
                    "column_statistics": column_stats,
                    "extraction_method": "csv_reader",
                    "delimiter": ",",  # Could be detected
                    "preview_rows": max_preview_rows
                }
            )
            
        except Exception as e:
            return Document(
                doc_id=doc_id,
                file_name=file_name,
                file_extension="csv",
                content_type=["spreadsheet", "csv", "error"],
                created_at=datetime.now(),
                text_content=f"Error extracting CSV content: {str(e)}",
                metadata={"extraction_error": str(e)}
            )
    
    def _extract_json(self, file_path: str) -> Document:
        """Extract content from JSON files."""
        doc_id = str(uuid.uuid4())
        file_name = os.path.basename(file_path)
        
        try:
            content = self._read_file_with_fallback(file_path)
            
            # Parse JSON
            json_data = json.loads(content)
            
            # Create readable representation
            if self.json_config.get("preserve_structure", True):
                text_content = json.dumps(json_data, indent=2, ensure_ascii=False)
            else:
                # Flatten to text
                text_parts = [f"JSON File: {file_name}"]
                text_parts.extend(self._flatten_json(json_data))
                text_content = "\n".join(text_parts)
            
            # Extract structure info
            structure_info = self._analyze_json_structure(json_data)
            
            return Document(
                doc_id=doc_id,
                file_name=file_name,
                file_extension="json",
                content_type=["data", "json", "structured"],
                created_at=datetime.now(),
                text_content=text_content,
                metadata={
                    "root_type": type(json_data).__name__,
                    "structure": structure_info,
                    "extraction_method": "json_parser"
                }
            )
            
        except Exception as e:
            return Document(
                doc_id=doc_id,
                file_name=file_name,
                file_extension="json",
                content_type=["data", "json", "error"],
                created_at=datetime.now(),
                text_content=f"Error extracting JSON content: {str(e)}",
                metadata={"extraction_error": str(e)}
            )
    
    def _extract_plain_text(self, file_path: str) -> Document:
        """Extract content from plain text files."""
        doc_id = str(uuid.uuid4())
        file_name = os.path.basename(file_path)
        file_ext = Path(file_path).suffix[1:].lower()
        
        try:
            content = self._read_file_with_fallback(file_path)
            
            # Basic text statistics
            lines = content.splitlines()
            non_empty_lines = [line for line in lines if line.strip()]
            
            return Document(
                doc_id=doc_id,
                file_name=file_name,
                file_extension=file_ext,
                content_type=["text", file_ext],
                created_at=datetime.now(),
                text_content=content,
                metadata={
                    "line_count": len(lines),
                    "non_empty_lines": len(non_empty_lines),
                    "character_count": len(content),
                    "extraction_method": "plain_text"
                }
            )
            
        except Exception as e:
            return Document(
                doc_id=doc_id,
                file_name=file_name,
                file_extension=file_ext,
                content_type=["text", file_ext, "error"],
                created_at=datetime.now(),
                text_content=f"Error reading text file: {str(e)}",
                metadata={"extraction_error": str(e)}
            )
    
    def _extract_markup(self, file_path: str) -> Document:
        """Extract content from markup files (XML, HTML)."""
        doc_id = str(uuid.uuid4())
        file_name = os.path.basename(file_path)
        file_ext = Path(file_path).suffix[1:].lower()
        
        try:
            content = self._read_file_with_fallback(file_path)
            
            # For now, preserve the markup
            # Could add BeautifulSoup parsing here if needed
            
            return Document(
                doc_id=doc_id,
                file_name=file_name,
                file_extension=file_ext,
                content_type=["markup", file_ext],
                created_at=datetime.now(),
                text_content=content,
                metadata={
                    "has_tags": "<" in content and ">" in content,
                    "extraction_method": "markup_raw"
                }
            )
            
        except Exception as e:
            return Document(
                doc_id=doc_id,
                file_name=file_name,
                file_extension=file_ext,
                content_type=["markup", file_ext, "error"],
                created_at=datetime.now(),
                text_content=f"Error reading markup file: {str(e)}",
                metadata={"extraction_error": str(e)}
            )
    
    def _flatten_json(self, obj: Any, prefix: str = "") -> List[str]:
        """Flatten JSON object to text lines."""
        lines = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    lines.extend(self._flatten_json(value, new_prefix))
                else:
                    lines.append(f"{new_prefix}: {value}")
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                new_prefix = f"{prefix}[{idx}]"
                if isinstance(item, (dict, list)):
                    lines.extend(self._flatten_json(item, new_prefix))
                else:
                    lines.append(f"{new_prefix}: {item}")
        else:
            lines.append(f"{prefix}: {obj}")
        
        return lines
    
    def _analyze_json_structure(self, obj: Any, max_depth: int = 3, current_depth: int = 0) -> Dict[str, Any]:
        """Analyze JSON structure."""
        if current_depth >= max_depth:
            return {"type": type(obj).__name__, "truncated": True}
        
        if isinstance(obj, dict):
            return {
                "type": "object",
                "keys": list(obj.keys())[:10],  # First 10 keys
                "key_count": len(obj)
            }
        elif isinstance(obj, list):
            return {
                "type": "array",
                "length": len(obj),
                "item_types": list(set(type(item).__name__ for item in obj[:10]))
            }
        else:
            return {"type": type(obj).__name__}