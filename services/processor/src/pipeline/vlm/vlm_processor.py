# services/processor/src/pipeline/vlm/vlm_processor.py

import asyncio
import aiohttp
import base64
import io
import json
import logging
from typing import Dict, List, Optional, Any, Union
from PIL import Image
import fitz  # PyMuPDF
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class VLMResult:
    """Result from VLM processing."""
    content_type: str
    confidence: float
    layout_description: str
    text_content: str
    visual_elements: List[Dict]
    metadata: Dict[str, Any]
    processing_time: float
    success: bool = True
    error_message: Optional[str] = None

class VLMProcessor:
    """Single-model VLM processor for document analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("vlm", {})
        self.enabled = self.config.get("enabled", False)
        
        if not self.enabled:
            logger.info("VLM processing disabled")
            return
            
        self.api_endpoint = self.config.get("llava", {}).get("api_endpoint", "http://localhost:11434")
        self.model = self.config.get("llava", {}).get("model", "llava:7b")
        self.timeout = self.config.get("llava", {}).get("timeout", 60)
        self.max_retries = self.config.get("llava", {}).get("max_retries", 2)
        
        # Strategy configuration
        self.strategy = self.config.get("strategy", "vlm_first")
        self.fallback_enabled = self.config.get("fallback", {}).get("enabled", True)
        self.fallback_timeout = self.config.get("fallback", {}).get("timeout_threshold", 30)
        self.confidence_threshold = self.config.get("fallback", {}).get("confidence_threshold", 0.5)
        
        # Prompts
        self.universal_prompt = self.config.get("prompts", {}).get("universal", self._default_prompt())
        
        logger.info(f"VLM Processor initialized with model: {self.model}")
    
    def _default_prompt(self) -> str:
        """Default universal prompt for document analysis."""
        return """Analyze this document page and provide structured information.

Identify:
1. Content type: email, presentation, report, chart, table, academic_paper, mixed, or other
2. Layout and structure description
3. All text content (preserve formatting and structure)
4. Visual elements like tables, charts, diagrams, images
5. Your confidence level (0.0 to 1.0)

Respond in JSON format:
{
  "content_type": "string",
  "confidence": 0.0,
  "layout": "detailed layout description",
  "text_content": "all extracted text with structure",
  "visual_elements": [
    {"type": "table", "description": "..."},
    {"type": "chart", "description": "..."}
  ],
  "metadata": {
    "page_structure": "...",
    "formatting_notes": "..."
  }
}

Ensure the JSON is valid and complete."""

    async def process_document(self, pdf_path: str, traditional_result: Optional[Dict] = None) -> Dict[str, Any]:
        """Process entire document with VLM."""
        if not self.enabled:
            logger.warning("VLM processing disabled, using traditional result")
            return traditional_result or {}
        
        try:
            doc = fitz.open(pdf_path)
            pages_results = []
            
            for page_num in range(min(len(doc), 10)):  # Limit pages for development
                page = doc[page_num]
                
                # Convert page to image
                page_image = self._page_to_image(page)
                
                # Get traditional text as fallback
                traditional_text = page.get_text() if traditional_result is None else ""
                
                # Process with VLM
                vlm_result = await self._process_page(page_image, traditional_text, page_num)
                
                pages_results.append({
                    "page_number": page_num,
                    "vlm_result": asdict(vlm_result),
                    "traditional_fallback": traditional_text if not vlm_result.success else None
                })
            
            doc.close()
            
            # Aggregate results
            document_result = self._aggregate_page_results(pages_results)
            
            return {
                "processing_method": "vlm_enhanced",
                "pages": pages_results,
                "document_summary": document_result,
                "vlm_config": {
                    "model": self.model,
                    "strategy": self.strategy
                }
            }
            
        except Exception as e:
            logger.error(f"VLM document processing failed: {str(e)}")
            return traditional_result or {"error": str(e), "processing_method": "failed"}
    
    async def _process_page(self, page_image: Image.Image, fallback_text: str, page_num: int) -> VLMResult:
        """Process single page with VLM."""
        start_time = datetime.now()
        
        try:
            # Call VLM API
            vlm_response = await self._call_vlm_api(page_image)
            
            # Parse response
            result = self._parse_vlm_response(vlm_response, page_num)
            
            # Check if fallback needed
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if (processing_time > self.fallback_timeout or 
                result.confidence < self.confidence_threshold):
                
                if self.fallback_enabled:
                    logger.warning(f"Page {page_num}: Using fallback due to "
                                 f"time={processing_time:.1f}s, confidence={result.confidence:.2f}")
                    return self._create_fallback_result(fallback_text, page_num, processing_time)
            
            result.processing_time = processing_time
            return result
            
        except Exception as e:
            logger.error(f"Page {page_num} VLM processing failed: {str(e)}")
            
            if self.fallback_enabled:
                processing_time = (datetime.now() - start_time).total_seconds()
                return self._create_fallback_result(fallback_text, page_num, processing_time, str(e))
            else:
                return VLMResult(
                    content_type="error",
                    confidence=0.0,
                    layout_description="",
                    text_content="",
                    visual_elements=[],
                    metadata={"error": str(e)},
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    success=False,
                    error_message=str(e)
                )
    
    async def _call_vlm_api(self, image: Image.Image) -> str:
        """Call Ollama LLaVA API."""
        
        # Convert image to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        payload = {
            "model": self.model,
            "prompt": self.universal_prompt,
            "images": [img_base64],
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent results
                "top_p": 0.9
            }
        }
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.max_retries + 1):
                try:
                    async with session.post(
                        f"{self.api_endpoint}/api/generate",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            return result.get("response", "")
                        else:
                            error_text = await response.text()
                            raise Exception(f"API error {response.status}: {error_text}")
                            
                except asyncio.TimeoutError:
                    if attempt < self.max_retries:
                        logger.warning(f"VLM API timeout, attempt {attempt + 1}/{self.max_retries + 1}")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise Exception("VLM API timeout after all retries")
                        
                except Exception as e:
                    if attempt < self.max_retries:
                        logger.warning(f"VLM API error, attempt {attempt + 1}/{self.max_retries + 1}: {str(e)}")
                        await asyncio.sleep(2 ** attempt)
                    else:
                        raise
    
    def _parse_vlm_response(self, response: str, page_num: int) -> VLMResult:
        """Parse VLM JSON response."""
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_text = response[json_start:json_end]
                data = json.loads(json_text)
            else:
                # Fallback parsing if no clear JSON structure
                data = self._fallback_parse(response)
            
            return VLMResult(
                content_type=data.get("content_type", "mixed"),
                confidence=float(data.get("confidence", 0.6)),
                layout_description=data.get("layout", ""),
                text_content=data.get("text_content", ""),
                visual_elements=data.get("visual_elements", []),
                metadata={
                    **data.get("metadata", {}),
                    "page_number": page_num,
                    "raw_response": response[:500]  # Keep sample for debugging
                },
                processing_time=0.0,  # Will be set by caller
                success=True
            )
            
        except Exception as e:
            logger.error(f"Failed to parse VLM response: {str(e)}")
            logger.debug(f"Raw response: {response[:200]}...")
            
            # Create basic result from raw response
            return VLMResult(
                content_type="mixed",
                confidence=0.3,
                layout_description="Parse error",
                text_content=response,  # Use raw response as text
                visual_elements=[],
                metadata={"parse_error": str(e), "page_number": page_num},
                processing_time=0.0,
                success=False,
                error_message=f"Parse error: {str(e)}"
            )
    
    def _fallback_parse(self, response: str) -> Dict:
        """Fallback parsing when JSON extraction fails."""
        return {
            "content_type": "mixed",
            "confidence": 0.4,
            "layout": "Fallback parsing used",
            "text_content": response,
            "visual_elements": [],
            "metadata": {"fallback_parse": True}
        }
    
    def _create_fallback_result(self, fallback_text: str, page_num: int, 
                              processing_time: float, error: Optional[str] = None) -> VLMResult:
        """Create result using traditional fallback."""
        return VLMResult(
            content_type="traditional_fallback",
            confidence=0.5,
            layout_description="Fallback to traditional processing",
            text_content=fallback_text,
            visual_elements=[],
            metadata={
                "page_number": page_num,
                "fallback_reason": error or "timeout/low_confidence",
                "original_processing_time": processing_time
            },
            processing_time=processing_time,
            success=True  # Fallback is still a success
        )
    
    def _page_to_image(self, page: fitz.Page) -> Image.Image:
        """Convert PDF page to PIL Image."""
        # Use 2x scaling for better VLM processing
        matrix = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=matrix)
        img_data = pix.tobytes("png")
        return Image.open(io.BytesIO(img_data))
    
    def _aggregate_page_results(self, pages_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate page results into document summary."""
        
        successful_pages = [p for p in pages_results if p["vlm_result"]["success"]]
        
        if not successful_pages:
            return {"error": "No successful VLM processing", "fallback_used": True}
        
        # Aggregate content types
        content_types = [p["vlm_result"]["content_type"] for p in successful_pages]
        primary_type = max(set(content_types), key=content_types.count) if content_types else "mixed"
        
        # Calculate average confidence
        confidences = [p["vlm_result"]["confidence"] for p in successful_pages]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Combine all text content
        all_text = "\n\n".join([
            p["vlm_result"]["text_content"] 
            for p in successful_pages 
            if p["vlm_result"]["text_content"]
        ])
        
        # Aggregate visual elements
        all_visual_elements = []
        for p in successful_pages:
            all_visual_elements.extend(p["vlm_result"]["visual_elements"])
        
        return {
            "primary_content_type": primary_type,
            "content_type_distribution": dict(zip(*[list(content_types), [content_types.count(ct) for ct in set(content_types)]])),
            "average_confidence": avg_confidence,
            "total_pages_processed": len(pages_results),
            "successful_vlm_pages": len(successful_pages),
            "combined_text_content": all_text,
            "all_visual_elements": all_visual_elements,
            "processing_summary": {
                "strategy_used": self.strategy,
                "fallback_pages": len(pages_results) - len(successful_pages)
            }
        }