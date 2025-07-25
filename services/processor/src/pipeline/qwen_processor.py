# services/processor/src/pipeline/qwen_processor.py

import asyncio
import aiohttp
import base64
import io
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from PIL import Image
import fitz

# Import existing VLM result structure
from pipeline.vlm.cpu_optimized_processor import VLMResult

logger = logging.getLogger(__name__)

class Qwen25VLProcessor:
    """Qwen 2.5 VL processor optimized for document understanding."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('vlm', {}).get('local_models', {}).get('qwen25vl', {})
        self.enabled = self.config.get('enabled', False)
        
        if not self.enabled:
            logger.warning("Qwen 2.5 VL processor disabled")
            return
        
        # Qwen configuration
        self.api_endpoint = self.config.get('api_endpoint', 'http://localhost:11434')
        self.model = self.config.get('model', 'qwen2.5-vl:32b')
        self.timeout = self.config.get('timeout', 120)
        self.max_retries = self.config.get('max_retries', 2)
        
        # Document-specific optimizations for Qwen
        self.image_size_limit = self.config.get('image_size_limit', [1024, 1024])
        self.max_pages_per_document = self.config.get('max_pages_per_document', 5)
        
        logger.info(f"Qwen 2.5 VL processor initialized:")
        logger.info(f"  Model: {self.model}")
        logger.info(f"  API endpoint: {self.api_endpoint}")
        logger.info(f"  Timeout: {self.timeout}s")
    
    async def process_document_with_progress(self, pdf_path: str) -> Dict[str, Any]:
        """Process document using Qwen 2.5 VL with progress tracking."""
        
        if not self.enabled:
            raise Exception("Qwen 2.5 VL processor not enabled")
        
        start_time = datetime.now()
        logger.info(f"Starting Qwen 2.5 VL processing of: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            pages_to_process = min(total_pages, self.max_pages_per_document)
            
            logger.info(f"Document has {total_pages} pages, processing first {pages_to_process} pages")
            
            pages_results = []
            successful_pages = 0
            
            for page_num in range(pages_to_process):
                page_start_time = datetime.now()
                logger.info(f"Processing page {page_num + 1}/{pages_to_process} with Qwen 2.5 VL...")
                
                page = doc[page_num]
                page_image = self._page_to_optimized_image(page)
                
                # Process with Qwen 2.5 VL
                page_result = await self._process_page_with_qwen(page_image, page_num)
                
                page_time = (datetime.now() - page_start_time).total_seconds()
                
                if page_result.success:
                    successful_pages += 1
                    logger.info(f"✅ Page {page_num + 1} processed successfully in {page_time:.1f}s")
                    logger.info(f"   Content type: {page_result.content_type}, Confidence: {page_result.confidence:.2f}")
                else:
                    logger.warning(f"⚠️ Page {page_num + 1} processing failed: {page_result.error_message}")
                
                pages_results.append({
                    'page_number': page_num,
                    'vlm_result': {
                        'success': page_result.success,
                        'content_type': page_result.content_type,
                        'confidence': page_result.confidence,
                        'layout_description': page_result.layout_description,
                        'text_content': page_result.text_content,
                        'visual_elements': page_result.visual_elements,
                        'metadata': page_result.metadata,
                        'processing_time': page_result.processing_time,
                        'error_message': page_result.error_message
                    },
                    'processing_time': page_time
                })
            
            doc.close()
            
            total_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Qwen 2.5 VL processing completed in {total_time:.1f}s")
            logger.info(f"Success rate: {successful_pages}/{pages_to_process} pages ({successful_pages/pages_to_process:.1%})")
            
            # Aggregate results
            document_summary = self._aggregate_page_results(pages_results)
            
            return {
                'processing_method': 'local_qwen25vl',
                'pages': pages_results,
                'document_summary': document_summary,
                'performance_metrics': {
                    'total_processing_time': total_time,
                    'successful_pages': successful_pages,
                    'total_pages_attempted': pages_to_process,
                    'success_rate': successful_pages / pages_to_process,
                    'average_page_time': total_time / pages_to_process
                }
            }
            
        except Exception as e:
            logger.error(f"Qwen 2.5 VL document processing failed: {str(e)}")
            raise
    
    async def _process_page_with_qwen(self, page_image: Image.Image, page_num: int) -> VLMResult:
        """Process single page with Qwen 2.5 VL."""
        
        start_time = datetime.now()
        
        try:
            # Call Qwen API
            qwen_response = await self._call_qwen_api(page_image)
            
            # Parse response
            result = self._parse_qwen_response(qwen_response, page_num)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Page {page_num}: Qwen processing error: {str(e)}")
            
            return VLMResult(
                content_type="error",
                confidence=0.0,
                layout_description="",
                text_content="",
                visual_elements=[],
                metadata={"error": str(e), "page_number": page_num},
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    async def _call_qwen_api(self, image: Image.Image) -> str:
        """Call Qwen 2.5 VL API through Ollama."""
        
        # Convert image to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Create document analysis prompt optimized for Qwen
        prompt = self._create_qwen_prompt()
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [img_base64],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_ctx": 4096,     # Qwen can handle larger context
                "num_predict": 1024   # Allow longer responses for detailed analysis
            }
        }
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.max_retries + 1):
                try:
                    if attempt > 0:
                        backoff_time = min(30, 2 ** attempt)
                        logger.info(f"Retrying Qwen API call after {backoff_time}s backoff (attempt {attempt + 1})")
                        await asyncio.sleep(backoff_time)
                    
                    async with session.post(
                        f"{self.api_endpoint}/api/generate",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            response_text = result.get("response", "")
                            
                            if response_text.strip():
                                logger.debug(f"Qwen API success on attempt {attempt + 1}")
                                return response_text
                            else:
                                raise Exception("Empty response from Qwen API")
                        else:
                            error_text = await response.text()
                            raise Exception(f"Qwen API error {response.status}: {error_text}")
                            
                except asyncio.TimeoutError:
                    if attempt < self.max_retries:
                        logger.warning(f"Qwen API timeout on attempt {attempt + 1}/{self.max_retries + 1}")
                    else:
                        raise asyncio.TimeoutError("Qwen API timeout after all retries")
                        
                except Exception as e:
                    if attempt < self.max_retries:
                        logger.warning(f"Qwen API error on attempt {attempt + 1}/{self.max_retries + 1}: {str(e)}")
                    else:
                        raise
        
        raise Exception("All Qwen API retry attempts failed")
    
    def _create_qwen_prompt(self) -> str:
        """Create analysis prompt optimized for Qwen 2.5 VL's capabilities."""
        return """Analyze this document page comprehensively. Qwen 2.5 VL excels at understanding complex document layouts and extracting structured information.

Please provide detailed analysis in JSON format:

{
  "content_type": "legal_document|corporate_memo|research_report|regulatory_filing|financial_document|technical_manual|correspondence|other",
  "confidence": 0.95,
  "layout_analysis": {
    "structure": "single_column|multi_column|mixed",
    "has_header": true,
    "has_footer": true,
    "section_count": 3,
    "reading_order": "top_to_bottom|left_to_right|complex"
  },
  "text_content": "Complete extracted text preserving structure and formatting...",
  "visual_elements": [
    {
      "type": "table|chart|diagram|image|signature|logo",
      "description": "detailed description",
      "location": "top|middle|bottom|left|right|center",
      "data_extracted": "any structured data if applicable"
    }
  ],
  "document_metadata": {
    "appears_to_be_page": "1 of 5",
    "date_indicators": ["2020-01-01"],
    "document_title": "extracted or inferred title",
    "document_type_confidence": "high|medium|low"
  },
  "key_information": {
    "entities_mentioned": ["companies", "people", "places"],
    "financial_figures": ["$1.2M", "15%"],
    "dates_and_periods": ["Q3 2019", "January 2020"],
    "technical_terms": ["industry-specific terminology"],
    "regulatory_references": ["FDA", "DEA", "regulatory codes"]
  },
  "quality_assessment": {
    "text_clarity": "excellent|good|fair|poor",
    "image_resolution": "high|medium|low",
    "document_completeness": "complete|partial|fragmented",
    "processing_confidence": "high|medium|low"
  }
}

Focus on accuracy and thoroughness. Leverage Qwen's strong capabilities for document understanding. Return only valid JSON."""

    def _parse_qwen_response(self, response: str, page_num: int) -> VLMResult:
        """Parse Qwen 2.5 VL response into VLMResult structure."""
        
        try:
            # Clean response and extract JSON
            response = response.strip()
            
            # Find JSON boundaries
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response[json_start:json_end]
                data = json.loads(json_text)
                
                # Extract structured information
                content_type = data.get('content_type', 'mixed')
                confidence = float(data.get('confidence', 0.7))
                
                # Combine layout analysis into description
                layout_analysis = data.get('layout_analysis', {})
                layout_description = f"Structure: {layout_analysis.get('structure', 'unknown')}, " \
                                   f"Sections: {layout_analysis.get('section_count', 'unknown')}, " \
                                   f"Reading order: {layout_analysis.get('reading_order', 'unknown')}"
                
                text_content = data.get('text_content', '')
                visual_elements = data.get('visual_elements', [])
                
                # Enhanced metadata combining multiple analysis aspects
                metadata = {
                    'page_number': page_num,
                    'qwen_analysis': True,
                    'layout_analysis': layout_analysis,
                    'document_metadata': data.get('document_metadata', {}),
                    'key_information': data.get('key_information', {}),
                    'quality_assessment': data.get('quality_assessment', {}),
                    'response_length': len(response)
                }
                
                return VLMResult(
                    content_type=content_type,
                    confidence=confidence,
                    layout_description=layout_description,
                    text_content=text_content,
                    visual_elements=visual_elements,
                    metadata=metadata,
                    processing_time=0.0,  # Will be set by caller
                    success=True
                )
            else:
                # No JSON structure found, use raw response
                return VLMResult(
                    content_type="mixed",
                    confidence=0.5,
                    layout_description="No structured analysis available",
                    text_content=response[:1000],  # Use first part of response
                    visual_elements=[],
                    metadata={
                        'page_number': page_num,
                        'json_extraction_failed': True,
                        'raw_response_preview': response[:200]
                    },
                    processing_time=0.0,
                    success=True,  # Still consider it successful
                    error_message="Response not in expected JSON format"
                )
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed for page {page_num}: {str(e)}")
            return VLMResult(
                content_type="mixed",
                confidence=0.3,
                layout_description="JSON parse error",
                text_content=response[:500] if response else "",
                visual_elements=[],
                metadata={
                    'page_number': page_num,
                    'json_error': str(e),
                    'raw_response_preview': response[:100] if response else ""
                },
                processing_time=0.0,
                success=False,
                error_message=f"JSON parsing failed: {str(e)}"
            )
    
    def _page_to_optimized_image(self, page: fitz.Page) -> Image.Image:
        """Convert PDF page to image optimized for Qwen 2.5 VL."""
        
        # Qwen 2.5 VL can handle high-resolution images effectively
        matrix = fitz.Matrix(2.0, 2.0)  # Good balance of quality and processing speed
        pix = page.get_pixmap(matrix=matrix)
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        
        # Optimize for Qwen's input requirements
        if image.size[0] > self.image_size_limit[0] or image.size[1] > self.image_size_limit[1]:
            image.thumbnail(self.image_size_limit, Image.Resampling.LANCZOS)
            logger.debug(f"Image resized to {image.size} for Qwen processing")
        
        return image
    
    def _aggregate_page_results(self, pages_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate Qwen page results into document summary."""
        
        successful_pages = [p for p in pages_results if p["vlm_result"]["success"]]
        
        if not successful_pages:
            return {
                "error": "No successful Qwen processing",
                "total_pages": len(pages_results),
                "fallback_used": True
            }
        
        # Aggregate content types
        content_types = [p["vlm_result"]["content_type"] for p in successful_pages]
        primary_type = max(set(content_types), key=content_types.count) if content_types else "mixed"
        
        # Combine all text content with page markers
        text_parts = []
        for page_data in successful_pages:
            page_num = page_data["page_number"]
            text = page_data["vlm_result"]["text_content"]
            if text:
                text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
        
        combined_text = "\n\n".join(text_parts)
        
        # Aggregate visual elements across all pages
        all_visual_elements = []
        for page_data in successful_pages:
            elements = page_data["vlm_result"]["visual_elements"]
            for element in elements:
                element_with_page = element.copy()
                element_with_page["page"] = page_data["page_number"] + 1
                all_visual_elements.append(element_with_page)
        
        # Aggregate key information from all pages
        all_entities = {}
        all_dates = set()
        all_financial_figures = set()
        
        for page_data in successful_pages:
            metadata = page_data["vlm_result"]["metadata"]
            key_info = metadata.get("key_information", {})
            
            # Collect entities
            for entity_type, entities in key_info.items():
                if isinstance(entities, list):
                    if entity_type not in all_entities:
                        all_entities[entity_type] = set()
                    all_entities[entity_type].update(entities)
            
            # Collect dates and financial figures
            all_dates.update(key_info.get("dates_and_periods", []))
            all_financial_figures.update(key_info.get("financial_figures", []))
        
        # Convert sets to lists for JSON serialization
        all_entities = {k: list(v) for k, v in all_entities.items()}
        
        # Calculate average confidence
        confidences = [p["vlm_result"]["confidence"] for p in successful_pages if p["vlm_result"]["confidence"] > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            "primary_content_type": primary_type,
            "average_confidence": avg_confidence,
            "combined_text_content": combined_text,
            "all_visual_elements": all_visual_elements,
            "aggregated_entities": all_entities,
            "document_dates": list(all_dates),
            "financial_figures": list(all_financial_figures),
            "successful_pages": len(successful_pages),
            "total_pages_attempted": len(pages_results),
            "success_rate": len(successful_pages) / len(pages_results),
            "processing_summary": {
                "model_used": self.model,
                "qwen_enhanced_analysis": True,
                "comprehensive_extraction": True
            }
        }