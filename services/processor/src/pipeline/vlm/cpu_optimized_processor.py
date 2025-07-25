# services/processor/src/pipeline/vlm/cpu_optimized_processor.py

import asyncio
import aiohttp
import base64
import io
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
from PIL import Image
import fitz  # PyMuPDF
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics

logger = logging.getLogger(__name__)

@dataclass
class VLMPerformanceMetrics:
    """Track performance metrics to optimize timeouts dynamically."""
    api_response_times: List[float]
    processing_times: List[float] 
    success_rate: float
    average_response_time: float
    p95_response_time: float  # 95th percentile - useful for timeout setting
    
    def update_metrics(self, new_time: float, success: bool):
        """Update performance metrics with new data point."""
        if success:
            self.api_response_times.append(new_time)
            self.processing_times.append(new_time)
        
        # Recalculate metrics
        if self.api_response_times:
            self.average_response_time = statistics.mean(self.api_response_times)
            
            # Calculate 95th percentile with fallback for older Python versions
            try:
                self.p95_response_time = statistics.quantile(self.api_response_times, 0.95) if len(self.api_response_times) >= 2 else self.average_response_time
            except AttributeError:
                # Fallback for Python < 3.8 or other compatibility issues
                if len(self.api_response_times) >= 2:
                    sorted_times = sorted(self.api_response_times)
                    p95_index = int(0.95 * (len(sorted_times) - 1))
                    self.p95_response_time = sorted_times[p95_index]
                else:
                    self.p95_response_time = self.average_response_time
        
        # Update success rate
        total_attempts = len(self.processing_times) + (1 if not success else 0)
        self.success_rate = len(self.api_response_times) / total_attempts if total_attempts > 0 else 0.0

class CPUOptimizedVLMProcessor:
    """VLM processor optimized for CPU processing with adaptive timeout management."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("vlm", {})
        self.enabled = self.config.get("enabled", False)
        
        if not self.enabled:
            logger.info("VLM processing disabled - will use traditional processing only")
            return
            
        # Initialize performance tracking
        self.performance_metrics = VLMPerformanceMetrics(
            api_response_times=[],
            processing_times=[],
            success_rate=0.0,
            average_response_time=0.0,
            p95_response_time=0.0
        )
        
        # CPU-optimized timeout configuration
        # These start conservative and adapt based on actual performance
        self.base_timeout = self.config.get("base_timeout", 180)  # 3 minutes initial
        self.max_timeout = self.config.get("max_timeout", 600)   # 10 minutes absolute max
        self.min_timeout = self.config.get("min_timeout", 60)    # 1 minute minimum
        
        # Dynamic timeout calculation
        self.adaptive_timeouts = self.config.get("adaptive_timeouts", True)
        self.timeout_multiplier = self.config.get("timeout_multiplier", 1.5)  # Timeout = avg_time * multiplier
        
        # API configuration
        self.api_endpoint = self.config.get("llava", {}).get("api_endpoint", "http://localhost:11434")
        self.model = self.config.get("llava", {}).get("model", "llava:7b")
        self.max_retries = self.config.get("max_retries", 2)  # Reduced for CPU
        
        # CPU-specific optimizations
        self.enable_progress_logging = self.config.get("enable_progress_logging", True)
        self.max_pages_per_document = self.config.get("max_pages_per_document", 5)  # Limit for development
        self.image_size_limit = self.config.get("image_size_limit", (1024, 1024))  # Reduce image size for faster processing
        
        # Simplified prompts for faster processing
        self.use_simplified_prompts = self.config.get("use_simplified_prompts", True)
        
        logger.info(f"CPU-Optimized VLM Processor initialized:")
        logger.info(f"  Model: {self.model}")
        logger.info(f"  Base timeout: {self.base_timeout}s")
        logger.info(f"  Adaptive timeouts: {self.adaptive_timeouts}")
        logger.info(f"  Max pages per document: {self.max_pages_per_document}")
    
    def get_current_timeout(self) -> float:
        """Calculate current timeout based on performance history."""
        if not self.adaptive_timeouts or not self.performance_metrics.api_response_times:
            return self.base_timeout
        
        # Use 95th percentile response time plus buffer
        adaptive_timeout = self.performance_metrics.p95_response_time * self.timeout_multiplier
        
        # Clamp to min/max bounds
        adaptive_timeout = max(self.min_timeout, min(self.max_timeout, adaptive_timeout))
        
        logger.debug(f"Adaptive timeout: {adaptive_timeout:.1f}s (based on P95: {self.performance_metrics.p95_response_time:.1f}s)")
        return adaptive_timeout
    
    def get_simplified_prompt(self) -> str:
        """Return a simplified prompt optimized for faster CPU processing."""
        if self.use_simplified_prompts:
            return """Analyze this document page quickly and return JSON:

{
  "type": "email|report|presentation|chart|mixed",
  "confidence": 0.8,
  "content": "brief summary of main content"
}

Be concise for faster processing."""
        else:
            # Use full prompt from config if available
            return self.config.get("prompts", {}).get("universal", self._default_prompt())
    
    def _default_prompt(self) -> str:
        """Default prompt for document analysis."""
        return """Analyze this document page and provide structured information.

Return JSON format:
{
  "content_type": "email|presentation|report|chart|table|mixed",
  "confidence": 0.8,
  "layout": "layout description",
  "text_content": "key text content",
  "visual_elements": []
}"""
    
    async def process_document_with_progress(self, pdf_path: str) -> Dict[str, Any]:
        """Process document with detailed progress logging for CPU optimization."""
        if not self.enabled:
            logger.warning("VLM processing disabled")
            return {"processing_method": "disabled", "reason": "vlm_disabled"}
        
        start_time = datetime.now()
        logger.info(f"Starting VLM processing of: {pdf_path}")
        
        try:
            # Open PDF and check page count
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            pages_to_process = min(total_pages, self.max_pages_per_document)
            
            logger.info(f"Document has {total_pages} pages, processing first {pages_to_process} pages")
            
            pages_results = []
            successful_pages = 0
            
            for page_num in range(pages_to_process):
                page_start_time = time.time()
                logger.info(f"Processing page {page_num + 1}/{pages_to_process}...")
                
                page = doc[page_num]
                
                # Convert page to optimized image
                page_image = self._page_to_optimized_image(page)
                logger.debug(f"Page {page_num + 1} converted to image: {page_image.size}")
                
                # Get traditional text as fallback
                traditional_text = page.get_text()
                
                # Process with VLM using current timeout
                current_timeout = self.get_current_timeout()
                logger.info(f"Using timeout: {current_timeout:.1f}s (based on {len(self.performance_metrics.api_response_times)} previous measurements)")
                
                vlm_result = await self._process_page_with_timeout(
                    page_image, 
                    traditional_text, 
                    page_num, 
                    current_timeout
                )
                
                page_time = time.time() - page_start_time
                
                if vlm_result.success:
                    successful_pages += 1
                    logger.info(f"✅ Page {page_num + 1} processed successfully in {page_time:.1f}s")
                    logger.info(f"   Content type: {vlm_result.content_type}, Confidence: {vlm_result.confidence:.2f}")
                else:
                    logger.warning(f"⚠️ Page {page_num + 1} failed: {vlm_result.error_message}")
                
                pages_results.append({
                    "page_number": page_num,
                    "vlm_result": asdict(vlm_result),
                    "processing_time": page_time,
                    "traditional_fallback": traditional_text if not vlm_result.success else None
                })
                
                # Update performance metrics
                self.performance_metrics.update_metrics(page_time, vlm_result.success)
                
                # Log performance summary after each page
                if self.performance_metrics.api_response_times:
                    avg_time = self.performance_metrics.average_response_time
                    success_rate = self.performance_metrics.success_rate
                    logger.info(f"   Performance: avg={avg_time:.1f}s, success_rate={success_rate:.1%}")
            
            doc.close()
            
            # Calculate overall results
            total_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"VLM processing completed in {total_time:.1f}s")
            logger.info(f"Success rate: {successful_pages}/{pages_to_process} pages ({successful_pages/pages_to_process:.1%})")
            
            # Aggregate results
            document_result = self._aggregate_page_results(pages_results)
            
            return {
                "processing_method": "cpu_optimized_vlm",
                "pages": pages_results,
                "document_summary": document_result,
                "performance_metrics": {
                    "total_processing_time": total_time,
                    "successful_pages": successful_pages,
                    "total_pages_attempted": pages_to_process,
                    "success_rate": successful_pages / pages_to_process,
                    "average_page_time": self.performance_metrics.average_response_time,
                    "current_timeout": self.get_current_timeout()
                },
                "optimization_settings": {
                    "adaptive_timeouts": self.adaptive_timeouts,
                    "image_size_limit": self.image_size_limit,
                    "simplified_prompts": self.use_simplified_prompts,
                    "max_pages_processed": pages_to_process
                }
            }
            
        except Exception as e:
            logger.error(f"VLM document processing failed: {str(e)}")
            return {
                "processing_method": "failed", 
                "error": str(e),
                "performance_metrics": asdict(self.performance_metrics)
            }
    
    async def _process_page_with_timeout(self, page_image: Image.Image, fallback_text: str, 
                                       page_num: int, timeout: float) -> 'VLMResult':
        """Process single page with specified timeout and detailed error handling."""
        start_time = time.time()
        
        try:
            # Log processing start
            if self.enable_progress_logging:
                logger.debug(f"Starting VLM API call for page {page_num} with {timeout:.1f}s timeout")
            
            # Call VLM API with current timeout
            vlm_response = await self._call_vlm_api_with_retry(page_image, timeout)
            
            # Parse response
            result = self._parse_vlm_response(vlm_response, page_num)
            
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            # Check quality of result
            if result.confidence < 0.3:
                logger.warning(f"Page {page_num}: Low confidence ({result.confidence:.2f}), but keeping result")
            
            return result
            
        except asyncio.TimeoutError:
            processing_time = time.time() - start_time
            logger.warning(f"Page {page_num}: VLM timeout after {processing_time:.1f}s")
            return self._create_fallback_result(fallback_text, page_num, processing_time, "timeout")
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Page {page_num}: VLM error after {processing_time:.1f}s: {str(e)}")
            return self._create_fallback_result(fallback_text, page_num, processing_time, str(e))
    
    async def _call_vlm_api_with_retry(self, image: Image.Image, timeout: float) -> str:
        """Call VLM API with retry logic optimized for CPU processing."""
        
        # Convert image to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Prepare payload with CPU-optimized settings
        payload = {
            "model": self.model,
            "prompt": self.get_simplified_prompt(),
            "images": [img_base64],
            "stream": False,
            "options": {
                "temperature": 0.1,    # Low temperature for consistent results
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_ctx": 2048,       # Reduced context for faster processing
                "num_predict": 512     # Limit response length for speed
            }
        }
        
        # Retry logic with exponential backoff
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    backoff_time = min(30, 2 ** attempt)  # Cap backoff at 30 seconds
                    logger.info(f"Retrying VLM API call after {backoff_time}s backoff (attempt {attempt + 1})")
                    await asyncio.sleep(backoff_time)
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.api_endpoint}/api/generate",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=timeout)
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            response_text = result.get("response", "")
                            
                            if response_text.strip():  # Ensure we got meaningful response
                                logger.debug(f"VLM API success on attempt {attempt + 1}")
                                return response_text
                            else:
                                raise Exception("Empty response from VLM API")
                        else:
                            error_text = await response.text()
                            raise Exception(f"API error {response.status}: {error_text}")
                            
            except asyncio.TimeoutError:
                if attempt < self.max_retries:
                    logger.warning(f"VLM API timeout on attempt {attempt + 1}/{self.max_retries + 1}")
                else:
                    raise asyncio.TimeoutError("VLM API timeout after all retries")
                    
            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(f"VLM API error on attempt {attempt + 1}/{self.max_retries + 1}: {str(e)}")
                else:
                    raise
        
        raise Exception("All retry attempts failed")
    
    def _page_to_optimized_image(self, page: fitz.Page) -> Image.Image:
        """Convert PDF page to optimized image for CPU processing."""
        # Use moderate resolution to balance quality vs processing speed
        matrix = fitz.Matrix(1.5, 1.5)  # 1.5x scaling - good balance for CPU
        pix = page.get_pixmap(matrix=matrix)
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        
        # Resize if too large (CPU optimization)
        if image.size[0] > self.image_size_limit[0] or image.size[1] > self.image_size_limit[1]:
            image.thumbnail(self.image_size_limit, Image.Resampling.LANCZOS)
            logger.debug(f"Image resized to {image.size} for CPU optimization")
        
        return image
    
    def _parse_vlm_response(self, response: str, page_num: int) -> 'VLMResult':
        """Parse VLM response with improved error handling."""
        try:
            # Clean response and extract JSON
            response = response.strip()
            
            # Find JSON boundaries
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response[json_start:json_end]
                data = json.loads(json_text)
                
                # Extract fields with defaults
                content_type = data.get("content_type", data.get("type", "mixed"))
                confidence = float(data.get("confidence", 0.6))
                layout = data.get("layout", data.get("description", ""))
                text_content = data.get("text_content", data.get("content", ""))
                visual_elements = data.get("visual_elements", [])
                
                return VLMResult(
                    content_type=content_type,
                    confidence=confidence,
                    layout_description=layout,
                    text_content=text_content,
                    visual_elements=visual_elements,
                    metadata={
                        "page_number": page_num,
                        "response_length": len(response),
                        "json_extracted": True
                    },
                    processing_time=0.0,  # Will be set by caller
                    success=True
                )
            else:
                # No JSON found, but we got a response
                return VLMResult(
                    content_type="mixed",
                    confidence=0.4,
                    layout_description="No structured response",
                    text_content=response[:500],  # Use first part of response
                    visual_elements=[],
                    metadata={
                        "page_number": page_num,
                        "json_extracted": False,
                        "raw_response_preview": response[:100]
                    },
                    processing_time=0.0,
                    success=True,  # Still successful, just unstructured
                    error_message="Response not in JSON format"
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
                    "page_number": page_num,
                    "json_error": str(e),
                    "raw_response_preview": response[:100] if response else ""
                },
                processing_time=0.0,
                success=False,
                error_message=f"JSON parsing failed: {str(e)}"
            )
    
    def _create_fallback_result(self, fallback_text: str, page_num: int, 
                              processing_time: float, error: str) -> 'VLMResult':
        """Create fallback result when VLM processing fails."""
        return VLMResult(
            content_type="traditional_fallback",
            confidence=0.5,
            layout_description="VLM processing failed, using traditional extraction",
            text_content=fallback_text,
            visual_elements=[],
            metadata={
                "page_number": page_num,
                "fallback_reason": error,
                "traditional_text_length": len(fallback_text)
            },
            processing_time=processing_time,
            success=True,  # Fallback is still a success
            error_message=f"VLM failed: {error}"
        )
    
    def _aggregate_page_results(self, pages_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate page results into document summary."""
        successful_pages = [p for p in pages_results if p["vlm_result"]["success"]]
        
        if not successful_pages:
            return {
                "error": "No successful VLM processing",
                "total_pages": len(pages_results),
                "fallback_used": True
            }
        
        # Aggregate content types
        content_types = [p["vlm_result"]["content_type"] for p in successful_pages]
        primary_type = max(set(content_types), key=content_types.count) if content_types else "mixed"
        
        # Calculate metrics
        confidences = [p["vlm_result"]["confidence"] for p in successful_pages if p["vlm_result"]["confidence"] > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            "primary_content_type": primary_type,
            "average_confidence": avg_confidence,
            "successful_pages": len(successful_pages),
            "total_pages_attempted": len(pages_results),
            "success_rate": len(successful_pages) / len(pages_results),
            "performance_summary": {
                "avg_response_time": self.performance_metrics.average_response_time,
                "p95_response_time": self.performance_metrics.p95_response_time,
                "overall_success_rate": self.performance_metrics.success_rate
            }
        }

# Import for the VLMResult class
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