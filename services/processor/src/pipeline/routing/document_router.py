# services/processor/src/pipeline/document_router.py

import os
import fitz
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import statistics

# Import our existing VLM processors
from pipeline.vlm.cpu_optimized_processor import CPUOptimizedVLMProcessor

from pipeline.enhanced_models.qwen_processor import Qwen25VLProcessor

logger = logging.getLogger(__name__)

@dataclass
class DocumentCharacteristics:
    """Analysis of document characteristics for routing decisions."""
    text_density: float           # Text per square unit of page area
    visual_element_count: int     # Number of images, charts, diagrams
    page_count: int              # Total pages in document
    average_font_size: float     # Average font size across document
    layout_complexity: float    # Measure of layout complexity (0.0 to 1.0)
    estimated_difficulty: str    # 'simple', 'moderate', 'complex', 'very_complex'
    recommended_processor: str   # 'local_qwen', 'cloud_gpt4v', 'cloud_claude', 'fallback'
    confidence: float           # Confidence in the routing recommendation

@dataclass
class ProcessingStrategy:
    """Complete processing strategy for a document."""
    primary_method: str         # Primary processing method to use
    fallback_methods: List[str] # Ordered list of fallback methods
    timeout_seconds: int        # Recommended timeout for primary method
    quality_threshold: float    # Minimum quality score to accept results
    cost_estimate: float       # Estimated processing cost (for cloud APIs)

class DocumentRouter:
    """Intelligent document routing system for optimal processing strategy selection."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the document router with configuration and processing capabilities."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize our enhanced processors
        self.processors = {}
        
        # Local VLM processors - we'll add Qwen here
        if config.get('vlm', {}).get('local_models', {}).get('qwen25vl', {}).get('enabled', False):
            self.processors['local_qwen'] = Qwen25VLProcessor(config)
        
        # Keep existing LLaVA as backup local processor
        if config.get('vlm', {}).get('local_models', {}).get('llava', {}).get('enabled', True):
            self.processors['local_llava'] = CPUOptimizedVLMProcessor(config)
        
        # Cloud processors - we'll implement these next
        if config.get('vlm', {}).get('cloud_models', {}).get('gpt4v', {}).get('enabled', False):
            self.processors['cloud_gpt4v'] = GPT4VisionProcessor(config)
            
        if config.get('vlm', {}).get('cloud_models', {}).get('claude', {}).get('enabled', False):
            self.processors['cloud_claude'] = ClaudeVisionProcessor(config)
        
        # Enhanced OCR fallback
        self.processors['fallback_ocr'] = EnhancedOCRProcessor(config)
        
        # Performance tracking for adaptive routing
        self.performance_history = {
            processor_name: {'success_rate': 0.0, 'avg_time': 0.0, 'total_processed': 0}
            for processor_name in self.processors.keys()
        }
        
        self.logger.info(f"Document router initialized with {len(self.processors)} processors")
    
    def analyze_document_characteristics(self, pdf_path: str) -> DocumentCharacteristics:
        """Perform rapid analysis of document to determine processing strategy."""
        
        try:
            doc = fitz.open(pdf_path)
            
            # Basic document metrics
            page_count = len(doc)
            total_text_length = 0
            total_page_area = 0
            visual_elements = 0
            font_sizes = []
            
            # Analyze first few pages for characteristics (sampling for speed)
            sample_pages = min(3, page_count)
            
            for page_num in range(sample_pages):
                page = doc[page_num]
                
                # Text analysis
                page_text = page.get_text()
                total_text_length += len(page_text)
                
                # Page area calculation
                rect = page.rect
                page_area = rect.width * rect.height
                total_page_area += page_area
                
                # Visual elements count
                images = page.get_images()
                drawings = page.get_drawings()
                visual_elements += len(images) + len(drawings)
                
                # Font size analysis - this gives us insight into document complexity
                blocks = page.get_text("dict")
                for block in blocks.get("blocks", []):
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line.get("spans", []):
                                if span.get("size"):
                                    font_sizes.append(span["size"])
            
            doc.close()
            
            # Calculate characteristics
            text_density = total_text_length / total_page_area if total_page_area > 0 else 0
            avg_font_size = statistics.mean(font_sizes) if font_sizes else 12
            visual_density = visual_elements / sample_pages
            
            # Determine layout complexity based on multiple factors
            layout_complexity = self._calculate_layout_complexity(
                text_density, visual_density, avg_font_size, page_count
            )
            
            # Estimate processing difficulty
            difficulty = self._estimate_difficulty(
                text_density, visual_density, avg_font_size, page_count, layout_complexity
            )
            
            # Recommend processor based on characteristics
            recommended_processor, confidence = self._recommend_processor(
                text_density, visual_density, page_count, difficulty
            )
            
            return DocumentCharacteristics(
                text_density=text_density,
                visual_element_count=visual_elements,
                page_count=page_count,
                average_font_size=avg_font_size,
                layout_complexity=layout_complexity,
                estimated_difficulty=difficulty,
                recommended_processor=recommended_processor,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing document {pdf_path}: {str(e)}")
            # Return safe defaults that route to fallback processing
            return DocumentCharacteristics(
                text_density=0.0,
                visual_element_count=0,
                page_count=1,
                average_font_size=12.0,
                layout_complexity=1.0,
                estimated_difficulty='very_complex',
                recommended_processor='fallback_ocr',
                confidence=0.0
            )
    
    def _calculate_layout_complexity(self, text_density: float, visual_density: float, 
                                   avg_font_size: float, page_count: int) -> float:
        """Calculate layout complexity score from 0.0 (simple) to 1.0 (very complex)."""
        
        complexity_factors = []
        
        # High text density indicates complex layouts
        if text_density > 0.015:  # Very dense text
            complexity_factors.append(0.8)
        elif text_density > 0.010:  # Moderately dense
            complexity_factors.append(0.5)
        else:  # Sparse text
            complexity_factors.append(0.2)
        
        # Small fonts suggest complex documents with lots of information
        if avg_font_size < 9:
            complexity_factors.append(0.9)
        elif avg_font_size < 11:
            complexity_factors.append(0.6)
        else:
            complexity_factors.append(0.3)
        
        # Visual elements add complexity
        if visual_density > 2:  # Many visual elements per page
            complexity_factors.append(0.7)
        elif visual_density > 0.5:
            complexity_factors.append(0.4)
        else:
            complexity_factors.append(0.1)
        
        # Long documents tend to be more complex
        if page_count > 20:
            complexity_factors.append(0.6)
        elif page_count > 5:
            complexity_factors.append(0.3)
        else:
            complexity_factors.append(0.1)
        
        return min(1.0, statistics.mean(complexity_factors))
    
    def _estimate_difficulty(self, text_density: float, visual_density: float, 
                           avg_font_size: float, page_count: int, 
                           layout_complexity: float) -> str:
        """Estimate overall document processing difficulty."""
        
        if layout_complexity > 0.7:
            return 'very_complex'
        elif layout_complexity > 0.5:
            return 'complex'
        elif layout_complexity > 0.3:
            return 'moderate'
        else:
            return 'simple'
    
    def _recommend_processor(self, text_density: float, visual_density: float, 
                           page_count: int, difficulty: str) -> Tuple[str, float]:
        """Recommend the best processor based on document characteristics."""
        
        # Check processor availability first
        available_processors = list(self.processors.keys())
        
        # High visual content benefits from advanced visual reasoning
        if visual_density > 1.0 and 'cloud_gpt4v' in available_processors:
            return 'cloud_gpt4v', 0.9
        
        # Dense text documents benefit from Claude's text analysis capabilities
        if text_density > 0.012 and 'cloud_claude' in available_processors:
            return 'cloud_claude', 0.85
        
        # Complex documents benefit from more powerful local models
        if difficulty in ['complex', 'very_complex'] and 'local_qwen' in available_processors:
            return 'local_qwen', 0.8
        
        # Simple to moderate documents can use efficient local processing
        if difficulty in ['simple', 'moderate'] and 'local_llava' in available_processors:
            return 'local_llava', 0.75
        
        # Default to best available processor
        if 'local_qwen' in available_processors:
            return 'local_qwen', 0.6
        elif 'local_llava' in available_processors:
            return 'local_llava', 0.5
        else:
            return 'fallback_ocr', 0.3
    
    def create_processing_strategy(self, characteristics: DocumentCharacteristics) -> ProcessingStrategy:
        """Create comprehensive processing strategy based on document characteristics."""
        
        primary_method = characteristics.recommended_processor
        
        # Create intelligent fallback sequence
        fallback_sequence = []
        
        if primary_method == 'cloud_gpt4v':
            fallback_sequence = ['cloud_claude', 'local_qwen', 'local_llava', 'fallback_ocr']
        elif primary_method == 'cloud_claude':
            fallback_sequence = ['cloud_gpt4v', 'local_qwen', 'local_llava', 'fallback_ocr']
        elif primary_method == 'local_qwen':
            fallback_sequence = ['cloud_claude', 'local_llava', 'fallback_ocr']
        elif primary_method == 'local_llava':
            fallback_sequence = ['local_qwen', 'fallback_ocr']
        else:  # fallback_ocr
            fallback_sequence = []
        
        # Filter to only include available processors
        available_fallbacks = [proc for proc in fallback_sequence if proc in self.processors]
        
        # Set timeout based on complexity
        if characteristics.estimated_difficulty == 'very_complex':
            timeout = 300  # 5 minutes for very complex documents
        elif characteristics.estimated_difficulty == 'complex':
            timeout = 180  # 3 minutes for complex documents
        elif characteristics.estimated_difficulty == 'moderate':
            timeout = 120  # 2 minutes for moderate documents
        else:
            timeout = 60   # 1 minute for simple documents
        
        # Estimate cost (simplified - you'd implement actual cost calculation)
        cost_estimate = 0.0
        if 'cloud' in primary_method:
            cost_estimate = 0.01 * characteristics.page_count  # Rough estimate
        
        return ProcessingStrategy(
            primary_method=primary_method,
            fallback_methods=available_fallbacks,
            timeout_seconds=timeout,
            quality_threshold=0.6,  # Minimum acceptable quality score
            cost_estimate=cost_estimate
        )
    
    async def route_and_process(self, pdf_path: str) -> Dict[str, Any]:
        """Main routing and processing method that orchestrates the entire workflow."""
        
        start_time = datetime.now()
        
        # Step 1: Analyze document characteristics
        self.logger.info(f"Analyzing document characteristics: {os.path.basename(pdf_path)}")
        characteristics = self.analyze_document_characteristics(pdf_path)
        
        self.logger.info(f"Document analysis complete:")
        self.logger.info(f"  Difficulty: {characteristics.estimated_difficulty}")
        self.logger.info(f"  Recommended processor: {characteristics.recommended_processor}")
        self.logger.info(f"  Confidence: {characteristics.confidence:.2f}")
        
        # Step 2: Create processing strategy
        strategy = self.create_processing_strategy(characteristics)
        
        # Step 3: Attempt processing with primary method
        result = await self._attempt_processing(pdf_path, strategy.primary_method, strategy.timeout_seconds)
        
        # Step 4: Try fallback methods if primary fails
        if not result.get('success', False) and strategy.fallback_methods:
            self.logger.warning(f"Primary method {strategy.primary_method} failed, trying fallbacks")
            
            for fallback_method in strategy.fallback_methods:
                self.logger.info(f"Attempting fallback: {fallback_method}")
                result = await self._attempt_processing(pdf_path, fallback_method, strategy.timeout_seconds)
                
                if result.get('success', False):
                    self.logger.info(f"Fallback method {fallback_method} succeeded")
                    break
        
        # Step 5: Update performance tracking
        processing_time = (datetime.now() - start_time).total_seconds()
        self._update_performance_tracking(result.get('method_used', 'unknown'), 
                                        result.get('success', False), processing_time)
        
        # Step 6: Enhance result with routing information
        result['routing_info'] = {
            'characteristics': characteristics,
            'strategy': strategy,
            'total_processing_time': processing_time
        }
        
        return result
    
    async def _attempt_processing(self, pdf_path: str, method_name: str, timeout: int) -> Dict[str, Any]:
        """Attempt processing with specified method."""
        
        if method_name not in self.processors:
            return {
                'success': False,
                'error': f"Processor {method_name} not available",
                'method_used': method_name
            }
        
        try:
            processor = self.processors[method_name]
            
            # Use asyncio timeout to enforce processing limits
            result = await asyncio.wait_for(
                processor.process_document_with_progress(pdf_path),
                timeout=timeout
            )
            
            # Evaluate result quality
            quality_score = self._evaluate_result_quality(result)
            
            return {
                'success': True,
                'result': result,
                'quality_score': quality_score,
                'method_used': method_name,
                'processing_time': result.get('performance_metrics', {}).get('total_processing_time', 0)
            }
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Processing timeout after {timeout}s with {method_name}")
            return {
                'success': False,
                'error': f"Timeout after {timeout} seconds",
                'method_used': method_name
            }
        except Exception as e:
            self.logger.error(f"Processing failed with {method_name}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'method_used': method_name
            }
    
    def _evaluate_result_quality(self, result: Dict[str, Any]) -> float:
        """Evaluate the quality of processing results."""
        
        quality_factors = []
        
        # Check if we got meaningful content
        if result.get('document_summary', {}).get('combined_text_content'):
            text_length = len(result['document_summary']['combined_text_content'])
            if text_length > 100:
                quality_factors.append(0.8)
            elif text_length > 50:
                quality_factors.append(0.5)
            else:
                quality_factors.append(0.2)
        
        # Check confidence scores
        avg_confidence = result.get('document_summary', {}).get('average_confidence', 0.0)
        quality_factors.append(avg_confidence)
        
        # Check success rate
        success_rate = result.get('performance_metrics', {}).get('success_rate', 0.0)
        quality_factors.append(success_rate)
        
        return statistics.mean(quality_factors) if quality_factors else 0.0
    
    def _update_performance_tracking(self, method_name: str, success: bool, processing_time: float):
        """Update performance history for adaptive routing improvements."""
        
        if method_name in self.performance_history:
            history = self.performance_history[method_name]
            
            # Update success rate with exponential moving average
            alpha = 0.1  # Learning rate
            history['success_rate'] = (alpha * (1.0 if success else 0.0) + 
                                     (1 - alpha) * history['success_rate'])
            
            # Update average processing time
            history['avg_time'] = (alpha * processing_time + 
                                 (1 - alpha) * history['avg_time'])
            
            history['total_processed'] += 1