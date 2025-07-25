# services/processor/src/pipeline/cloud_processors.py

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

logger = logging.getLogger(__name__)

@dataclass
class CloudProcessingResult:
    """Result from cloud-based document processing."""
    success: bool
    content_type: str
    confidence: float
    extracted_text: str
    structured_analysis: Dict[str, Any]
    visual_elements: List[Dict]
    processing_time: float
    cost_estimate: float
    model_used: str
    error_message: Optional[str] = None

class GPT4VisionProcessor:
    """OpenAI GPT-4 Vision processor for complex visual document analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('vlm', {}).get('cloud_models', {}).get('gpt4v', {})
        self.api_key = self.config.get('api_key')
        self.model = self.config.get('model', 'gpt-4-vision-preview')
        self.max_tokens = self.config.get('max_tokens', 2000)
        self.enabled = self.config.get('enabled', False) and bool(self.api_key)
        
        if not self.enabled:
            logger.warning("GPT-4 Vision processor disabled: no API key provided")
        else:
            logger.info(f"GPT-4 Vision processor initialized with model: {self.model}")
    
    async def process_document_with_progress(self, pdf_path: str) -> Dict[str, Any]:
        """Process document using GPT-4 Vision with progress tracking."""
        
        if not self.enabled:
            raise Exception("GPT-4 Vision processor not enabled")
        
        start_time = datetime.now()
        
        try:
            # Convert PDF pages to images
            doc = fitz.open(pdf_path)
            page_results = []
            total_cost = 0.0
            
            # Process first 5 pages to balance quality and cost
            max_pages = min(len(doc), 5)
            
            for page_num in range(max_pages):
                logger.info(f"Processing page {page_num + 1}/{max_pages} with GPT-4 Vision")
                
                page = doc[page_num]
                page_image = self._page_to_image(page)
                
                # Process page with GPT-4 Vision
                page_result = await self._process_page_with_gpt4v(page_image, page_num)
                page_results.append(page_result)
                
                total_cost += page_result.cost_estimate
            
            doc.close()
            
            # Aggregate results
            aggregated_result = self._aggregate_page_results(page_results)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'processing_method': 'cloud_gpt4v',
                'pages': page_results,
                'document_summary': aggregated_result,
                'performance_metrics': {
                    'total_processing_time': processing_time,
                    'pages_processed': max_pages,
                    'total_cost': total_cost,
                    'success_rate': len([p for p in page_results if p.success]) / len(page_results)
                }
            }
            
        except Exception as e:
            logger.error(f"GPT-4 Vision processing failed: {str(e)}")
            raise
    
    async def _process_page_with_gpt4v(self, image: Image.Image, page_num: int) -> CloudProcessingResult:
        """Process single page with GPT-4 Vision."""
        
        start_time = datetime.now()
        
        try:
            # Convert image to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Prepare the prompt for comprehensive document analysis
            prompt = self._create_analysis_prompt()
            
            # Make API request
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                }
                
                payload = {
                    'model': self.model,
                    'messages': [
                        {
                            'role': 'user',
                            'content': [
                                {'type': 'text', 'text': prompt},
                                {
                                    'type': 'image_url',
                                    'image_url': {
                                        'url': f'data:image/png;base64,{image_base64}',
                                        'detail': 'high'
                                    }
                                }
                            ]
                        }
                    ],
                    'max_tokens': self.max_tokens,
                    'temperature': 0.1
                }
                
                async with session.post(
                    'https://api.openai.com/v1/chat/completions',
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        content = result['choices'][0]['message']['content']
                        
                        # Parse the structured response
                        parsed_result = self._parse_gpt4v_response(content, page_num)
                        
                        # Estimate cost (simplified calculation)
                        cost = self._estimate_cost(result.get('usage', {}))
                        parsed_result.cost_estimate = cost
                        
                        processing_time = (datetime.now() - start_time).total_seconds()
                        parsed_result.processing_time = processing_time
                        
                        return parsed_result
                    else:
                        error_text = await response.text()
                        raise Exception(f"API error {response.status}: {error_text}")
        
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return CloudProcessingResult(
                success=False,
                content_type='error',
                confidence=0.0,
                extracted_text='',
                structured_analysis={},
                visual_elements=[],
                processing_time=processing_time,
                cost_estimate=0.0,
                model_used=self.model,
                error_message=str(e)
            )
    
    def _create_analysis_prompt(self) -> str:
        """Create comprehensive analysis prompt for GPT-4 Vision."""
        return """Analyze this document page comprehensively and provide a structured JSON response.

Focus on:
1. Document type identification (legal, corporate, research, regulatory, etc.)
2. Layout and structure analysis
3. Text content extraction with context preservation
4. Visual elements identification (tables, charts, diagrams, images)
5. Key information extraction relevant to pharmaceutical/opioid industry documents

Respond with valid JSON in this format:
{
  "content_type": "legal_document|corporate_memo|research_report|regulatory_filing|other",
  "confidence": 0.9,
  "document_structure": {
    "has_header": true,
    "has_footer": true,
    "column_count": 1,
    "section_count": 3
  },
  "extracted_text": "Complete text content with structure preserved...",
  "visual_elements": [
    {"type": "table", "description": "Financial data table", "location": "center"},
    {"type": "chart", "description": "Sales trend graph", "location": "bottom"}
  ],
  "key_information": {
    "companies_mentioned": ["Company A", "Company B"],
    "dates_mentioned": ["2020-01-01"],
    "financial_figures": ["$1.2M", "$500K"],
    "key_topics": ["opioid marketing", "regulatory compliance"]
  },
  "document_quality": {
    "text_clarity": "high|medium|low",
    "image_quality": "high|medium|low",
    "completeness": "complete|partial|fragmented"
  }
}

Only return valid JSON, no additional text."""

    def _parse_gpt4v_response(self, content: str, page_num: int) -> CloudProcessingResult:
        """Parse GPT-4 Vision response into structured result."""
        
        try:
            # Try to extract JSON from response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = content[json_start:json_end]
                data = json.loads(json_text)
                
                return CloudProcessingResult(
                    success=True,
                    content_type=data.get('content_type', 'unknown'),
                    confidence=float(data.get('confidence', 0.8)),
                    extracted_text=data.get('extracted_text', ''),
                    structured_analysis=data,
                    visual_elements=data.get('visual_elements', []),
                    processing_time=0.0,  # Will be set by caller
                    cost_estimate=0.0,    # Will be set by caller
                    model_used=self.model
                )
            else:
                # No valid JSON, but we got content
                return CloudProcessingResult(
                    success=True,
                    content_type='mixed',
                    confidence=0.6,
                    extracted_text=content,
                    structured_analysis={'raw_response': content},
                    visual_elements=[],
                    processing_time=0.0,
                    cost_estimate=0.0,
                    model_used=self.model
                )
                
        except json.JSONDecodeError:
            return CloudProcessingResult(
                success=False,
                content_type='error',
                confidence=0.0,
                extracted_text=content,
                structured_analysis={},
                visual_elements=[],
                processing_time=0.0,
                cost_estimate=0.0,
                model_used=self.model,
                error_message=f"Failed to parse JSON response"
            )
    
    def _page_to_image(self, page: fitz.Page) -> Image.Image:
        """Convert PDF page to optimized image for GPT-4 Vision."""
        
        # Use high resolution for GPT-4 Vision (it can handle detailed images well)
        matrix = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=matrix)
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        
        # Optimize image size for API limits (GPT-4 Vision handles up to 2048x2048 well)
        if image.size[0] > 2048 or image.size[1] > 2048:
            image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        
        return image
    
    def _estimate_cost(self, usage: Dict[str, Any]) -> float:
        """Estimate processing cost based on token usage."""
        
        # Simplified cost calculation (you'd use actual OpenAI pricing)
        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)
        
        # Approximate GPT-4 Vision pricing (as of 2024)
        input_cost_per_token = 0.01 / 1000  # $0.01 per 1K input tokens
        output_cost_per_token = 0.03 / 1000  # $0.03 per 1K output tokens
        
        return (input_tokens * input_cost_per_token + output_tokens * output_cost_per_token)
    
    def _aggregate_page_results(self, page_results: List[CloudProcessingResult]) -> Dict[str, Any]:
        """Aggregate results from multiple pages into document summary."""
        
        successful_pages = [r for r in page_results if r.success]
        
        if not successful_pages:
            return {'error': 'No pages processed successfully'}
        
        # Aggregate content types
        content_types = [r.content_type for r in successful_pages]
        primary_type = max(set(content_types), key=content_types.count)
        
        # Combine all extracted text
        combined_text = '\n\n'.join([r.extracted_text for r in successful_pages])
        
        # Aggregate visual elements
        all_visual_elements = []
        for r in successful_pages:
            all_visual_elements.extend(r.visual_elements)
        
        # Calculate average confidence
        avg_confidence = sum([r.confidence for r in successful_pages]) / len(successful_pages)
        
        return {
            'primary_content_type': primary_type,
            'average_confidence': avg_confidence,
            'combined_text_content': combined_text,
            'all_visual_elements': all_visual_elements,
            'successful_pages': len(successful_pages),
            'total_pages': len(page_results),
            'success_rate': len(successful_pages) / len(page_results)
        }

class ClaudeVisionProcessor:
    """Claude 3.5 Sonnet processor for advanced text-heavy document analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('vlm', {}).get('cloud_models', {}).get('claude', {})
        self.api_key = self.config.get('api_key')
        self.model = self.config.get('model', 'claude-3-5-sonnet-20241022')
        self.max_tokens = self.config.get('max_tokens', 2000)
        self.enabled = self.config.get('enabled', False) and bool(self.api_key)
        
        if not self.enabled:
            logger.warning("Claude Vision processor disabled: no API key provided")
        else:
            logger.info(f"Claude Vision processor initialized with model: {self.model}")
    
    async def process_document_with_progress(self, pdf_path: str) -> Dict[str, Any]:
        """Process document using Claude Vision with focus on text analysis."""
        
        if not self.enabled:
            raise Exception("Claude Vision processor not enabled")
        
        start_time = datetime.now()
        
        try:
            # Convert PDF pages to images
            doc = fitz.open(pdf_path)
            page_results = []
            total_cost = 0.0
            
            # Process more pages with Claude since it's better at text analysis
            max_pages = min(len(doc), 8)
            
            for page_num in range(max_pages):
                logger.info(f"Processing page {page_num + 1}/{max_pages} with Claude Vision")
                
                page = doc[page_num]
                page_image = self._page_to_image(page)
                
                # Process page with Claude
                page_result = await self._process_page_with_claude(page_image, page_num)
                page_results.append(page_result)
                
                total_cost += page_result.cost_estimate
            
            doc.close()
            
            # Aggregate results
            aggregated_result = self._aggregate_page_results(page_results)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'processing_method': 'cloud_claude',
                'pages': page_results,
                'document_summary': aggregated_result,
                'performance_metrics': {
                    'total_processing_time': processing_time,
                    'pages_processed': max_pages,
                    'total_cost': total_cost,
                    'success_rate': len([p for p in page_results if p.success]) / len(page_results)
                }
            }
            
        except Exception as e:
            logger.error(f"Claude Vision processing failed: {str(e)}")
            raise
    
    async def _process_page_with_claude(self, image: Image.Image, page_num: int) -> CloudProcessingResult:
        """Process single page with Claude Vision."""
        
        start_time = datetime.now()
        
        try:
            # Convert image to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Create prompt optimized for Claude's strengths in text analysis
            prompt = self._create_claude_prompt()
            
            # Make API request to Claude
            async with aiohttp.ClientSession() as session:
                headers = {
                    'x-api-key': self.api_key,
                    'Content-Type': 'application/json',
                    'anthropic-version': '2023-06-01'
                }
                
                payload = {
                    'model': self.model,
                    'max_tokens': self.max_tokens,
                    'temperature': 0.1,
                    'messages': [
                        {
                            'role': 'user',
                            'content': [
                                {'type': 'text', 'text': prompt},
                                {
                                    'type': 'image',
                                    'source': {
                                        'type': 'base64',
                                        'media_type': 'image/png',
                                        'data': image_base64
                                    }
                                }
                            ]
                        }
                    ]
                }
                
                async with session.post(
                    'https://api.anthropic.com/v1/messages',
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        content = result['content'][0]['text']
                        
                        # Parse the structured response
                        parsed_result = self._parse_claude_response(content, page_num)
                        
                        # Estimate cost
                        cost = self._estimate_claude_cost(result.get('usage', {}))
                        parsed_result.cost_estimate = cost
                        
                        processing_time = (datetime.now() - start_time).total_seconds()
                        parsed_result.processing_time = processing_time
                        
                        return parsed_result
                    else:
                        error_text = await response.text()
                        raise Exception(f"Claude API error {response.status}: {error_text}")
        
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return CloudProcessingResult(
                success=False,
                content_type='error',
                confidence=0.0,
                extracted_text='',
                structured_analysis={},
                visual_elements=[],
                processing_time=processing_time,
                cost_estimate=0.0,
                model_used=self.model,
                error_message=str(e)
            )
    
    def _create_claude_prompt(self) -> str:
        """Create analysis prompt optimized for Claude's text analysis strengths."""
        return """Analyze this document page with particular attention to text content, document structure, and regulatory/legal context.

This document is likely from the pharmaceutical/opioid industry context. Please provide a comprehensive analysis focusing on:

1. Document classification and purpose
2. Complete text extraction with structural preservation
3. Identification of key regulatory, legal, or business information
4. Analysis of any data presentations or technical content
5. Assessment of document authenticity and completeness

Respond with structured JSON:
{
  "content_type": "legal_document|regulatory_filing|corporate_communication|research_document|financial_report|other",
  "confidence": 0.95,
  "document_context": {
    "industry_relevance": "high|medium|low",
    "regulatory_implications": true,
    "time_period": "estimated date range",
    "document_purpose": "brief description"
  },
  "extracted_text": "Complete, accurately transcribed text with original structure...",
  "key_entities": {
    "organizations": ["company names", "regulatory bodies"],
    "people": ["names mentioned"],
    "locations": ["places mentioned"],
    "dates": ["2020-01-01", "Q3 2019"],
    "financial_data": ["$1.2M", "15% increase"],
    "products": ["drug names", "product codes"],
    "regulatory_references": ["FDA approval", "DEA classification"]
  },
  "structural_elements": {
    "headers": ["section titles"],
    "signatures": true,
    "letterhead": true,
    "attachments_referenced": ["Exhibit A"],
    "page_indicators": "Page 1 of 5"
  },
  "analysis_quality": {
    "text_legibility": "excellent|good|fair|poor",
    "document_completeness": "complete|partial|fragment",
    "confidence_level": "high|medium|low"
  }
}

Focus on accuracy and completeness of text extraction. Only return valid JSON."""

    def _parse_claude_response(self, content: str, page_num: int) -> CloudProcessingResult:
        """Parse Claude's response into structured result."""
        
        try:
            # Extract JSON from Claude's response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = content[json_start:json_end]
                data = json.loads(json_text)
                
                return CloudProcessingResult(
                    success=True,
                    content_type=data.get('content_type', 'unknown'),
                    confidence=float(data.get('confidence', 0.8)),
                    extracted_text=data.get('extracted_text', ''),
                    structured_analysis=data,
                    visual_elements=data.get('structural_elements', {}).get('visual_elements', []),
                    processing_time=0.0,
                    cost_estimate=0.0,
                    model_used=self.model
                )
            else:
                # Use full response as extracted text
                return CloudProcessingResult(
                    success=True,
                    content_type='mixed',
                    confidence=0.7,
                    extracted_text=content,
                    structured_analysis={'raw_response': content},
                    visual_elements=[],
                    processing_time=0.0,
                    cost_estimate=0.0,
                    model_used=self.model
                )
                
        except json.JSONDecodeError:
            return CloudProcessingResult(
                success=False,
                content_type='error',
                confidence=0.0,
                extracted_text=content,
                structured_analysis={},
                visual_elements=[],
                processing_time=0.0,
                cost_estimate=0.0,
                model_used=self.model,
                error_message="Failed to parse JSON response"
            )
    
    def _page_to_image(self, page: fitz.Page) -> Image.Image:
        """Convert PDF page to image optimized for Claude Vision."""
        
        # Claude works well with high-quality images
        matrix = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=matrix)
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        
        # Claude can handle large images efficiently
        if image.size[0] > 1600 or image.size[1] > 2048:
            image.thumbnail((1600, 2048), Image.Resampling.LANCZOS)
        
        return image
    
    def _estimate_claude_cost(self, usage: Dict[str, Any]) -> float:
        """Estimate Claude processing cost."""
        
        # Simplified cost calculation for Claude
        input_tokens = usage.get('input_tokens', 0)
        output_tokens = usage.get('output_tokens', 0)
        
        # Approximate Claude pricing
        input_cost_per_token = 0.003 / 1000
        output_cost_per_token = 0.015 / 1000
        
        return (input_tokens * input_cost_per_token + output_tokens * output_cost_per_token)
    
    def _aggregate_page_results(self, page_results: List[CloudProcessingResult]) -> Dict[str, Any]:
        """Aggregate Claude results focusing on comprehensive text analysis."""
        
        successful_pages = [r for r in page_results if r.success]
        
        if not successful_pages:
            return {'error': 'No pages processed successfully'}
        
        # Combine all text with clear page separators
        combined_text_parts = []
        for i, result in enumerate(successful_pages):
            combined_text_parts.append(f"--- Page {i + 1} ---\n{result.extracted_text}")
        
        combined_text = '\n\n'.join(combined_text_parts)
        
        # Aggregate key entities across pages
        all_entities = {}
        for result in successful_pages:
            entities = result.structured_analysis.get('key_entities', {})
            for entity_type, entity_list in entities.items():
                if entity_type not in all_entities:
                    all_entities[entity_type] = set()
                all_entities[entity_type].update(entity_list)
        
        # Convert sets back to lists for JSON serialization
        all_entities = {k: list(v) for k, v in all_entities.items()}
        
        # Determine primary content type
        content_types = [r.content_type for r in successful_pages]
        primary_type = max(set(content_types), key=content_types.count)
        
        return {
            'primary_content_type': primary_type,
            'average_confidence': sum([r.confidence for r in successful_pages]) / len(successful_pages),
            'combined_text_content': combined_text,
            'aggregated_entities': all_entities,
            'successful_pages': len(successful_pages),
            'total_pages': len(page_results),
            'success_rate': len(successful_pages) / len(page_results),
            'document_analysis_summary': 'Processed with Claude Vision focusing on comprehensive text extraction and entity identification'
        }