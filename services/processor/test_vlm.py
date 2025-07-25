import asyncio
from src.pipeline.vlm.vlm_processor import VLMProcessor

config = {'vlm': {'enabled': True, 'llava': {'model': 'llava:7b', 'api_endpoint': 'http://localhost:11434'}}}
processor = VLMProcessor(config)

# Test with a sample PDF
# result = asyncio.run(processor.process_document('path/to/test.pdf'))
print('VLM processor initialized successfully')
