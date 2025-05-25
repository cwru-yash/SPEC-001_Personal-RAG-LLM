# quick_test.py
from services.processor.src.pipeline.extractors.image_extractor import ImageExtractor
from services.processor.src.models.document import Document

# Test image processing
extractor = ImageExtractor()
doc = extractor.extract("/path/to/test/image.png")
print(f"Extracted text: {doc.text_content}")
print(f"Image analysis: {doc.metadata}")