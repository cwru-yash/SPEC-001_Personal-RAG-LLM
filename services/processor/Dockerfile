FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OCR and PDF processing
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    libmagic1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/
COPY conf/ /app/conf/

# Keep container running 
# CMD ["python", "-u", "-c", "import time; print('Processor service started'); time.sleep(float('inf'))"]
# Use a more stable approach to keep the container running
CMD ["tail", "-f", "/dev/null"]