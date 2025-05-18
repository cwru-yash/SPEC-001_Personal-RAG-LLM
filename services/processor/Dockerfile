# FROM python:3.11-slim

# WORKDIR /app

# # Create required directories
# RUN mkdir -p src conf

# # Add a simple entrypoint script
# RUN echo 'print("Processor service running...")' > src/main.py

# # Keep container running
# CMD ["python", "-u", "-c", "import time; print('Processor service started'); time.sleep(float('inf'))"]

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

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-dev

# Copy source code
COPY src/ /app/src/
COPY conf/ /app/conf/

# Run with Hydra
CMD ["python", "-m", "src.main"]