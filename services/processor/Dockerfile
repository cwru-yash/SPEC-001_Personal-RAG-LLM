FROM python:3.11-slim

WORKDIR /app

# Create required directories
RUN mkdir -p src conf

# Add a simple entrypoint script
RUN echo 'print("Processor service running...")' > src/main.py

# Keep container running
CMD ["python", "-u", "-c", "import time; print('Processor service started'); time.sleep(float('inf'))"]
