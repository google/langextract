# Production Dockerfile for LangExtract with libmagic support
FROM python:3.10-slim

# Install system dependencies including libmagic
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install LangExtract from PyPI
RUN pip install --no-cache-dir langextract

# Set default command
CMD ["python"]
