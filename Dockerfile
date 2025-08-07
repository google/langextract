# Production Dockerfile for LangExtract
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install LangExtract from PyPI
RUN pip install --no-cache-dir langextract

# Set default command
CMD ["python"]
