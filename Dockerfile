# Production Dockerfile for LangExtract
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install LangExtract from PyPI
RUN uv pip install --system --no-cache langextract

# Set default command
CMD ["python"]
