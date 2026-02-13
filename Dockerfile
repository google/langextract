# Production Dockerfile for LangExtract API
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Ensure system CA certificates are up-to-date
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

# Install LangExtract and web server dependencies
RUN pip install --no-cache-dir langextract fastapi uvicorn[standard] python-multipart beautifulsoup4 certifi

# Copy the API application
COPY app.py .

# Render uses PORT env var; default to 10000
ENV PORT=10000
EXPOSE 10000

CMD ["python", "app.py"]
