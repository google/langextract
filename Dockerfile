# Production Dockerfile for LangExtract API
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install LangExtract and web server dependencies
RUN pip install --no-cache-dir langextract fastapi uvicorn[standard] python-multipart

# Copy the API application
COPY app.py .

# Render uses PORT env var; default to 10000
ENV PORT=10000
EXPOSE 10000

CMD ["python", "app.py"]
