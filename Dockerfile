FROM python:3.10-slim

WORKDIR /app

# Copy dependency file first for better layer caching
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# Copy the full project (langextract library + server)
COPY . .

# Install the langextract package itself in editable mode
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
