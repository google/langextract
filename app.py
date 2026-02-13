"""FastAPI web service for LangExtract.

Exposes the langextract library via HTTP endpoints so other services
can call extraction over REST.
"""

from __future__ import annotations

import dataclasses
import os
import traceback
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

import langextract as lx
from langextract.core.data import Extraction, ExampleData

app = FastAPI(
    title="LangExtract API",
    description="HTTP API for extracting structured information from text using LLMs.",
    version="1.0.0",
)


# ── Request / Response Models ───────────────────────────────────────────


class ExtractionExample(BaseModel):
    """A single example extraction used to guide the model."""

    extraction_class: str = Field(description="The class/type of the extraction")
    extraction_text: str = Field(description="The text of the extraction")
    description: str | None = Field(default=None, description="Optional description")
    attributes: dict[str, str | list[str]] | None = Field(
        default=None, description="Optional attributes dict"
    )


class ExampleItem(BaseModel):
    """One few-shot example: source text + its expected extractions."""

    text: str = Field(description="The source text for this example")
    extractions: list[ExtractionExample] = Field(
        description="Extractions found in the text"
    )


class ExtractRequest(BaseModel):
    """Body for POST /extract."""

    text: str | None = Field(
        default=None,
        description="Raw text to extract from (mutually exclusive with 'url')",
    )
    url: str | None = Field(
        default=None,
        description="URL whose content will be fetched and extracted (mutually exclusive with 'text')",
    )
    prompt_description: str = Field(
        description="Instructions describing what to extract"
    )
    examples: list[ExampleItem] = Field(
        description="At least one few-shot example to guide extraction"
    )
    model_id: str = Field(
        default="gemini-2.5-flash",
        description="LLM model ID",
    )
    api_key: str | None = Field(
        default=None,
        description="API key (falls back to LANGEXTRACT_API_KEY env var)",
    )
    max_char_buffer: int = Field(default=1000)
    temperature: float | None = Field(default=None)
    batch_length: int = Field(default=10)
    max_workers: int = Field(default=10)
    additional_context: str | None = Field(default=None)


class ExtractionOut(BaseModel):
    extraction_class: str
    extraction_text: str
    start_pos: int | None = None
    end_pos: int | None = None
    alignment_status: str | None = None
    description: str | None = None
    attributes: dict[str, Any] | None = None


class ExtractResponse(BaseModel):
    document_id: str
    extractions: list[ExtractionOut]


# ── Helpers ─────────────────────────────────────────────────────────────


def _to_example_data(item: ExampleItem) -> ExampleData:
    extractions = [
        Extraction(
            extraction_class=e.extraction_class,
            extraction_text=e.extraction_text,
            description=e.description,
            attributes=e.attributes,
        )
        for e in item.extractions
    ]
    return ExampleData(text=item.text, extractions=extractions)


def _annotated_doc_to_response(doc) -> ExtractResponse:
    extractions_out: list[ExtractionOut] = []
    for ext in doc.extractions or []:
        extractions_out.append(
            ExtractionOut(
                extraction_class=ext.extraction_class,
                extraction_text=ext.extraction_text,
                start_pos=ext.char_interval.start_pos if ext.char_interval else None,
                end_pos=ext.char_interval.end_pos if ext.char_interval else None,
                alignment_status=ext.alignment_status.value if ext.alignment_status else None,
                description=ext.description,
                attributes=ext.attributes,
            )
        )
    return ExtractResponse(
        document_id=doc.document_id,
        extractions=extractions_out,
    )


# ── Endpoints ───────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
def home():
    return _WEB_UI_HTML


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractResponse | list[ExtractResponse])
def extract(req: ExtractRequest):
    if not req.text and not req.url:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'text' or 'url'.",
        )
    if req.text and req.url:
        raise HTTPException(
            status_code=422,
            detail="Provide 'text' or 'url', not both.",
        )

    examples = [_to_example_data(e) for e in req.examples]
    input_value = req.url if req.url else req.text

    try:
        result = lx.extract(
            text_or_documents=input_value,
            prompt_description=req.prompt_description,
            examples=examples,
            model_id=req.model_id,
            api_key=req.api_key,
            max_char_buffer=req.max_char_buffer,
            temperature=req.temperature,
            batch_length=req.batch_length,
            max_workers=req.max_workers,
            additional_context=req.additional_context,
            fetch_urls=bool(req.url),
            show_progress=False,
        )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if isinstance(result, list):
        return [_annotated_doc_to_response(doc) for doc in result]
    return _annotated_doc_to_response(result)


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    prompt_description: str = Form(...),
    example_text: str = Form(""),
    example_class: str = Form(""),
    example_extraction: str = Form(""),
    model_id: str = Form("gemini-2.5-flash"),
    api_key: str = Form(""),
):
    """Accept a file upload, extract its text, and run LangExtract."""
    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=422,
            detail="Could not read file as text. Please upload a plain-text file (.txt, .csv, .html, etc.).",
        )

    if not example_text or not example_class or not example_extraction:
        raise HTTPException(
            status_code=422,
            detail="At least one example (text, class, extraction) is required.",
        )

    examples = [
        ExampleData(
            text=example_text,
            extractions=[
                Extraction(
                    extraction_class=example_class,
                    extraction_text=example_extraction,
                )
            ],
        )
    ]

    try:
        result = lx.extract(
            text_or_documents=text,
            prompt_description=prompt_description,
            examples=examples,
            model_id=model_id,
            api_key=api_key if api_key else None,
            show_progress=False,
        )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    doc = result[0] if isinstance(result, list) else result
    resp = _annotated_doc_to_response(doc)
    return resp


# ── Web UI HTML ────────────────────────────────────────────────────────

_WEB_UI_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LangExtract</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: #f5f7fa; color: #1a1a2e; line-height: 1.6;
  }
  .container { max-width: 820px; margin: 0 auto; padding: 2rem 1rem; }
  h1 { font-size: 1.8rem; margin-bottom: .25rem; }
  .subtitle { color: #666; margin-bottom: 2rem; font-size: .95rem; }
  .card {
    background: #fff; border-radius: 12px; padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.08); margin-bottom: 1.5rem;
  }
  .card h2 { font-size: 1.1rem; margin-bottom: 1rem; color: #333; }
  label { display: block; font-weight: 600; font-size: .85rem; margin-bottom: .3rem; color: #444; }
  input[type=text], textarea, select {
    width: 100%; padding: .6rem .75rem; border: 1px solid #ddd; border-radius: 8px;
    font-size: .9rem; font-family: inherit; margin-bottom: .75rem;
    transition: border-color .2s;
  }
  input:focus, textarea:focus, select:focus { outline: none; border-color: #4a6cf7; }
  textarea { resize: vertical; min-height: 60px; }
  .drop-zone {
    border: 2px dashed #ccc; border-radius: 12px; padding: 2rem;
    text-align: center; cursor: pointer; transition: .2s;
    background: #fafbfc; margin-bottom: .75rem;
  }
  .drop-zone:hover, .drop-zone.dragover { border-color: #4a6cf7; background: #f0f4ff; }
  .drop-zone p { color: #888; font-size: .9rem; }
  .drop-zone .filename { color: #4a6cf7; font-weight: 600; margin-top: .5rem; }
  .row { display: flex; gap: 1rem; }
  .row > * { flex: 1; }
  button[type=submit] {
    width: 100%; padding: .75rem; font-size: 1rem; font-weight: 600;
    background: #4a6cf7; color: #fff; border: none; border-radius: 10px;
    cursor: pointer; transition: background .2s;
  }
  button[type=submit]:hover { background: #3b5de7; }
  button[type=submit]:disabled { background: #a0b0f0; cursor: not-allowed; }
  #results { display: none; }
  .extraction-item {
    padding: .75rem; border-left: 4px solid #4a6cf7;
    background: #f8f9ff; border-radius: 0 8px 8px 0; margin-bottom: .5rem;
  }
  .extraction-item .class-tag {
    display: inline-block; background: #4a6cf7; color: #fff;
    padding: .1rem .5rem; border-radius: 4px; font-size: .75rem; font-weight: 600;
    margin-bottom: .3rem;
  }
  .extraction-item .text { font-size: .95rem; }
  .extraction-item .meta { font-size: .75rem; color: #888; margin-top: .2rem; }
  .error-msg { color: #e53e3e; background: #fff5f5; padding: 1rem; border-radius: 8px; }
  .spinner {
    display: inline-block; width: 18px; height: 18px;
    border: 3px solid #fff; border-top-color: transparent;
    border-radius: 50%; animation: spin .6s linear infinite;
    vertical-align: middle; margin-right: .5rem;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .doc-preview { max-height: 200px; overflow-y: auto; background: #f9f9f9;
    padding: .75rem; border-radius: 8px; font-size: .85rem; white-space: pre-wrap;
    color: #555; margin-bottom: .75rem; border: 1px solid #eee;
  }
</style>
</head>
<body>
<div class="container">
  <h1>LangExtract</h1>
  <p class="subtitle">Upload a document and extract structured information using LLMs.</p>

  <form id="uploadForm">
    <div class="card">
      <h2>1. Upload Document</h2>
      <div class="drop-zone" id="dropZone">
        <p>Drag &amp; drop a text file here, or click to browse</p>
        <div class="filename" id="fileName"></div>
      </div>
      <input type="file" id="fileInput" accept=".txt,.csv,.html,.htm,.md,.json,.xml,.log" hidden>
      <div id="docPreview" class="doc-preview" style="display:none"></div>
    </div>

    <div class="card">
      <h2>2. Extraction Prompt</h2>
      <label for="prompt">What should be extracted?</label>
      <textarea id="prompt" placeholder="e.g. Extract all person names and their roles"></textarea>
    </div>

    <div class="card">
      <h2>3. Few-Shot Example</h2>
      <p style="font-size:.85rem;color:#666;margin-bottom:.75rem;">
        Give one example so the model knows what to look for.
      </p>
      <label for="exText">Example source text</label>
      <textarea id="exText" rows="2" placeholder="e.g. Dr. Jane Smith is the lead researcher."></textarea>
      <div class="row">
        <div>
          <label for="exClass">Class / type</label>
          <input type="text" id="exClass" placeholder="e.g. person">
        </div>
        <div>
          <label for="exExtraction">Extracted text</label>
          <input type="text" id="exExtraction" placeholder="e.g. Dr. Jane Smith">
        </div>
      </div>
    </div>

    <div class="card">
      <h2>4. Settings</h2>
      <div class="row">
        <div>
          <label for="modelId">Model</label>
          <select id="modelId">
            <option value="gemini-2.5-flash" selected>gemini-2.5-flash</option>
            <option value="gemini-2.5-pro">gemini-2.5-pro</option>
            <option value="gpt-4o">gpt-4o</option>
          </select>
        </div>
        <div>
          <label for="apiKey">API Key (optional if set on server)</label>
          <input type="text" id="apiKey" placeholder="sk-... or AIza...">
        </div>
      </div>
    </div>

    <button type="submit" id="submitBtn">Extract</button>
  </form>

  <div class="card" id="results" style="margin-top:1.5rem;">
    <h2>Results</h2>
    <div id="resultBody"></div>
  </div>
</div>

<script>
const dropZone   = document.getElementById('dropZone');
const fileInput   = document.getElementById('fileInput');
const fileName    = document.getElementById('fileName');
const docPreview  = document.getElementById('docPreview');
const form        = document.getElementById('uploadForm');
const submitBtn   = document.getElementById('submitBtn');
const resultsCard = document.getElementById('results');
const resultBody  = document.getElementById('resultBody');

let selectedFile = null;

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('dragover');
  if (e.dataTransfer.files.length) pickFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length) pickFile(fileInput.files[0]); });

function pickFile(f) {
  selectedFile = f;
  fileName.textContent = f.name + ' (' + (f.size / 1024).toFixed(1) + ' KB)';
  const reader = new FileReader();
  reader.onload = () => {
    docPreview.style.display = 'block';
    docPreview.textContent = reader.result.slice(0, 2000) + (reader.result.length > 2000 ? '\\n...truncated...' : '');
  };
  reader.readAsText(f);
}

form.addEventListener('submit', async e => {
  e.preventDefault();
  if (!selectedFile) return alert('Please select a file first.');
  if (!document.getElementById('prompt').value.trim()) return alert('Please enter a prompt.');
  if (!document.getElementById('exText').value.trim()) return alert('Please provide an example.');

  submitBtn.disabled = true;
  submitBtn.innerHTML = '<span class="spinner"></span> Extracting...';
  resultsCard.style.display = 'none';

  const fd = new FormData();
  fd.append('file', selectedFile);
  fd.append('prompt_description', document.getElementById('prompt').value);
  fd.append('example_text', document.getElementById('exText').value);
  fd.append('example_class', document.getElementById('exClass').value);
  fd.append('example_extraction', document.getElementById('exExtraction').value);
  fd.append('model_id', document.getElementById('modelId').value);
  fd.append('api_key', document.getElementById('apiKey').value);

  try {
    const res = await fetch('/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Server error');
    renderResults(data);
  } catch (err) {
    resultsCard.style.display = 'block';
    resultBody.innerHTML = '<div class="error-msg">Error: ' + escHtml(err.message) + '</div>';
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Extract';
  }
});

function renderResults(data) {
  resultsCard.style.display = 'block';
  if (!data.extractions || data.extractions.length === 0) {
    resultBody.innerHTML = '<p style="color:#888;">No extractions found.</p>';
    return;
  }
  resultBody.innerHTML = '<p style="margin-bottom:.75rem;font-size:.85rem;color:#666;">'
    + data.extractions.length + ' extraction(s) found</p>'
    + data.extractions.map(ex =>
      '<div class="extraction-item">'
      + '<span class="class-tag">' + escHtml(ex.extraction_class) + '</span>'
      + '<div class="text">' + escHtml(ex.extraction_text) + '</div>'
      + '<div class="meta">'
        + (ex.start_pos != null ? 'pos ' + ex.start_pos + '-' + ex.end_pos : '')
        + (ex.alignment_status ? ' &middot; ' + ex.alignment_status : '')
        + (ex.description ? ' &middot; ' + escHtml(ex.description) : '')
      + '</div>'
      + '</div>'
    ).join('');
}

function escHtml(s) {
  const d = document.createElement('div'); d.textContent = s; return d.innerHTML;
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
