"""FastAPI web service for LangExtract.

Exposes the langextract library via HTTP endpoints so other services
can call extraction over REST.
"""

from __future__ import annotations

import dataclasses
import os
import re
import traceback
from typing import Any

from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import requests as http_requests
import uvicorn

import langextract as lx
from langextract.core.data import Extraction, ExampleData
from langextract.io import SCRAPER_DEFAULT, SCRAPER_FIRECRAWL

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
    scraper: str = Field(
        default="default",
        description="Scraping backend: 'default' or 'firecrawl'",
    )
    firecrawl_api_key: str | None = Field(
        default=None,
        description="Firecrawl API key (falls back to FIRECRAWL_API_KEY env var)",
    )


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


def _fetch_url_text(
    url: str,
    scraper: str = "default",
    firecrawl_api_key: str | None = None,
) -> str:
    """Fetch a URL and return plain text.

    Args:
        url: The URL to fetch.
        scraper: 'default' for requests+BeautifulSoup, 'firecrawl' for
            the Firecrawl API.
        firecrawl_api_key: API key when using scraper='firecrawl'.
    """
    if scraper == SCRAPER_FIRECRAWL:
        from langextract.io import _download_with_firecrawl
        return _download_with_firecrawl(
            url=url,
            firecrawl_api_key=firecrawl_api_key,
            show_progress=False,
        )

    try:
        resp = http_requests.get(url, timeout=60)
        resp.raise_for_status()
    except http_requests.exceptions.SSLError:
        # Retry without certificate verification for sites with bad chains
        resp = http_requests.get(url, timeout=60, verify=False)
        resp.raise_for_status()

    # Detect encoding
    text = None
    raw = resp.content
    for enc in ("utf-8", "latin-1", "ascii", "utf-16"):
        try:
            text = raw.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if text is None:
        text = raw.decode("utf-8", errors="replace")

    # Convert HTML to plain text if needed
    ct = resp.headers.get("Content-Type", "")
    is_html = "text/html" in ct or text.strip()[:50].lower().startswith(("<!doctype", "<html"))
    if is_html:
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer",
                         "nav", "aside", "form"]):
            tag.decompose()
        # Try to find main content
        main = (soup.find("main") or soup.find(attrs={"role": "main"})
                or soup.find("article") or soup.find("body") or soup)
        # Convert block elements to line breaks
        for br in main.find_all("br"):
            br.replace_with("\n")
        for block in main.find_all(["h1","h2","h3","h4","h5","h6",
                                     "p","li","tr","div","article","section"]):
            block.insert_before("\n\n")
        text = main.get_text()
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text


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
            scraper=req.scraper,
            firecrawl_api_key=req.firecrawl_api_key,
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
    prompt_description: str = Form(...),
    example_text: str = Form(""),
    example_class: str = Form(""),
    example_extraction: str = Form(""),
    model_id: str = Form("gemini-2.5-flash"),
    api_key: str = Form(""),
    max_char_buffer: int = Form(2000),
    batch_length: int = Form(15),
    max_workers: int = Form(10),
    file: UploadFile | None = File(None),
    url: str = Form(""),
    scraper: str = Form("default"),
    firecrawl_api_key: str = Form(""),
):
    """Accept a file upload or URL and run LangExtract."""
    # Determine input source
    has_file = file is not None and file.filename
    has_url = bool(url.strip())

    if not has_file and not has_url:
        raise HTTPException(status_code=422, detail="Provide a file or a URL.")
    if has_file and has_url:
        raise HTTPException(status_code=422, detail="Provide a file or a URL, not both.")

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

    if has_file:
        content = await file.read()
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=422,
                detail="Could not read file as text. Please upload a plain-text file.",
            )
    else:
        # Fetch URL ourselves so we can handle SSL issues
        try:
            text = _fetch_url_text(
                url.strip(),
                scraper=scraper,
                firecrawl_api_key=firecrawl_api_key if firecrawl_api_key else None,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Could not fetch URL: {exc}",
            ) from exc

    try:
        result = lx.extract(
            text_or_documents=text,
            prompt_description=prompt_description,
            examples=examples,
            model_id=model_id,
            api_key=api_key if api_key else None,
            max_char_buffer=max_char_buffer,
            batch_length=batch_length,
            max_workers=max_workers,
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
  .container { max-width: 860px; margin: 0 auto; padding: 2rem 1rem; }
  h1 { font-size: 1.8rem; margin-bottom: .25rem; }
  .subtitle { color: #666; margin-bottom: 2rem; font-size: .95rem; }
  .card {
    background: #fff; border-radius: 12px; padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.08); margin-bottom: 1.5rem;
  }
  .card h2 { font-size: 1.1rem; margin-bottom: 1rem; color: #333; }
  label { display: block; font-weight: 600; font-size: .85rem; margin-bottom: .3rem; color: #444; }
  input[type=text], input[type=url], input[type=number], textarea, select {
    width: 100%; padding: .6rem .75rem; border: 1px solid #ddd; border-radius: 8px;
    font-size: .9rem; font-family: inherit; margin-bottom: .75rem;
    transition: border-color .2s;
  }
  input:focus, textarea:focus, select:focus { outline: none; border-color: #4a6cf7; }
  textarea { resize: vertical; min-height: 60px; }

  /* Tabs for source input */
  .tabs { display: flex; gap: 0; margin-bottom: 1rem; }
  .tab {
    flex: 1; padding: .6rem; text-align: center; font-size: .85rem; font-weight: 600;
    cursor: pointer; border: 1px solid #ddd; background: #fafbfc; color: #666;
    transition: .2s;
  }
  .tab:first-child { border-radius: 8px 0 0 8px; }
  .tab:last-child { border-radius: 0 8px 8px 0; }
  .tab.active { background: #4a6cf7; color: #fff; border-color: #4a6cf7; }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }

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

  /* Preset chips */
  .presets { display: flex; flex-wrap: wrap; gap: .5rem; margin-bottom: 1rem; }
  .preset-chip {
    padding: .35rem .75rem; border-radius: 20px; font-size: .8rem; font-weight: 500;
    cursor: pointer; border: 1px solid #ddd; background: #fafbfc; color: #555;
    transition: .2s;
  }
  .preset-chip:hover { border-color: #4a6cf7; color: #4a6cf7; }
  .preset-chip.active { background: #4a6cf7; color: #fff; border-color: #4a6cf7; }

  button[type=submit] {
    width: 100%; padding: .75rem; font-size: 1rem; font-weight: 600;
    background: #4a6cf7; color: #fff; border: none; border-radius: 10px;
    cursor: pointer; transition: background .2s;
  }
  button[type=submit]:hover { background: #3b5de7; }
  button[type=submit]:disabled { background: #a0b0f0; cursor: not-allowed; }

  #results { display: none; }
  .results-header { display: flex; justify-content: space-between; align-items: center;
    margin-bottom: .75rem; }
  .results-count { font-size: .85rem; color: #666; }
  .filter-bar { display: flex; gap: .5rem; flex-wrap: wrap; margin-bottom: .75rem; }
  .filter-chip {
    padding: .25rem .6rem; border-radius: 15px; font-size: .75rem; font-weight: 500;
    cursor: pointer; border: 1px solid #ddd; background: #fff; color: #555;
    transition: .15s;
  }
  .filter-chip:hover { border-color: #4a6cf7; }
  .filter-chip.active { background: #eef1ff; border-color: #4a6cf7; color: #4a6cf7; }
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
  .timer { font-size: .85rem; color: #888; margin-top: .5rem; text-align: center; }
  details { margin-top: .5rem; }
  details summary { cursor: pointer; font-size: .85rem; color: #666; }
  .collapse-content { margin-top: .5rem; }

  /* Color palette for different classes */
  .tag-0 { background: #4a6cf7; } .tag-1 { background: #e53e3e; }
  .tag-2 { background: #38a169; } .tag-3 { background: #d69e2e; }
  .tag-4 { background: #805ad5; } .tag-5 { background: #dd6b20; }
  .tag-6 { background: #319795; } .tag-7 { background: #b83280; }
  .border-0 { border-left-color: #4a6cf7; } .border-1 { border-left-color: #e53e3e; }
  .border-2 { border-left-color: #38a169; } .border-3 { border-left-color: #d69e2e; }
  .border-4 { border-left-color: #805ad5; } .border-5 { border-left-color: #dd6b20; }
  .border-6 { border-left-color: #319795; } .border-7 { border-left-color: #b83280; }

  @media (max-width: 600px) {
    .row { flex-direction: column; gap: .5rem; }
  }
</style>
</head>
<body>
<div class="container">
  <h1>LangExtract</h1>
  <p class="subtitle">Extract structured information from documents using LLMs.</p>

  <form id="uploadForm">
    <!-- Step 1: Source -->
    <div class="card">
      <h2>1. Document Source</h2>
      <div class="tabs">
        <div class="tab active" data-tab="file">Upload File</div>
        <div class="tab" data-tab="url">Paste URL</div>
      </div>
      <div class="tab-panel active" id="panel-file">
        <div class="drop-zone" id="dropZone">
          <p>Drag &amp; drop a text file here, or click to browse</p>
          <div class="filename" id="fileName"></div>
        </div>
        <input type="file" id="fileInput" accept=".txt,.csv,.html,.htm,.md,.json,.xml,.log" hidden>
        <div id="docPreview" class="doc-preview" style="display:none"></div>
      </div>
      <div class="tab-panel" id="panel-url">
        <label for="urlInput">Document URL</label>
        <input type="url" id="urlInput"
          placeholder="https://www.ordenjuridico.gob.mx/Documentos/Federal/html/wo17186.html">
        <div class="row" style="margin-top:.75rem;">
          <div>
            <label for="scraperSelect">Scraping method</label>
            <select id="scraperSelect">
              <option value="default" selected>Default (requests + BeautifulSoup)</option>
              <option value="firecrawl">Firecrawl (JS-rendered, anti-bot, gov sites)</option>
            </select>
          </div>
          <div id="firecrawlKeyGroup" style="display:none;">
            <label for="firecrawlKey">Firecrawl API Key</label>
            <input type="text" id="firecrawlKey" placeholder="fc-...">
          </div>
        </div>
        <p style="font-size:.8rem;color:#888;">Use Firecrawl for pages that fail with the default scraper (JS-heavy, government, anti-bot sites).</p>
      </div>
    </div>

    <!-- Step 2: Preset or custom prompt -->
    <div class="card">
      <h2>2. What to Extract</h2>
      <p style="font-size:.85rem;color:#666;margin-bottom:.75rem;">
        Pick a preset or write your own prompt. Presets auto-fill the example below.
      </p>
      <div class="presets" id="presets">
        <div class="preset-chip" data-preset="legal_articles">Legal Articles</div>
        <div class="preset-chip" data-preset="legal_definitions">Definitions</div>
        <div class="preset-chip" data-preset="legal_obligations">Obligations &amp; Rights</div>
        <div class="preset-chip" data-preset="legal_penalties">Penalties &amp; Sanctions</div>
        <div class="preset-chip" data-preset="entities">People &amp; Orgs</div>
        <div class="preset-chip" data-preset="dates">Dates &amp; Deadlines</div>
        <div class="preset-chip" data-preset="custom">Custom</div>
      </div>
      <label for="prompt">Extraction prompt</label>
      <textarea id="prompt" rows="3"
        placeholder="Describe what the model should extract from the document..."></textarea>
    </div>

    <!-- Step 3: Example -->
    <div class="card">
      <h2>3. Few-Shot Example</h2>
      <p style="font-size:.85rem;color:#666;margin-bottom:.75rem;">
        One example so the model knows the expected output format.
      </p>
      <label for="exText">Example source text</label>
      <textarea id="exText" rows="2"
        placeholder="e.g. Articulo 1o.- Las disposiciones de este Codigo regiran en toda la Republica..."></textarea>
      <div class="row">
        <div>
          <label for="exClass">Class / type</label>
          <input type="text" id="exClass" placeholder="e.g. articulo">
        </div>
        <div>
          <label for="exExtraction">Extracted text</label>
          <input type="text" id="exExtraction"
            placeholder="e.g. Las disposiciones de este Codigo regiran en toda la Republica">
        </div>
      </div>
    </div>

    <!-- Step 4: Settings -->
    <div class="card">
      <h2>4. Settings</h2>
      <div class="row">
        <div>
          <label for="modelId">Model</label>
          <select id="modelId">
            <option value="gemini-2.5-flash" selected>gemini-2.5-flash (fast)</option>
            <option value="gemini-2.5-pro">gemini-2.5-pro (best)</option>
            <option value="gpt-4o">gpt-4o</option>
          </select>
        </div>
        <div>
          <label for="apiKey">API Key (optional if set on server)</label>
          <input type="text" id="apiKey" placeholder="AIza... or sk-...">
        </div>
      </div>
      <details>
        <summary>Advanced parameters</summary>
        <div class="collapse-content">
          <div class="row">
            <div>
              <label for="chunkSize">Chunk size (chars)</label>
              <input type="number" id="chunkSize" value="2000" min="500" max="10000">
            </div>
            <div>
              <label for="batchLen">Batch length</label>
              <input type="number" id="batchLen" value="15" min="1" max="50">
            </div>
            <div>
              <label for="workers">Max workers</label>
              <input type="number" id="workers" value="10" min="1" max="20">
            </div>
          </div>
        </div>
      </details>
    </div>

    <button type="submit" id="submitBtn">Extract</button>
    <div class="timer" id="timer" style="display:none"></div>
  </form>

  <!-- Results -->
  <div class="card" id="results" style="margin-top:1.5rem;">
    <div class="results-header">
      <h2>Results</h2>
      <span class="results-count" id="resultsCount"></span>
    </div>
    <div class="filter-bar" id="filterBar"></div>
    <div id="resultBody"></div>
  </div>
</div>

<script>
/* ── Presets ─────────────────────────────────────────────────────── */
const PRESETS = {
  legal_articles: {
    prompt: "Extract every numbered article (Articulo) from this legal document. For each article, extract its full text content including any sub-sections (fracciones). The extraction_class should be 'articulo' and extraction_text should be the complete article text.",
    exText: "Articulo 1o.- Las disposiciones de este Codigo regiran en toda la Republica en asuntos del orden federal.",
    exClass: "articulo",
    exExtraction: "Las disposiciones de este Codigo regiran en toda la Republica en asuntos del orden federal."
  },
  legal_definitions: {
    prompt: "Extract all legal definitions found in this document. Look for terms being formally defined, including phrases like 'se entiende por', 'se considera', 'para los efectos de'. The extraction_class should be 'definicion'.",
    exText: "Se entiende por domicilio de una persona fisica el lugar donde reside con el proposito de establecerse en el.",
    exClass: "definicion",
    exExtraction: "domicilio de una persona fisica el lugar donde reside con el proposito de establecerse en el"
  },
  legal_obligations: {
    prompt: "Extract all obligations, rights, and duties described in this legal document. Look for language indicating requirements ('debe', 'esta obligado', 'tiene derecho', 'podra'). Classify each as 'obligacion', 'derecho', or 'prohibicion'.",
    exText: "Toda persona tiene derecho a la proteccion de la ley contra injerencias arbitrarias en su vida privada.",
    exClass: "derecho",
    exExtraction: "Toda persona tiene derecho a la proteccion de la ley contra injerencias arbitrarias en su vida privada"
  },
  legal_penalties: {
    prompt: "Extract all penalties, sanctions, fines, and consequences described in this legal document. Look for monetary amounts, imprisonment terms, and other sanctions. The extraction_class should be 'sancion'.",
    exText: "El que incurra en esta falta sera sancionado con multa de hasta cien dias de salario minimo.",
    exClass: "sancion",
    exExtraction: "multa de hasta cien dias de salario minimo"
  },
  entities: {
    prompt: "Extract all named entities: people, organizations, institutions, and government bodies mentioned in this document.",
    exText: "El Presidente de la Republica y el Congreso de la Union estableceran las bases.",
    exClass: "institucion",
    exExtraction: "Congreso de la Union"
  },
  dates: {
    prompt: "Extract all dates, deadlines, time periods, and temporal references from this document. Include specific dates, durations, and statutory time limits.",
    exText: "El plazo para interponer el recurso sera de quince dias habiles contados a partir del dia siguiente.",
    exClass: "plazo",
    exExtraction: "quince dias habiles contados a partir del dia siguiente"
  },
  custom: {
    prompt: "", exText: "", exClass: "", exExtraction: ""
  }
};

/* ── DOM refs ────────────────────────────────────────────────────── */
const dropZone    = document.getElementById('dropZone');
const fileInput   = document.getElementById('fileInput');
const fileNameEl  = document.getElementById('fileName');
const docPreview  = document.getElementById('docPreview');
const urlInput    = document.getElementById('urlInput');
const form        = document.getElementById('uploadForm');
const submitBtn   = document.getElementById('submitBtn');
const resultsCard = document.getElementById('results');
const resultBody  = document.getElementById('resultBody');
const filterBar   = document.getElementById('filterBar');
const resultsCount= document.getElementById('resultsCount');
const timerEl     = document.getElementById('timer');

let selectedFile = null;
let allExtractions = [];
let classColors = {};

/* ── Scraper toggle ─────────────────────────────────────────────── */
const scraperSelect = document.getElementById('scraperSelect');
const firecrawlKeyGroup = document.getElementById('firecrawlKeyGroup');
scraperSelect.addEventListener('change', () => {
  firecrawlKeyGroup.style.display = scraperSelect.value === 'firecrawl' ? 'block' : 'none';
});

/* ── Tab switching ───────────────────────────────────────────────── */
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('panel-' + tab.dataset.tab).classList.add('active');
  });
});

/* ── Preset selection ────────────────────────────────────────────── */
document.querySelectorAll('.preset-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
    chip.classList.add('active');
    const p = PRESETS[chip.dataset.preset];
    if (p) {
      document.getElementById('prompt').value = p.prompt;
      document.getElementById('exText').value = p.exText;
      document.getElementById('exClass').value = p.exClass;
      document.getElementById('exExtraction').value = p.exExtraction;
    }
  });
});

/* ── File handling ───────────────────────────────────────────────── */
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
  fileNameEl.textContent = f.name + ' (' + (f.size / 1024).toFixed(1) + ' KB)';
  const reader = new FileReader();
  reader.onload = () => {
    docPreview.style.display = 'block';
    docPreview.textContent = reader.result.slice(0, 2000)
      + (reader.result.length > 2000 ? '\\n...truncated...' : '');
  };
  reader.readAsText(f);
}

/* ── Form submit ─────────────────────────────────────────────────── */
form.addEventListener('submit', async e => {
  e.preventDefault();
  const isUrl = document.querySelector('.tab.active').dataset.tab === 'url';
  const hasFile = selectedFile != null;
  const hasUrl  = urlInput.value.trim().length > 0;

  if (!isUrl && !hasFile) return alert('Please select a file first.');
  if (isUrl && !hasUrl) return alert('Please enter a URL.');
  if (!document.getElementById('prompt').value.trim()) return alert('Please enter a prompt.');
  if (!document.getElementById('exText').value.trim()) return alert('Please provide an example.');

  submitBtn.disabled = true;
  submitBtn.innerHTML = '<span class="spinner"></span> Extracting...';
  resultsCard.style.display = 'none';
  timerEl.style.display = 'block';

  const t0 = Date.now();
  const tickId = setInterval(() => {
    const s = ((Date.now() - t0) / 1000).toFixed(0);
    timerEl.textContent = 'Processing... ' + s + 's';
  }, 1000);

  const fd = new FormData();
  fd.append('prompt_description', document.getElementById('prompt').value);
  fd.append('example_text', document.getElementById('exText').value);
  fd.append('example_class', document.getElementById('exClass').value);
  fd.append('example_extraction', document.getElementById('exExtraction').value);
  fd.append('model_id', document.getElementById('modelId').value);
  fd.append('api_key', document.getElementById('apiKey').value);
  fd.append('max_char_buffer', document.getElementById('chunkSize').value);
  fd.append('batch_length', document.getElementById('batchLen').value);
  fd.append('max_workers', document.getElementById('workers').value);

  if (isUrl) {
    fd.append('url', urlInput.value.trim());
    fd.append('scraper', scraperSelect.value);
    fd.append('firecrawl_api_key', document.getElementById('firecrawlKey').value);
  } else {
    fd.append('file', selectedFile);
  }

  try {
    const res = await fetch('/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Server error');
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    renderResults(data, elapsed);
  } catch (err) {
    resultsCard.style.display = 'block';
    resultBody.innerHTML = '<div class="error-msg">Error: ' + escHtml(err.message) + '</div>';
  } finally {
    clearInterval(tickId);
    timerEl.style.display = 'none';
    submitBtn.disabled = false;
    submitBtn.textContent = 'Extract';
  }
});

/* ── Render results ──────────────────────────────────────────────── */
function renderResults(data, elapsed) {
  resultsCard.style.display = 'block';
  allExtractions = data.extractions || [];

  if (allExtractions.length === 0) {
    resultsCount.textContent = '';
    filterBar.innerHTML = '';
    resultBody.innerHTML = '<p style="color:#888;">No extractions found.</p>';
    return;
  }

  // Build class color map
  const classes = [...new Set(allExtractions.map(e => e.extraction_class))];
  classColors = {};
  classes.forEach((c, i) => { classColors[c] = i % 8; });

  // Summary
  resultsCount.textContent = allExtractions.length + ' result(s) in ' + elapsed + 's';

  // Filter chips
  filterBar.innerHTML = '<div class="filter-chip active" data-filter="__all__">All</div>'
    + classes.map(c =>
      '<div class="filter-chip" data-filter="' + escAttr(c) + '">'
      + escHtml(c) + ' (' + allExtractions.filter(e => e.extraction_class === c).length + ')'
      + '</div>'
    ).join('');

  filterBar.querySelectorAll('.filter-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      filterBar.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      const f = chip.dataset.filter;
      renderList(f === '__all__' ? allExtractions : allExtractions.filter(e => e.extraction_class === f));
    });
  });

  renderList(allExtractions);
}

function renderList(items) {
  resultBody.innerHTML = items.map(ex => {
    const ci = classColors[ex.extraction_class] || 0;
    return '<div class="extraction-item border-' + ci + '">'
      + '<span class="class-tag tag-' + ci + '">' + escHtml(ex.extraction_class) + '</span>'
      + '<div class="text">' + escHtml(ex.extraction_text) + '</div>'
      + '<div class="meta">'
        + (ex.start_pos != null ? 'pos ' + ex.start_pos + '-' + ex.end_pos : '')
        + (ex.alignment_status ? ' &middot; ' + ex.alignment_status : '')
        + (ex.description ? ' &middot; ' + escHtml(ex.description) : '')
      + '</div>'
      + '</div>';
  }).join('');
}

function escHtml(s) {
  const d = document.createElement('div'); d.textContent = s; return d.innerHTML;
}
function escAttr(s) {
  return s.replace(/&/g,'&amp;').replace(/"/g,'&quot;');
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
