"""FastAPI application wrapping the LangExtract library."""

from __future__ import annotations

import asyncio
import functools
import logging
import os
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import langextract as lx
from langextract.core import data
from langextract import data_lib

from server.models import (
    AnnotatedDocumentResponse,
    BatchExtractRequest,
    BatchExtractResponse,
    CharIntervalModel,
    ErrorResponse,
    ExtractRequest,
    ExtractResponse,
    ExtractionResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)

try:
    _VERSION = version("langextract")
except PackageNotFoundError:
    _VERSION = "dev"

app = FastAPI(
    title="LangExtract API",
    description="HTTP API for extracting structured information from text using LLMs.",
    version=_VERSION,
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_api_key(request_key: str | None) -> str | None:
    """Return the API key from the request, or fall back to env vars."""
    return (
        request_key
        or os.getenv("LANGEXTRACT_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )


def _build_examples(raw: list) -> list[data.ExampleData]:
    """Convert Pydantic example models to langextract ExampleData objects."""
    examples: list[data.ExampleData] = []
    for ex in raw:
        extractions = [
            data.Extraction(
                extraction_class=e.extraction_class,
                extraction_text=e.extraction_text,
                description=e.description,
                attributes=e.attributes,
            )
            for e in ex.extractions
        ]
        examples.append(data.ExampleData(text=ex.text, extractions=extractions))
    return examples


def _serialize_document(
    adoc: data.AnnotatedDocument,
) -> AnnotatedDocumentResponse:
    """Serialize an AnnotatedDocument to the API response model."""
    d = data_lib.annotated_document_to_dict(adoc)
    extractions = []
    for ext in d.get("extractions", []):
        ci = ext.get("char_interval")
        extractions.append(
            ExtractionResponse(
                extraction_class=ext.get("extraction_class", ""),
                extraction_text=ext.get("extraction_text", ""),
                char_interval=CharIntervalModel(**ci) if ci else None,
                alignment_status=ext.get("alignment_status"),
                extraction_index=ext.get("extraction_index"),
                group_index=ext.get("group_index"),
                description=ext.get("description"),
                attributes=ext.get("attributes"),
            )
        )
    return AnnotatedDocumentResponse(
        document_id=d.get("document_id", ""),
        text=d.get("text"),
        extractions=extractions,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok", version=_VERSION)


@app.post(
    "/extract",
    response_model=ExtractResponse,
    tags=["extraction"],
    summary="Extract structured information from text",
    responses={400: {"model": ErrorResponse}},
)
async def extract(req: ExtractRequest) -> ExtractResponse:
    """Extract structured information from a single text input.

    Wraps `langextract.extract()` — pass a prompt description and at least
    one few-shot example to guide the extraction.
    """
    api_key = _resolve_api_key(req.api_key)
    examples = _build_examples(req.examples)

    kwargs: dict = dict(
        text_or_documents=req.text,
        prompt_description=req.prompt_description,
        examples=examples,
        model_id=req.model_id,
        api_key=api_key,
        max_char_buffer=req.max_char_buffer,
        temperature=req.temperature,
        fence_output=req.fence_output,
        use_schema_constraints=req.use_schema_constraints,
        batch_length=req.batch_length,
        max_workers=req.max_workers,
        additional_context=req.additional_context,
        resolver_params=req.resolver_params,
        language_model_params=req.language_model_params,
        model_url=req.model_url,
        extraction_passes=req.extraction_passes,
        context_window_chars=req.context_window_chars,
        fetch_urls=req.fetch_urls,
        show_progress=False,
    )
    # Strip None values so langextract uses its own defaults
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, functools.partial(lx.extract, **kwargs)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Extraction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if isinstance(result, list):
        result = result[0]

    return ExtractResponse(document=_serialize_document(result))


@app.post(
    "/extract/batch",
    response_model=BatchExtractResponse,
    tags=["extraction"],
    summary="Extract structured information from multiple documents",
    responses={400: {"model": ErrorResponse}},
)
async def extract_batch(req: BatchExtractRequest) -> BatchExtractResponse:
    """Extract structured information from a batch of documents.

    Each document is processed and returned as an annotated document with
    extractions mapped to source text positions.
    """
    api_key = _resolve_api_key(req.api_key)
    examples = _build_examples(req.examples)

    documents = [
        data.Document(
            text=doc.text,
            document_id=doc.document_id,
            additional_context=doc.additional_context,
        )
        for doc in req.documents
    ]

    kwargs: dict = dict(
        text_or_documents=documents,
        prompt_description=req.prompt_description,
        examples=examples,
        model_id=req.model_id,
        api_key=api_key,
        max_char_buffer=req.max_char_buffer,
        temperature=req.temperature,
        fence_output=req.fence_output,
        use_schema_constraints=req.use_schema_constraints,
        batch_length=req.batch_length,
        max_workers=req.max_workers,
        additional_context=req.additional_context,
        resolver_params=req.resolver_params,
        language_model_params=req.language_model_params,
        model_url=req.model_url,
        extraction_passes=req.extraction_passes,
        context_window_chars=req.context_window_chars,
        show_progress=False,
    )
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    try:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None, functools.partial(lx.extract, **kwargs)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Batch extraction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if isinstance(results, data.AnnotatedDocument):
        results = [results]

    return BatchExtractResponse(
        documents=[_serialize_document(doc) for doc in results]
    )
