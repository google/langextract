"""FastAPI web service for LangExtract.

Exposes the langextract library via HTTP endpoints so other services
can call extraction over REST.
"""

from __future__ import annotations

import dataclasses
import os
import traceback
from typing import Any

from fastapi import FastAPI, HTTPException
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
