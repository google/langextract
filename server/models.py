"""Pydantic models for the LangExtract API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CharIntervalModel(BaseModel):
    start_pos: int | None = None
    end_pos: int | None = None


class ExtractionInput(BaseModel):
    """Extraction used in few-shot examples (input)."""

    extraction_class: str
    extraction_text: str
    description: str | None = None
    attributes: dict[str, str | list[str]] | None = None


class ExampleDataInput(BaseModel):
    """A single few-shot example with text and expected extractions."""

    text: str
    extractions: list[ExtractionInput] = Field(default_factory=list)


class DocumentInput(BaseModel):
    """A document for batch extraction."""

    text: str
    document_id: str | None = None
    additional_context: str | None = None


class ExtractRequest(BaseModel):
    """Request body for the /extract endpoint."""

    text: str = Field(
        ..., description="The source text to extract information from, or a URL."
    )
    prompt_description: str = Field(
        ..., description="Instructions for what to extract from the text."
    )
    examples: list[ExampleDataInput] = Field(
        ...,
        min_length=1,
        description="Few-shot examples to guide the extraction.",
    )
    model_id: str = Field(
        default="gemini-2.5-flash",
        description="The model ID to use for extraction.",
    )
    api_key: str | None = Field(
        default=None,
        description="API key for the LLM provider. Falls back to server env var.",
    )
    max_char_buffer: int = Field(
        default=1000,
        description="Max number of characters per chunk for inference.",
    )
    temperature: float | None = Field(
        default=None,
        description="Sampling temperature for generation.",
    )
    fence_output: bool | None = Field(
        default=None,
        description="Whether to expect/generate fenced output.",
    )
    use_schema_constraints: bool = Field(
        default=True,
        description="Whether to generate schema constraints for models.",
    )
    batch_length: int = Field(
        default=10,
        description="Number of text chunks processed per batch.",
    )
    max_workers: int = Field(
        default=10,
        description="Maximum parallel workers for concurrent processing.",
    )
    additional_context: str | None = Field(
        default=None,
        description="Additional context to be added to the prompt.",
    )
    resolver_params: dict | None = Field(
        default=None,
        description="Parameters for the output resolver/parser.",
    )
    language_model_params: dict | None = Field(
        default=None,
        description="Additional parameters for the language model.",
    )
    model_url: str | None = Field(
        default=None,
        description="Endpoint URL for self-hosted or on-prem models.",
    )
    extraction_passes: int = Field(
        default=1,
        description="Number of sequential extraction passes.",
    )
    context_window_chars: int | None = Field(
        default=None,
        description="Characters from previous chunk to include as context.",
    )
    fetch_urls: bool = Field(
        default=True,
        description="Whether to auto-download content when input is a URL.",
    )


class BatchExtractRequest(BaseModel):
    """Request body for the /extract/batch endpoint."""

    documents: list[DocumentInput] = Field(
        ...,
        min_length=1,
        description="List of documents to extract information from.",
    )
    prompt_description: str = Field(
        ..., description="Instructions for what to extract from the text."
    )
    examples: list[ExampleDataInput] = Field(
        ...,
        min_length=1,
        description="Few-shot examples to guide the extraction.",
    )
    model_id: str = Field(
        default="gemini-2.5-flash",
        description="The model ID to use for extraction.",
    )
    api_key: str | None = Field(
        default=None,
        description="API key for the LLM provider. Falls back to server env var.",
    )
    max_char_buffer: int = Field(default=1000)
    temperature: float | None = Field(default=None)
    fence_output: bool | None = Field(default=None)
    use_schema_constraints: bool = Field(default=True)
    batch_length: int = Field(default=10)
    max_workers: int = Field(default=10)
    additional_context: str | None = Field(default=None)
    resolver_params: dict | None = Field(default=None)
    language_model_params: dict | None = Field(default=None)
    model_url: str | None = Field(default=None)
    extraction_passes: int = Field(default=1)
    context_window_chars: int | None = Field(default=None)


class ExtractionResponse(BaseModel):
    """A single extraction in the response."""

    extraction_class: str
    extraction_text: str
    char_interval: CharIntervalModel | None = None
    alignment_status: str | None = None
    extraction_index: int | None = None
    group_index: int | None = None
    description: str | None = None
    attributes: dict[str, str | list[str]] | None = None


class AnnotatedDocumentResponse(BaseModel):
    """Response for a single annotated document."""

    document_id: str
    text: str | None = None
    extractions: list[ExtractionResponse] = Field(default_factory=list)


class ExtractResponse(BaseModel):
    """Response body for the /extract endpoint."""

    document: AnnotatedDocumentResponse


class BatchExtractResponse(BaseModel):
    """Response body for the /extract/batch endpoint."""

    documents: list[AnnotatedDocumentResponse]


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


class ErrorResponse(BaseModel):
    detail: str
