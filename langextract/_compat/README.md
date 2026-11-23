# Backward Compatibility Layer

This directory contains backward compatibility shims for deprecated imports.

## Deprecation Timeline

All code in this directory will be removed in LangExtract v2.0.0.

## Migration Guide

The following imports are deprecated and should be updated:

### Inference Module
- `from langextract.inference import BaseLanguageModel` → `from langextract.core.base_model import BaseLanguageModel`
- `from langextract.inference import ScoredOutput` → `from langextract.core.types import ScoredOutput`
- `from langextract.inference import InferenceOutputError` → `from langextract.core.exceptions import InferenceOutputError`
- `from langextract.inference import GeminiLanguageModel` → `from langextract.providers.gemini import GeminiLanguageModel`
- `from langextract.inference import OpenAILanguageModel` → `from langextract.providers.openai import OpenAILanguageModel`
- `from langextract.inference import OllamaLanguageModel` → `from langextract.providers.ollama import OllamaLanguageModel`

### Schema Module
- `from langextract.schema import BaseSchema` → `from langextract.core.schema import BaseSchema`
- `from langextract.schema import Constraint` → `from langextract.core.schema import Constraint`
- `from langextract.schema import ConstraintType` → `from langextract.core.schema import ConstraintType`
- `from langextract.schema import EXTRACTIONS_KEY` → `from langextract.core.schema import EXTRACTIONS_KEY`
- `from langextract.schema import GeminiSchema` → `from langextract.providers.schemas.gemini import GeminiSchema`

### Exceptions Module
- All exceptions: `from langextract.exceptions import *` → `from langextract.core.exceptions import *`

### Registry Module
- `from langextract.registry import *` → `from langextract.plugins import *`
- `from langextract.providers.registry import *` → `from langextract.providers.router import *`

## For Contributors

Do not add new code to this directory. All new development should use the canonical imports from `core/` and `providers/`.
