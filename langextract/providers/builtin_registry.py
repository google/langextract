# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Built-in provider registration configuration.

This module defines the registration details for all built-in providers,
using patterns from the centralized patterns module.
"""

from typing import TypedDict

from langextract.providers import patterns


class ProviderConfig(TypedDict):
  """Configuration for a provider registration."""

  patterns: tuple[str, ...]
  target: str
  priority: int


# Built-in provider configurations using centralized patterns
BUILTIN_PROVIDERS: list[ProviderConfig] = [
    {
        'patterns': patterns.GEMINI_PATTERNS,
        'target': 'langextract.providers.gemini:GeminiLanguageModel',
        'priority': patterns.GEMINI_PRIORITY,
    },
    {
        'patterns': patterns.OLLAMA_PATTERNS,
        'target': 'langextract.providers.ollama:OllamaLanguageModel',
        'priority': patterns.OLLAMA_PRIORITY,
    },
    {
        'patterns': patterns.LLAMA_CPP_PATTERNS,
        'target': 'langextract.providers.llama_cpp:LlamaCppLanguageModel',
        'priority': patterns.LLAMA_CPP_PRIORITY,
    },
    {
        'patterns': patterns.OPENAI_PATTERNS,
        'target': 'langextract.providers.openai:OpenAILanguageModel',
        'priority': patterns.OPENAI_PRIORITY,
    },
]
