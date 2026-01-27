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

"""Centralized pattern definitions for built-in providers.

This module defines all patterns and priorities for built-in providers
in one place to avoid duplication.
"""

# Gemini provider patterns
GEMINI_PATTERNS = (r'^gemini',)
GEMINI_PRIORITY = 10

# OpenAI provider patterns
OPENAI_PATTERNS = (
    r'^gpt-4',
    r'^gpt4\.',
    r'^gpt-5',
    r'^gpt5\.',
)
OPENAI_PRIORITY = 10

# Ollama provider patterns
OLLAMA_PATTERNS = (
    # Standard Ollama naming patterns
    r'^gemma',  # gemma2:2b, gemma2:9b, etc.
    r'^llama',  # llama3.2:1b, llama3.1:8b, etc.
    r'^mistral',  # mistral:7b, mistral-nemo:12b, etc.
    r'^mixtral',  # mixtral:8x7b, mixtral:8x22b, etc.
    r'^phi',  # phi3:3.8b, phi3:14b, etc.
    r'^qwen',  # qwen2.5:0.5b to 72b
    r'^deepseek',  # deepseek-coder-v2, etc.
    r'^command-r',  # command-r:35b, command-r-plus:104b
    r'^starcoder',  # starcoder2:3b, starcoder2:7b, etc.
    r'^codellama',  # codellama:7b, codellama:13b, etc.
    r'^codegemma',  # codegemma:2b, codegemma:7b
    r'^tinyllama',  # tinyllama:1.1b
    r'^wizardcoder',  # wizardcoder:7b, wizardcoder:13b, etc.
    r'^gpt-oss',  # Open source GPT variants
    r'^ministral',  # ministral-3:3b, ministral-3:8b, etc.
    r'^devstral',  # devstral-small-2:24b, etc.
    r'^nemotron',  # nemotron-mini:4b, nemotron-3-nano:30b, etc.
    r'^granite',  # granite4:350m, granite4:1b, granite4:3b, etc.
    r'^glm',  # glm4:9b, etc.
    r'^rnj',  # rnj-1:8b, etc.
    r'^olmo',  # olmo-3:7b, etc.
    r'^smollm',  # smollm2:135m, smollm2:360m, smollm2:1.7b, etc.
    r'^lfm',  # lfm2.5-thinking, etc.
    r'^falcon',  # falcon3:1b, falcon3:3b, falcon3:7b, falcon3:10b, etc.
    r'^yi',  # yi:6b, yi:9b, yi-coder:1.5b, yi-coder:9b, etc.
    # HuggingFace model patterns
    r'^meta-llama/[Ll]lama',
    r'^google/gemma',
    r'^mistralai/[Mm]istral',
    r'^mistralai/[Mm]ixtral',
    r'^microsoft/phi',
    r'^Qwen/',
    r'^deepseek-ai/',
    r'^bigcode/starcoder',
    r'^codellama/',
    r'^TinyLlama/',
    r'^WizardLM/',
)
OLLAMA_PRIORITY = 10
