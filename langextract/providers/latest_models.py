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

"""Pinned current-generation Claude and Grok model IDs.

Update this module when Anthropic or xAI ship a new generally-available
generation. Aliases here are the dateless API IDs (pinned snapshots, not
evergreen pointers).
"""

# Claude 5 generation (as of 2026-08). Haiku remains on 4.5.
CLAUDE_DEFAULT = 'claude-sonnet-5'
CLAUDE_FLAGSHIP = 'claude-opus-5'
CLAUDE_FRONTIER = 'claude-fable-5'
CLAUDE_FAST = 'claude-haiku-4-5'
CLAUDE_MODELS = (
    CLAUDE_DEFAULT,
    CLAUDE_FLAGSHIP,
    CLAUDE_FRONTIER,
    CLAUDE_FAST,
    'claude-haiku-4-5-20251001',
)

# Grok 4.6 generation (as of 2026-08).
GROK_DEFAULT = 'grok-4.6'
GROK_PREVIOUS = 'grok-4.5'
XAI_BASE_URL = 'https://api.x.ai/v1'
GROK_MODELS = (
    GROK_DEFAULT,
    GROK_PREVIOUS,
    'grok-4.3',
    'grok-4',
    'grok-build',
    'grok-build-0.1',
)
