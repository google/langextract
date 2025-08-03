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

"""Base exceptions for LangExtract.

This module defines the base exception class that all LangExtract exceptions
inherit from. Individual modules define their own specific exceptions.
"""

__all__ = ["LangExtractError"]


class LangExtractError(Exception):
  """Base exception for all LangExtract errors.

  All exceptions raised by LangExtract should inherit from this class.
  This allows users to catch all LangExtract-specific errors with a single
  except clause.
  """