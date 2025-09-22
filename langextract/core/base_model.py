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

"""Base interfaces for language models."""
from __future__ import annotations

import abc
from collections.abc import Iterator, Sequence
import json
from typing import Any, Mapping

import yaml

from langextract.core import schema
from langextract.core import types

__all__ = ["BaseLanguageModel"]


class BaseLanguageModel(abc.ABC):
    """An abstract inference class for managing LLM inference.

    Attributes:
      _constraint: A `Constraint` object specifying constraints for model output.
    """

    def __init__(self, constraint: types.Constraint | None = None, **kwargs: Any):
        """Initializes the BaseLanguageModel with an optional constraint.

        Args:
          constraint: Applies constraints when decoding the output. Defaults to no
            constraint.
          **kwargs: Additional keyword arguments passed to the model.
        """
        self._constraint = constraint or types.Constraint()
        self._schema: schema.BaseSchema | None = None
        self._fence_output_override: bool | None = None
        self._extra_kwargs: dict[str, Any] = kwargs.copy()

    @classmethod
    def get_schema_class(cls) -> type[Any] | None:
        """Return the schema class this provider supports."""
        return None

    def apply_schema(self, schema_instance: schema.BaseSchema | None) -> None:
        """Apply a schema instance to this provider.

        Optional method that providers can override to store the schema instance
        for runtime use. The default implementation stores it as _schema.

        Args:
          schema_instance: The schema instance to apply, or None to clear.
        """
        self._schema = schema_instance

    @property
    def schema(self) -> schema.BaseSchema | None:
        """The current schema instance if one is configured.

        Returns:
          The schema instance or None if no schema is applied.
        """
        return self._schema

    def set_fence_output(self, fence_output: bool | None) -> None:
        """Set explicit fence output preference.

        Args:
          fence_output: True to force fences, False to disable, None for auto.
        """
        if not hasattr(self, "_fence_output_override"):
            self._fence_output_override = None
        self._fence_output_override = fence_output

    @property
    def requires_fence_output(self) -> bool:
        """Whether this model requires fence output for parsing.

        Uses explicit override if set, otherwise computes from schema.
        Returns True if no schema or schema doesn't require raw output.
        """
        if (
            hasattr(self, "_fence_output_override")
            and self._fence_output_override is not None
        ):
            return self._fence_output_override

        schema_obj = self.schema
        if schema_obj is None:
            return True
        return not schema_obj.requires_raw_output

    def merge_kwargs(
        self, runtime_kwargs: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Merge stored extra kwargs with runtime kwargs.

        Runtime kwargs take precedence over stored kwargs.

        Args:
          runtime_kwargs: Kwargs provided at inference time, or None.

        Returns:
          Merged kwargs dictionary.
        """
        base = getattr(self, "_extra_kwargs", {}) or {}
        incoming = dict(runtime_kwargs or {})
        return {**base, **incoming}

    @abc.abstractmethod
    def infer(
        self, batch_prompts: Sequence[str], **kwargs
    ) -> Iterator[Sequence[types.ScoredOutput]]:
        """Implements language model inference.

        Args:
          batch_prompts: Batch of inputs for inference. Single element list can be
            used for a single input.
          **kwargs: Additional arguments for inference, like temperature and
            max_decode_steps.

        Returns: Batch of Sequence of probable output text outputs, sorted by
          descending score.
        """

    def infer_batch(
        self,
        prompts: Sequence[str],
        batch_size: int = 32,  # pylint: disable=unused-argument
    ) -> list[list[types.ScoredOutput]]:
        """Batch inference with configurable batch size.

        This is a convenience method that collects all results from infer().

        Args:
          prompts: List of prompts to process.
          batch_size: Batch size (currently unused, for future optimization).

        Returns:
          List of lists of ScoredOutput objects.
        """
        results = []
        for output in self.infer(prompts):
            results.append(list(output))
        return results

    def parse_output(self, output: str) -> Any:
        """Parses model output as JSON or YAML.

        Note: This expects raw JSON/YAML without code fences.
        Code fence extraction is handled by resolver.py.

        Args:
          output: Raw output string from the model.

        Returns:
          Parsed Python object (dict or list).

        Raises:
          ValueError: If output cannot be parsed as JSON or YAML.
        """
        # Check if we have a format_type attribute (providers should set this)
        format_type = getattr(self, "format_type", types.FormatType.JSON)

        try:
            if format_type == types.FormatType.JSON:
                return json.loads(output)
            else:
                return yaml.safe_load(output)
        except Exception as e:
            raise ValueError(
                f"Failed to parse output as {format_type.name}: {str(e)}"
            ) from e
