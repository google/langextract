"""Provider implementation for vLLM"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Generator, List, Optional, Union

import langextract as lx
from langextract.core import base_model, exceptions, types

logger = logging.getLogger(__name__)

VLLM_PATTERNS = (r"^vllm:",)

# Lazy import to avoid hard dependency at import time
try:
    from vllm import LLM, SamplingParams  # type: ignore
except Exception:
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore


@lx.providers.registry.register(*VLLM_PATTERNS, priority=20)
class VLLMLanguageModel(base_model.BaseLanguageModel):
    """Minimal direct vLLM integration.

    Features:
      - Plain string or messages input
      - True batching (single vLLM generate call)
      - SamplingParams passthrough/override
      - Auto/custom stop tokens
      - Minimal JSON cleaning (opt-in via return_json=True)
    """

    def __init__(
        self,
        model_id: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        gpu_memory_utilization: float = 0.7,
        max_model_len: int = 1024,
        **kwargs: Any,
    ):
        if LLM is None or SamplingParams is None:
            raise exceptions.InferenceConfigError(
                "vLLM is not installed. Please run: pip install vllm"
            )

        super().__init__()

        self.model_id = model_id
        self.system_prompt = system_prompt

        # Normalize model id ("vllm:foo/bar" -> "foo/bar")
        actual_model_id = model_id.split(":", 1)[1] if model_id.startswith("vllm:") else model_id

        # Split kwargs into engine kwargs vs SamplingParams overrides
        sampling_keys = self._sampling_param_names()
        engine_block = {"temperature", "top_p", "max_tokens", "gpu_memory_utilization", "max_model_len", "stop"}
        sampling_overrides = {k: v for k, v in kwargs.items() if k in sampling_keys}
        engine_kwargs = {k: v for k, v in kwargs.items() if k not in sampling_keys and k not in engine_block}

        # Safe engine defaults (can be overridden)
        engine_kwargs.setdefault("enforce_eager", True)
        engine_kwargs.setdefault("disable_custom_all_reduce", True)

        # Initialize engine
        self._engine = LLM(
            model=actual_model_id,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            **engine_kwargs,
        )

        # Tokenizer (for templates/stop detection)
        self.tokenizer = self._engine.get_tokenizer()
        if self.tokenizer is None:
            raise exceptions.InferenceConfigError("Failed to obtain tokenizer from vLLM engine.")
        self.use_chat_template = bool(getattr(self.tokenizer, "chat_template", None))

        # Stop detection (custom > auto > None)
        stop_tokens = self._auto_stop(kwargs.get("stop"))

        # Build initial SamplingParams (defaults + overrides)
        base_sampling = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stop": stop_tokens,          # None => let engine/model handle EOS
            "skip_special_tokens": True,
        }
        base_sampling.update(sampling_overrides)
        self._sampling = self._build_sampling(base_sampling)

    # -------------------------
    # Internal helpers
    # -------------------------

    def _sampling_param_names(self) -> set[str]:
        """Discover valid SamplingParams fields (fallback to common set)."""
        try:
            code = SamplingParams.__init__.__code__  # type: ignore[attr-defined]
            names = set(code.co_varnames[: code.co_argcount])
            names.discard("self")
            return names
        except Exception:
            return {
                "temperature", "top_p", "top_k", "min_p", "top_a",
                "max_tokens",
                "stop", "stop_token_ids",
                "presence_penalty", "frequency_penalty", "repetition_penalty", "length_penalty",
                "n", "best_of",
                "use_beam_search", "beam_width",
                "seed", "logprobs",
                "ignore_eos", "skip_special_tokens", "include_stop_str_in_output",
            }

    def _build_sampling(self, params: Dict[str, Any]) -> SamplingParams:
        """Filter unknown keys and build SamplingParams safely."""
        valid = self._sampling_param_names()
        safe = {k: v for k, v in params.items() if k in valid}
        try:
            return SamplingParams(**safe)
        except Exception as e:
            raise exceptions.InferenceConfigError(f"Invalid SamplingParams: {type(e).__name__}: {e}") from e

    def _auto_stop(self, custom: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
        """Return custom stops if provided; otherwise only high-confidence stops."""
        if custom:
            if isinstance(custom, str):
                custom = [custom]
            custom = [str(s) for s in custom if s]
            # dedup keep order
            seen, out = set(), []
            for s in custom:
                if s not in seen:
                    seen.add(s)
                    out.append(s)
            return out or None

        stops: List[str] = []
        eos = getattr(self.tokenizer, "eos_token", None)
        if eos:
            stops.append(str(eos))
        if self.use_chat_template:
            tmpl = getattr(self.tokenizer, "chat_template", "") or ""
            for mark in ("<|eot_id|>", "<|im_end|>", "</s>", "<|endoftext|>"):
                if mark in tmpl:
                    stops.append(mark)
        pad = getattr(self.tokenizer, "pad_token", None)
        stops = [s for s in stops if s and s != pad]
        # dedup keep order
        seen, out = set(), []
        for s in stops:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out or None

    def _apply_template(self, messages: List[Dict[str, str]]) -> str:
        """Apply tokenizer chat template when available; fallback to stitched transcript."""
        if self.use_chat_template:
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                logger.warning("Chat template failed: %r", e)
        # Fallback (plain stitched dialogue)
        buf = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
        buf.append("assistant:")
        return "\n".join(buf)

    def _format_one(self, item: Union[str, List[Dict[str, str]]]) -> str:
        """Format one input (string or messages) into a ready prompt."""
        if isinstance(item, str):
            if not self.use_chat_template:
                return item
            msgs = []
            if self.system_prompt:
                msgs.append({"role": "system", "content": self.system_prompt})
            msgs.append({"role": "user", "content": item})
            return self._apply_template(msgs)

        # messages list
        msgs = list(item)
        if self.system_prompt and not any(m.get("role") == "system" for m in msgs):
            msgs.insert(0, {"role": "system", "content": self.system_prompt})
        return self._apply_template(msgs)

    def _clean_json_min(self, text: str) -> str:
        """Minimal JSON fix (only when return_json=True)."""
        t = text.strip()
        # 1) Already valid
        try:
            json.loads(t)
            return t
        except Exception:
            pass
        # 2) Close unmatched top-level '{' braces
        if t.startswith("{") and not t.endswith("}"):
            open_n, close_n = t.count("{"), t.count("}")
            if open_n > close_n:
                t2 = t + ("}" * (open_n - close_n))
                try:
                    json.loads(t2)
                    return t2
                except Exception:
                    return t
        # 3) Remove trailing commas before '}' or ']'
        if t.startswith("{") and t.endswith("}"):
            lines = t.split("\n")
            cleaned = []
            for i, line in enumerate(lines):
                s = line.rstrip()
                if i < len(lines) - 1:
                    nxt = lines[i + 1].lstrip()
                    if s.endswith(",") and (nxt.startswith("}") or nxt.startswith("]")):
                        s = s[:-1]
                cleaned.append(s)
            t2 = "\n".join(cleaned)
            try:
                json.loads(t2)
                return t2
            except Exception:
                return t
        return t

    def _generate_batch(self, prompts: List[str], sampling: SamplingParams) -> List[str]:
        """Single batched call to vLLM; returns first candidate per input."""
        outputs = self._engine.generate(prompts, sampling, use_tqdm=False)
        results: List[str] = []
        for out in outputs:
            if not out.outputs:
                results.append("")
            else:
                results.append(out.outputs[0].text)
        return results

    # -------------------------
    # Public API
    # -------------------------

    def infer(
        self,
        batch_prompts: List[Union[str, List[Dict[str, str]]]],
        **kwargs: Any,
    ) -> Generator[List[types.ScoredOutput], None, None]:
        """Run batched inference with vLLM.

        Args:
            batch_prompts: Each item is either a plain string or a list of messages
                           like {'role': 'user|system|assistant', 'content': '...'}.
            **kwargs: SamplingParams overrides (e.g., temperature/top_p/max_tokens/seed/stop/...).
                      Extra flag:
                        - return_json: bool = False -> apply minimal JSON cleaning if True.

        Yields:
            For each input, yields a list with a single ScoredOutput.
        """
        # Merge per-call SamplingParams overrides
        if kwargs:
            valid = self._sampling_param_names()
            ov = {k: v for k, v in kwargs.items() if k in valid}
            if ov:
                cur = {k: getattr(self._sampling, k, None) for k in valid}
                cur.update(ov)
                cur.setdefault("stop", getattr(self._sampling, "stop", None))
                cur.setdefault("skip_special_tokens", getattr(self._sampling, "skip_special_tokens", True))
                self._sampling = self._build_sampling(cur)

        do_json = bool(kwargs.get("return_json", False))

        # Format inputs
        try:
            formatted = [self._format_one(p) for p in batch_prompts]
        except Exception as e:
            raise exceptions.InferenceConfigError(f"Failed to format prompts: {type(e).__name__}: {e}") from e

        # Generate
        try:
            texts = self._generate_batch(formatted, self._sampling)
        except Exception as e:
            raise exceptions.InferenceRuntimeError(f"vLLM engine error: {type(e).__name__}: {e}", original=e) from e

        # Yield results
        for t in texts:
            out = self._clean_json_min(t) if do_json else t
            yield [types.ScoredOutput(score=1.0, output=out)]
