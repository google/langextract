"""Provider implementation for vLLM (Direct Library Integration Only)."""

from typing import List, Generator, Dict, Any

import langextract as lx
from langextract.core import base_model, exceptions, types

VLLM_PATTERNS = (
    r"^vllm:",
)


@lx.providers.registry.register(*VLLM_PATTERNS, priority=20)
class VLLMLanguageModel(base_model.BaseLanguageModel):
    """LangExtract provider for direct vLLM library integration.

    This provider handles model IDs matching: ['^vllm:']
    Uses vLLM library directly for optimal performance.
    """

    def __init__(
        self,
        model_id: str,
        max_workers: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        gpu_memory_utilization: float = 0.7,  # 进一步降低GPU内存利用率
        max_model_len: int = 1024,  # 进一步减少最大序列长度
        **kwargs,
    ):
        """Initialize the direct vLLM provider.

        Args:
            model_id: Model identifier in format "vllm:model_name_or_path"
            max_workers: Maximum parallel workers (affects batch size)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            gpu_memory_utilization: GPU memory utilization ratio (0.0-1.0)
            max_model_len: Maximum sequence length (reduces KV cache memory)
            **kwargs: Additional vLLM initialization parameters
        """
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise exceptions.InferenceConfigError(
                "vLLM library not installed. Please install with: pip install vllm"
            )

        super().__init__()

        self.model_id = model_id
        self.max_workers = max_workers
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len

        # Extract actual model name from identifier
        if model_id.startswith("vllm:"):
            actual_model_id = model_id.split(":", 1)[1]
        else:
            actual_model_id = model_id

        # Separate vLLM engine kwargs from sampling kwargs
        engine_kwargs = {
            k: v for k, v in kwargs.items() 
            if k not in ['temperature', 'top_p', 'max_tokens', 'gpu_memory_utilization', 'max_model_len']
        }

        # Initialize vLLM engine with memory control
        # Disable torch.compile to avoid PY_SSIZE_T_CLEAN error
        engine_kwargs.setdefault('enforce_eager', True)
        engine_kwargs.setdefault('disable_custom_all_reduce', True)
        
        self._vllm_engine = LLM(
            model=actual_model_id,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            **engine_kwargs
        )

        # Configure sampling parameters
        self._sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    def _process_batch(self, prompts: List[str]) -> List[types.ScoredOutput]:
        """Process batch using direct vLLM engine."""
        try:
            outputs = self._vllm_engine.generate(
                prompts,
                self._sampling_params,
                use_tqdm=False
            )

            results = []
            for output in outputs:
                result = output.outputs[0].text
                results.append(types.ScoredOutput(score=1.0, output=result))

            return results

        except Exception as e:
            raise exceptions.InferenceRuntimeError(
                f"vLLM engine error: {str(e)}", original=e
            ) from e

    def infer(self, batch_prompts: List[str], **kwargs) -> Generator[List[types.ScoredOutput], None, None]:
        """Run inference using direct vLLM integration.

        Args:
            batch_prompts: List of prompts to process.
            **kwargs: Additional sampling parameters (temperature, top_p, max_tokens).

        Yields:
            Lists of ScoredOutput objects, one per batch.
        """
        # Update sampling parameters if provided
        sampling_params = self._sampling_params
        if kwargs:
            sampling_kwargs = {
                k: kwargs[k] for k in ['temperature', 'top_p', 'max_tokens'] 
                if k in kwargs
            }
            if sampling_kwargs:
                # Create new sampling params with updated values
                current_params = {
                    'temperature': sampling_params.temperature,
                    'top_p': sampling_params.top_p,
                    'max_tokens': sampling_params.max_tokens,
                }
                current_params.update(sampling_kwargs)
                
                from vllm import SamplingParams
                sampling_params = SamplingParams(**current_params)

        # Store original sampling params and update for this inference
        original_params = self._sampling_params
        self._sampling_params = sampling_params

        try:
            # Process in batches for better memory management
            # Adjust batch size based on max_workers and available resources
            batch_size = max(1, self.max_workers * 4)

            for i in range(0, len(batch_prompts), batch_size):
                batch_chunk = batch_prompts[i:i + batch_size]
                chunk_results = self._process_batch(batch_chunk)
                yield chunk_results

        finally:
            # Restore original sampling parameters
            self._sampling_params = original_params

    @classmethod
    def get_supported_models(cls) -> List[Dict[str, Any]]:
        """Get list of supported models and their capabilities."""
        return [
            {
                "id": "vllm:*",
                "name": "vLLM Direct",
                "description": "Direct vLLM library integration with PagedAttention support",
                "capabilities": ["chat", "completion", "batch_processing"],
                "max_tokens": 131072,  # 128k context window (model dependent)
                "supports_parallel": True,
                "supports_batching": True,
            }
        ]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        return {
            "model_id": self.model_id,
            "max_workers": self.max_workers,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "sampling_params": {
                "temperature": self._sampling_params.temperature,
                "top_p": self._sampling_params.top_p,
                "max_tokens": self._sampling_params.max_tokens,
            },
            "provider_type": "vllm_direct",
        }