"""Provider implementation for doubao."""

import os
import langextract as lx
from langextract_doubao.schema import doubaoSchema
from langextract.core.base_model import BaseLanguageModel
from langextract.core.types import ScoredOutput
from volcenginesdkarkruntime import Ark


@lx.providers.registry.register(r'^doubao', priority=10)
class doubaoLanguageModel(BaseLanguageModel):
    """LangExtract provider for doubao.

    This provider handles model IDs matching: ['^doubao']
    """

    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        """Initialize the doubao provider.

        Args:
            model_id: The model identifier.
            api_key: API key for authentication.
            **kwargs: Additional provider-specific parameters.
        """
        super().__init__()
        self.model_id = model_id
        self.api_key = api_key or os.environ.get('ARK_API_KEY')
        self.response_schema = kwargs.get('response_schema')
        self.structured_output = kwargs.get('structured_output', False)

        self.client = Ark(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=self.api_key
        )
        self._extra_kwargs = kwargs

    @classmethod
    def get_schema_class(cls):
        """Tell LangExtract about our schema support."""
        from langextract_doubao.schema import doubaoSchema
        return doubaoSchema

    def apply_schema(self, schema_instance):
        """Apply or clear schema configuration."""
        super().apply_schema(schema_instance)
        if schema_instance:
            config = schema_instance.to_provider_config()
            self.response_schema = config.get('response_schema')
            self.structured_output = config.get('structured_output', False)
        else:
            self.response_schema = None
            self.structured_output = False

    def infer(self, batch_prompts, **kwargs):
        """Run inference on a batch of prompts.

        Args:
            batch_prompts: List of prompts to process.
            **kwargs: Additional inference parameters.

        Yields:
            Lists of ScoredOutput objects, one per prompt.
        """
        for prompt in batch_prompts:
            api_params = {
                "model": self.model_id,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "system", "content": "You are an ai assistant"}
                ]
            }

            completion = self.client.chat.completions.create(**api_params)  
            text = getattr(completion.choices[0].message, "content", "")  
            # 调试：打印原始输出  
            print("[DEBUG] Doubao raw output:", repr(text))  
            if not text:  
                raise RuntimeError("Doubao returned empty output")  
            yield [ScoredOutput(score=1.0, output=text)]  
