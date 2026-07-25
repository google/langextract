"""Provider implementation for matrixai."""

import os
import langextract as lx
from langextract_matrixai.schema import matrixaiSchema

from langextract.core.base_model import BaseLanguageModel
from langextract.core.types import ScoredOutput
from openai import OpenAI

@lx.providers.registry.register(
    r'^matrixai',
    r'^deepseek',  # Also register for deepseek model IDs
    priority=30  # Higher priority than Ollama (which has priority=10)
)
class matrixaiLanguageModel(BaseLanguageModel):
    """LangExtract provider for matrixai.

    This provider handles model IDs matching: ['^matrixai']
    """

    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        """Initialize the matrixai provider.

        Args:
            model_id: The model identifier.
            api_key: API key for authentication.
            **kwargs: Additional provider-specific parameters.
        """
        super().__init__()
        self.model_id = model_id
        
        # Debug: Print which method is used to get the API key
        if api_key:
            self.api_key = api_key
            print("DEBUG: Using API key passed as parameter")
        elif os.environ.get('MATRIXAI_API_KEY'):
            self.api_key = os.environ.get('MATRIXAI_API_KEY')
            print("DEBUG: Using MATRIXAI_API_KEY environment variable")
        elif os.environ.get('DEEPSEEK_API_KEY'):
            self.api_key = os.environ.get('DEEPSEEK_API_KEY')
            print("DEBUG: Using DEEPSEEK_API_KEY environment variable")
        else:
            raise ValueError(
                "API key is required for matrixai provider. "
                "Please set MATRIXAI_API_KEY or DEEPSEEK_API_KEY environment variable, "
                "or pass api_key parameter."
            )
        
        # Check if API key is available, raise informative error if not
        if not self.api_key:
            raise ValueError(
                "API key is required for matrixai provider. "
                "Please set MATRIXAI_API_KEY or DEEPSEEK_API_KEY environment variable, "
                "or pass api_key parameter."
            )
        
        self.response_schema = kwargs.get('response_schema')
        self.structured_output = kwargs.get('structured_output', True)
        self.base_url = os.environ.get("MATRIXAI_BASE_URL", "https://api.deepseek.com")
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        self._extra_kwargs = kwargs

    @classmethod
    def get_schema_class(cls):
        """Tell LangExtract about our schema support."""
        from langextract_matrixai.schema import matrixaiSchema
        return matrixaiSchema

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
                    {"role": "system", "content": "你是教研分析助手"}
                ],
                
                "stream": False,
            }
            
            completion = self.client.chat.completions.create(**api_params)  
            text = getattr(completion.choices[0].message, "content", "")
            
            if not text:
                raise RuntimeError("MatrixAI returned empty output")
            yield [ScoredOutput(score=1.0, output=text)]
    