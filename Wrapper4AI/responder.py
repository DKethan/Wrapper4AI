from typing import List, Dict, Optional
from .config import Config
from .handlers.openai_handler import OpenAIHandler
from .handlers.bedrock_handler import BedrockHandler
from .handlers.deepseek_handler import DeepSeekHandler
from .handlers.meta_llama_handler import MetaLlamaHandler
from .handlers.huggingface_handler import HuggingFaceHandler
from .handlers.anthropic_handler import AnthropicHandler
from .handlers.perplexity_ai_handler import PerplexityAIHandler
from .handlers.hf_inference_handler import HuggingFaceHandler as HuggingFaceInferenceHandler

class GenerativeResponder:
    """
    Handles interaction with model-specific handlers.
    Responsible for response generation, history management, and routing.
    """

    def __init__(self, model: str = None, provider: str = None):
        self.provider = provider or Config.get_default_provider()
        self.model = model or Config.get_default_model()
        self.handler = self._load_handler(self.provider, self.model)
        self.history: List[Dict] = []

    def _load_handler(self, provider: str, model: str):
        provider = provider.lower()

        if provider == "openai":
            return OpenAIHandler(model)
        elif provider == "deepseek":
            return DeepSeekHandler(model)
        elif provider == "bedrock":
            return BedrockHandler(model)
        elif provider == "meta":
            return MetaLlamaHandler(model)
        elif provider == "huggingface":
            return HuggingFaceHandler(model)
        elif provider == "huggingface_inference":
            return HuggingFaceInferenceHandler(model)
        elif provider == "anthropic":
            return AnthropicHandler(model)
        elif provider == "perplexity":
            return PerplexityAIHandler(model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate_response(
        self,
        messages: Optional[List[Dict]] = None,
        user_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a response from the model using the provided messages and/or user prompt.

        Args:
            messages: Chat history as a list of role-content dicts.
            user_prompt: A new user message to append to history.

        Returns:
            Generated model response as a string.

        Raises:
            RuntimeError: If response generation fails.
        """
        try:
            # Clone history or use provided messages
            combined_messages = list(messages) if messages is not None else list(self.history)

            # Add user prompt if provided
            if user_prompt:
                combined_messages.append({"role": "user", "content": user_prompt})

            if not combined_messages:
                raise ValueError("Message list is empty. Cannot generate response.")

            response = self.handler._generate_response(combined_messages)
            self.history.append({"role": "assistant", "content": response})
            return response
        except Exception as e:
            raise RuntimeError(f"Response generation failed: {str(e)}")

    def reset_history(self):
        """Clears conversation history."""
        self.history = []

    def print_history(self):
        """Prints the current conversation history."""
        for msg in self.history:
            print(f"{msg['role'].capitalize()}: {msg['content']}")

    def generate_title(self, prompt: str) -> str:
        """Generates a short title from a prompt."""
        return self.handler._generate_title(prompt)

    def truncate_history(self, messages: List[Dict]) -> List[Dict]:
        """Trims the message history to fit within token limits."""
        return self.handler._truncate_history(messages)

    def count_tokens(self, messages: List[Dict]) -> int:
        """Counts total tokens used by messages."""
        return self.handler._count_tokens(messages)
