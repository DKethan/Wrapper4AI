from typing import List, Dict
import anthropic
import tiktoken  # For approximate token counting
from ..config import Config
from ..base_handler import BaseHandler


class AnthropicHandler(BaseHandler):
    """Handler for Anthropic Claude models using the official SDK."""

    def __init__(self, model: str = "claude-3-opus-20240229"):
        super().__init__(model)
        self.client = anthropic.Anthropic(
            api_key=Config.get_api_key("anthropic")
        )
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def _generate_response(self, messages: List[Dict]) -> str:
        """Generate response using Claude's message API."""
        try:
            system_prompt, conversation = self._extract_system_message(messages)
            truncated = self._truncate_history(conversation)

            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=truncated,
                temperature=0.7,
                max_tokens=1024
            )
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {str(e)}")

    def _generate_title(self, prompt: str) -> str:
        """Generate short title using Haiku model."""
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=20,
                temperature=0.3,
                messages=[{
                    "role": "user",
                    "content": f"Generate a 3-5 word title for: {prompt}"
                }]
            )
            return response.content[0].text.strip('"')
        except Exception as e:
            raise RuntimeError(f"Title generation failed: {str(e)}")

    def _extract_system_message(self, messages: List[Dict]) -> (str, List[Dict]):
        """Separate system prompts from conversation history."""
        system_parts = []
        conversation = []

        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                conversation.append(msg)

        return ("\n".join(system_parts), conversation)

    def _truncate_history(self, messages: List[Dict]) -> List[Dict]:
        """Truncate conversation history while preserving message order."""
        max_tokens = Config.get_max_history_tokens()
        tokens_per_message = 3
        trimmed = []
        total_tokens = 0

        # Claude requires alternating user/assistant messages
        for msg in messages:
            content = msg.get("content", "")
            token_count = len(self.encoder.encode(content)) + tokens_per_message
            if total_tokens + token_count > max_tokens:
                break
            trimmed.append(msg)
            total_tokens += token_count

        return trimmed

    def _count_tokens(self, messages: List[Dict]) -> int:
        """Count approximate tokens using tiktoken."""
        tokens_per_message = 3
        return sum(
            len(self.encoder.encode(msg.get("content", ""))) + tokens_per_message
            for msg in messages
        )