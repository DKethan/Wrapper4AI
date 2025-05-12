from typing import List, Dict
import requests
import tiktoken
from ..config import Config
from ..base_handler import BaseHandler


class PerplexityAIHandler(BaseHandler):
    """Handler for Perplexity AI using their native API."""

    API_BASE = "https://api.perplexity.ai"

    def __init__(self, model: str = "pplx-7b-online"):
        super().__init__(model)
        self.api_key = Config.get_api_key("perplexity")
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def _generate_response(self, messages: List[Dict]) -> str:
        """Generate response using Perplexity's API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "messages": self._truncate_history(messages),
                "temperature": 0.7,
                "max_tokens": 1024
            }

            response = requests.post(
                f"{self.API_BASE}/chat/completions",
                headers=headers,
                json=payload
            ).json()

            return response['choices'][0]['message']['content']
        except Exception as e:
            raise RuntimeError(f"Perplexity API request failed: {str(e)}")

    def _generate_title(self, prompt: str) -> str:
        """Generate short title using faster model."""
        try:
            response = self.client.chat.completions.create(
                model="sonar-small-chat",
                messages=[
                    {"role": "system", "content": "Generate 3-5 word title. Only respond with title."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=20
            )
            return response.choices[0].message.content.strip('\"')
        except Exception as e:
            raise RuntimeError(f"Title generation failed: {str(e)}")

    def _truncate_history(self, messages: List[Dict]) -> List[Dict]:
        """Truncate conversation history to fit token limit."""
        max_tokens = Config.get_max_history_tokens()
        tokens_per_message = 3
        trimmed = []
        total_tokens = 0

        for message in reversed(messages):
            content = message.get("content", "")
            role = message.get("role", "")

            token_count = len(self.encoder.encode(content)) + tokens_per_message
            if total_tokens + token_count > max_tokens:
                break
            trimmed.insert(0, {"role": role, "content": content})
            total_tokens += token_count

        return trimmed

    def _count_tokens(self, messages: List[Dict]) -> int:
        """Count tokens for accurate usage tracking."""
        tokens_per_message = 3
        return sum(
            len(self.encoder.encode(msg.get("content", ""))) + tokens_per_message
            for msg in messages
        )