import openai
import tiktoken
from typing import List, Dict, Optional
from ..config import Config
from ..base_handler import BaseHandler

class OpenAIHandler(BaseHandler):
    def __init__(self, model: str = "gpt-4o"):
        super().__init__(model)
        self.client = openai.Client(api_key=Config.get_api_key("openai", model))
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def _generate_response(self, messages: List[Dict]) -> str:
        """Generate response with history management"""
        try:
            truncated = self._truncate_history(messages)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=truncated,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Response generation failed: {str(e)}")

    def _generate_title(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Generate a 3-5 word title. Respond only with title."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content.strip('\"')
        except Exception as e:
            raise RuntimeError(f"Title generation failed: {str(e)}")

    def _truncate_history(self, messages: List[Dict]) -> List[Dict]:
        """Return a trimmed copy of messages to stay within token limit (keep latest context)"""
        max_tokens = Config.get_max_history_tokens()
        tokens_per_message = 3
        encoded = self.encoder

        trimmed = []
        total_tokens = 0

        # Traverse in reverse (most recent first)
        for message in reversed(messages):
            if not isinstance(message, dict) or "content" not in message:
                continue  # Skip malformed messages

            token_count = len(encoded.encode(message["content"])) + tokens_per_message
            if total_tokens + token_count > max_tokens:
                break
            trimmed.insert(0, message)
            total_tokens += token_count

        return trimmed

    def _count_tokens(self, messages: List[Dict]) -> int:
        """Accurate token counting"""
        tokens_per_message = 3
        return sum(
            len(self.encoder.encode(msg["content"])) + tokens_per_message
            for msg in messages
        )
