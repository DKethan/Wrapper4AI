from typing import List, Dict
import google.generativeai as genai
import tiktoken  # For approximate token counting
from ..config import Config
from ..base_handler import BaseHandler


class GeminiHandler(BaseHandler):
    """Handler for Google Gemini models using the official Python SDK."""

    def __init__(self, model: str = "gemini-pro"):
        super().__init__(model)
        genai.configure(api_key=Config.get_api_key("google"))
        self.client = genai.GenerativeModel(model)
        self.encoder = tiktoken.get_encoding("cl100k_base")  # Approximate tokenizer

    def _generate_response(self, messages: List[Dict]) -> str:
        """Generate response using Gemini's chat API."""
        try:
            history = self._convert_to_gemini_format(messages)
            chat = self.client.start_chat(history=history[:-1])
            response = chat.send_message(
                content=history[-1]["parts"][0],
                safety_settings={
                    'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                    'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE'
                }
            )
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}")

    def _generate_title(self, prompt: str) -> str:
        """Generate short title using Gemini Flash model."""
        try:
            model = genai.GenerativeModel('gemini-flash')
            response = model.generate_content(
                f"Generate a 3-5 word title for: {prompt}",
                safety_settings={'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE'},
                generation_config={"temperature": 0.3}
            )
            return response.text.strip('"')
        except Exception as e:
            raise RuntimeError(f"Title generation failed: {str(e)}")

    def _convert_to_gemini_format(self, messages: List[Dict]) -> List[Dict]:
        """Convert standard messages to Gemini's format with role/parts structure."""
        converted = []
        for msg in messages:
            role = "user" if msg["role"] in ["user", "system"] else "model"
            converted.append({
                "role": role,
                "parts": [msg["content"]]
            })
        return converted

    def _truncate_history(self, messages: List[Dict]) -> List[Dict]:
        """Truncate conversation history to fit Gemini's token limit."""
        max_tokens = Config.get_max_history_tokens()
        tokens_per_message = 3
        trimmed = []
        total_tokens = 0

        for message in reversed(messages):
            content = message.get("content", "")
            token_count = len(self.encoder.encode(content)) + tokens_per_message
            if total_tokens + token_count > max_tokens:
                break
            trimmed.insert(0, message)
            total_tokens += token_count

        return trimmed

    def _count_tokens(self, messages: List[Dict]) -> int:
        """Count approximate tokens using tiktoken."""
        tokens_per_message = 3
        return sum(
            len(self.encoder.encode(msg.get("content", ""))) + tokens_per_message
            for msg in messages
        )