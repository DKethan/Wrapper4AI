from typing import List, Dict
from huggingface_hub import InferenceClient
from ..config import Config
from ..base_handler import BaseHandler


class HuggingFaceHandler(BaseHandler):
    """Handler for Hugging Face models using InferenceClient."""

    def __init__(self, model: str = "HuggingFaceH4/zephyr-7b-beta"):
        super().__init__(model)
        self.client = InferenceClient(
            model=model,
            token=Config.get_api_key("huggingface", model)
        )
        self.max_tokens = Config.get_max_history_tokens()

    def _generate_response(self, messages: List[Dict]) -> str:
        """Generate only the first assistant response."""
        try:
            truncated = self._truncate_history(messages)

            # Build prompt ending with a user message (so assistant responds once)
            prompt = ""
            for msg in truncated:
                if msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n"
            if truncated[-1]["role"] == "user":
                prompt += "Assistant:"

            # Get model output
            output = self.client.text_generation(
                prompt, max_new_tokens=256, temperature=0.7
            ).strip()

            # Return content until next "User:" or second "Assistant:"
            if "Assistant:" in output:
                response = output.split("Assistant:")[1]
                if "User:" in response:
                    return response.split("User:")[0].strip()
                return response.strip()
            return output
        except Exception as e:
            raise RuntimeError(f"Response generation failed: {str(e)}")

    def _generate_title(self, prompt: str) -> str:
        """Generate a 3–5 word title from a prompt using clear instruction."""
        try:
            instruction = "Generate a short 3-5 word catchy title for this message. Reply only with the title."
            full_prompt = f"<|user|>\n{instruction}\n{prompt}\n<|assistant|>\n"
            response = self.client.text_generation(full_prompt, max_new_tokens=10, temperature=0.3)
            return response.strip().strip('"').split("\n")[0]
        except Exception as e:
            raise RuntimeError(f"Title generation failed: {str(e)}")

    def _truncate_history(self, messages: List[Dict]) -> List[Dict]:
        """Trim messages to stay within a token budget."""
        trimmed = []
        total_tokens = 0

        for msg in reversed(messages):
            if not isinstance(msg, dict) or "content" not in msg:
                continue
            content = msg["content"]
            token_estimate = len(content) // 4  # Approx. 4 chars per token
            if total_tokens + token_estimate > self.max_tokens:
                break
            trimmed.insert(0, msg)
            total_tokens += token_estimate

        return trimmed

    def _count_tokens(self, messages: List[Dict]) -> int:
        """Rough token count for a message list."""
        return sum(len(msg.get("content", "")) // 4 for msg in messages)

    def _build_prompt(self, messages: List[Dict]) -> str:
        """Build prompt using Zephyr-style chat formatting with <|user|> and <|assistant|>."""
        prompt = ""
        for msg in messages:
            if msg["role"] == "user":
                prompt += f"<|user|>\n{msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"<|assistant|>\n{msg['content']}\n"
        if messages[-1]["role"] == "user":
            prompt += "<|assistant|>\n"
        return prompt

