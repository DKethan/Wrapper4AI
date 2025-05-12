from typing import List, Dict
import tiktoken
from ..config import Config
from ..base_handler import BaseHandler
import boto3
import json


class MetaLlamaHandler(BaseHandler):
    """Handler for Meta Llama models (typically via AWS Bedrock or self-hosted)"""

    def __init__(self, model: str = "meta.llama3-8b-instruct-v1"):
        super().__init__(model)
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=Config.get_region("meta"),
            aws_access_key_id=Config.get_api_key("meta"),
            aws_secret_access_key=Config.get_api_key("meta_secret")
        )
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def _generate_response(self, messages: List[Dict]) -> str:
        """Generate response using Llama's instruction format"""
        try:
            prompt = self._format_llama_messages(messages)
            body = {
                "prompt": prompt,
                "max_gen_len": 512,
                "temperature": 0.7
            }

            response = self.client.invoke_model(
                modelId=self.model,
                body=json.dumps(body)
            )

            return json.loads(response["body"].read())["generation"]
        except Exception as e:
            raise RuntimeError(f"Llama API error: {str(e)}")

    def _generate_title(self, prompt: str) -> str:
        try:
            response = self.client.invoke_model(
                modelId="meta.llama3-7b-instruct-v1",
                body=json.dumps({
                    "prompt": f"Generate 3-5 word title for: {prompt}",
                    "temperature": 0.3,
                    "max_gen_len": 20
                })
            )
            return json.loads(response["body"].read())["generation"].strip('"')
        except Exception as e:
            raise RuntimeError(f"Title generation failed: {str(e)}")

    def _format_llama_messages(self, messages: List[Dict]) -> str:
        """Convert messages to Llama's instruction format"""
        formatted = []
        for msg in messages:
            if msg["role"] == "system":
                formatted.append(f"<<SYS>>\n{msg['content']}\n<</SYS>>")
            elif msg["role"] == "user":
                formatted.append(f"[INST] {msg['content']} [/INST]")
            else:
                formatted.append(msg["content"])
        return "\n".join(formatted)

    def _truncate_history(self, messages: List[Dict]) -> List[Dict]:
        max_tokens = Config.get_max_history_tokens()
        trimmed = []
        total_tokens = 0

        for msg in reversed(messages):
            content = msg.get("content", "")
            tokens = len(self.encoder.encode(content)) + 3  # Add message overhead
            if total_tokens + tokens > max_tokens:
                break
            trimmed.insert(0, msg)
            total_tokens += tokens

        return trimmed

    def _count_tokens(self, messages: List[Dict]) -> int:
        """Accurate token counting"""
        tokens_per_message = 3
        return sum(
            len(self.encoder.encode(msg["content"])) + tokens_per_message
            for msg in messages
        )