from typing import List, Dict
import boto3
import json
import tiktoken
from ..config import Config
from ..base_handler import BaseHandler


class BedrockHandler(BaseHandler):
    """Handler for AWS Bedrock foundation models."""

    def __init__(self, model: str = "anthropic.claude-3-sonnet-20240229-v1:0"):
        super().__init__(model)
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=Config.get_region("aws"),
            aws_access_key_id=Config.get_api_key("aws"),
            aws_secret_access_key=Config.get_api_key("aws_secret")
        )
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def _generate_response(self, messages: List[Dict]) -> str:
        """Generate response using Bedrock's API with model-specific handling."""
        try:
            provider = self.model.split('.')[0]
            truncated_messages = self._truncate_history(messages)

            if provider == "anthropic":
                return self._handle_anthropic(truncated_messages)
            elif provider == "meta":
                return self._handle_llama(truncated_messages)
            elif provider == "amazon":
                return self._handle_titan(truncated_messages)
            else:
                raise RuntimeError(f"Unsupported Bedrock provider: {provider}")

        except Exception as e:
            raise RuntimeError(f"Bedrock API error: {str(e)}")

    def _generate_title(self, prompt: str) -> str:
        """Generate title using Claude Haiku."""
        try:
            response = self.client.invoke_model(
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                body=json.dumps({
                    "messages": [{
                        "role": "user",
                        "content": f"Generate a 3-5 word title for: {prompt}"
                    }],
                    "system": "Respond only with the title, no formatting.",
                    "max_tokens": 20,
                    "temperature": 0.3
                }),
                contentType="application/json"
            )
            result = json.loads(response["body"].read())
            return result["content"][0]["text"].strip('"')
        except Exception as e:
            raise RuntimeError(f"Title generation failed: {str(e)}")

    def _handle_anthropic(self, messages: List[Dict]) -> str:
        """Process Anthropic Claude models."""
        system_prompt, history = self._extract_system_message(messages)

        body = {
            "messages": history,
            "max_tokens": 1024,
            "temperature": 0.7,
            "system": system_prompt
        }

        response = self.client.invoke_model(
            modelId=self.model,
            body=json.dumps(body),
            contentType="application/json"
        )
        return json.loads(response["body"].read())["content"][0]["text"]

    def _handle_llama(self, messages: List[Dict]) -> str:
        """Process Meta Llama models."""
        formatted_messages = []
        for msg in messages:
            if msg["role"] == "system":
                formatted_messages.append(f"<<SYS>>\n{msg['content']}\n<</SYS>>")
            elif msg["role"] == "user":
                formatted_messages.append(f"[INST] {msg['content']} [/INST]")
            else:
                formatted_messages.append(msg["content"])

        body = {
            "prompt": "\n".join(formatted_messages),
            "max_gen_len": 1024,
            "temperature": 0.7
        }

        response = self.client.invoke_model(
            modelId=self.model,
            body=json.dumps(body),
            contentType="application/json"
        )
        return json.loads(response["body"].read())["generation"]

    def _handle_titan(self, messages: List[Dict]) -> str:
        """Process Amazon Titan models."""
        body = {
            "inputText": "\n".join(msg["content"] for msg in messages),
            "textGenerationConfig": {
                "maxTokenCount": 1024,
                "temperature": 0.7
            }
        }

        response = self.client.invoke_model(
            modelId=self.model,
            body=json.dumps(body),
            contentType="application/json"
        )
        return json.loads(response["body"].read())["results"][0]["outputText"]

    def _extract_system_message(self, messages: List[Dict]) -> (str, List[Dict]):
        """Extract and combine system messages."""
        system_messages = [msg["content"] for msg in messages if msg["role"] == "system"]
        history = [msg for msg in messages if msg["role"] != "system"]
        return ("\n".join(system_messages), history)

    def _truncate_history(self, messages: List[Dict]) -> List[Dict]:
        """Truncate messages to fit context window."""
        max_tokens = Config.get_max_history_tokens()
        tokens_per_message = 3
        total_tokens = 0
        truncated = []

        for msg in reversed(messages):
            content = msg.get("content", "")
            token_count = len(self.encoder.encode(content)) + tokens_per_message
            if total_tokens + token_count > max_tokens:
                break
            truncated.insert(0, msg)
            total_tokens += token_count

        return truncated

    def _count_tokens(self, messages: List[Dict]) -> int:
        """Calculate approximate token count."""
        return sum(
            len(self.encoder.encode(msg.get("content", ""))) + 3
            for msg in messages
        )