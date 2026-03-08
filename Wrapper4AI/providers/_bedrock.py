"""AWS Bedrock provider (Anthropic, Meta Llama, Amazon Titan) via boto3."""

from __future__ import annotations

from collections.abc import Iterator
import json

from wrapper4ai.config import DEFAULTS, resolve_api_key, resolve_region, resolve_secret_key
from wrapper4ai.exceptions import AuthenticationError, MissingDependencyError, APIError
from wrapper4ai.providers import BaseProvider
from wrapper4ai._utils import truncate_messages


def _extract_system(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    system_parts: list[str] = []
    rest: list[dict[str, str]] = []
    for m in messages:
        if m.get("role") == "system":
            system_parts.append(m.get("content") or "")
        else:
            rest.append(m)
    return "\n".join(system_parts), rest


class BedrockProvider(BaseProvider):
    """AWS Bedrock foundation models (Claude, Llama, Titan)."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(model, api_key=api_key, **kwargs)
        try:
            import boto3
        except ImportError as e:
            raise MissingDependencyError("bedrock", "boto3") from e
        region = resolve_region(
            "bedrock",
            explicit_region=kwargs.get("region") if isinstance(kwargs.get("region"), str) else None,
        )
        access = resolve_api_key("bedrock", explicit_key=api_key)
        secret = resolve_secret_key(
            "bedrock",
            explicit_key=kwargs.get("secret_key") if isinstance(kwargs.get("secret_key"), str) else None,
        )
        if not access or not secret:
            raise AuthenticationError(
                "AWS credentials required (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)",
                provider="bedrock",
            )
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            aws_access_key_id=access,
            aws_secret_access_key=secret,
        )
        self._max_history = int(DEFAULTS.get("max_history_tokens", 8000))

    def _provider_prefix(self) -> str:
        return self.model.split(".")[0].lower()

    def generate(self, messages: list[dict[str, str]]) -> str:
        truncated = truncate_messages(
            messages,
            self._max_history,
            preserve_system=True,
        )
        prefix = self._provider_prefix()
        if prefix == "anthropic":
            return self._generate_anthropic(truncated)
        if prefix == "meta":
            return self._generate_llama(truncated)
        if prefix == "amazon":
            return self._generate_titan(truncated)
        raise APIError(f"Unsupported Bedrock model prefix: {prefix}")

    def _generate_anthropic(self, messages: list[dict[str, str]]) -> str:
        system, rest = _extract_system(messages)
        max_tokens = int(self._kwargs.get("max_tokens", DEFAULTS["max_tokens"]))
        temperature = float(self._kwargs.get("temperature", DEFAULTS["temperature"]))
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": rest,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            body["system"] = system
        r = self._client.invoke_model(
            modelId=self.model,
            body=json.dumps(body),
            contentType="application/json",
        )
        data = json.loads(r["body"].read())
        content = data.get("content")
        if not content or not content[0].get("text"):
            raise APIError("Empty Bedrock response")
        return content[0]["text"].strip()

    def _generate_llama(self, messages: list[dict[str, str]]) -> str:
        parts: list[str] = []
        for m in messages:
            role = m.get("role") or "user"
            content = m.get("content") or ""
            if role == "system":
                parts.append(f"<<SYS>>\n{content}\n<</SYS>>")
            elif role == "user":
                parts.append(f"[INST] {content} [/INST]")
            else:
                parts.append(content)
        prompt = "\n".join(parts)
        max_tokens = int(self._kwargs.get("max_tokens", DEFAULTS["max_tokens"]))
        temperature = float(self._kwargs.get("temperature", DEFAULTS["temperature"]))
        body = {
            "prompt": prompt,
            "max_gen_len": max_tokens,
            "temperature": temperature,
        }
        r = self._client.invoke_model(
            modelId=self.model,
            body=json.dumps(body),
            contentType="application/json",
        )
        data = json.loads(r["body"].read())
        return (data.get("generation") or "").strip()

    def _generate_titan(self, messages: list[dict[str, str]]) -> str:
        text = "\n".join(m.get("content") or "" for m in messages)
        max_tokens = int(self._kwargs.get("max_tokens", DEFAULTS["max_tokens"]))
        temperature = float(self._kwargs.get("temperature", DEFAULTS["temperature"]))
        body = {
            "inputText": text,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": temperature,
            },
        }
        r = self._client.invoke_model(
            modelId=self.model,
            body=json.dumps(body),
            contentType="application/json",
        )
        data = json.loads(r["body"].read())
        results = data.get("results")
        if not results or not results[0].get("outputText"):
            raise APIError("Empty Bedrock Titan response")
        return results[0]["outputText"].strip()

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        # Bedrock streaming is model-specific; yield full response as one chunk for simplicity
        full = self.generate(messages)
        if full:
            yield full

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        return sum(len(m.get("content", "") or "") // 4 + 3 for m in messages)
