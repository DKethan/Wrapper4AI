# Wrapper4AI

**Multi-provider, pluggable LLM wrapper with token counting, history management, streaming, and seamless extensibility.**

Unified Python client for OpenAI, Anthropic, Google Gemini, Mistral AI, DeepSeek, AWS Bedrock, Hugging Face, and Perplexity AI.

---

## Features

- **Unified interface** for multiple LLM providers
- **Chat history** with optional tracking and token trimming
- **Streaming** support for all providers
- **Token counting** (tiktoken when available)
- **Provider registry** — register custom backends via `BaseProvider`
- **Config from env** — API keys and regions from environment variables

---

## Installation

```bash
pip install wrapper4ai
```

Install with optional dependencies for the providers you use:

```bash
# One provider
pip install wrapper4ai[openai]

# Multiple
pip install wrapper4ai[openai,anthropic,google,mistral]

# All providers
pip install wrapper4ai[all]
```

Optional extras: `openai`, `anthropic`, `google`, `mistral`, `bedrock`, `huggingface`, `all`, `dev`.

---

## Supported Providers

| Provider     | Default model              | Optional extra |
|-------------|----------------------------|----------------|
| OpenAI      | `gpt-4o`                   | `openai`       |
| Anthropic   | `claude-sonnet-4-20250514`  | `anthropic`    |
| Google      | `gemini-2.0-flash`         | `google`       |
| Mistral AI  | `mistral-large-latest`      | `mistral`      |
| DeepSeek    | `deepseek-chat`            | —              |
| AWS Bedrock | Claude / Llama / Titan     | `bedrock`      |
| Hugging Face| `HuggingFaceH4/zephyr-7b-beta` | `huggingface` |
| Perplexity  | `sonar-pro`                 | —              |

---

## Usage

### Connect and chat

```python
from wrapper4ai import connect

client = connect("openai", "gpt-4o", api_key="sk-...")
response = client.chat("Tell me a joke.")
print(response)
```

API keys can be set via environment variables (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `MISTRAL_API_KEY`) or passed as `api_key=`.

### Streaming

```python
client = connect("anthropic", "claude-sonnet-4-20250514")
for chunk in client.stream("Write a short poem."):
    print(chunk, end="")
```

### With history

History is kept by default; you can clear it or turn it off per call:

```python
client = connect("openai", "gpt-4o")
client.chat("My name is Alex.")
client.chat("What is my name?")  # uses history
client.chat("Hello!", use_history=False)  # no history update
client.clear_history()
```

### Stateless completion

```python
messages = [
    {"role": "user", "content": "What is 2+2?"},
]
reply = client.complete(messages)  # does not change client history
```

### System prompt and token counting

```python
client = connect("openai", "gpt-4o", system_prompt="You are a helpful assistant.")
client.chat("Hi!")
n = client.count_tokens("Some text")
n = client.count_tokens(client.history)
total = client.total_tokens_used
title = client.generate_title("A long discussion about AI")
```

### List providers and custom backends

```python
from wrapper4ai import list_providers, connect
from wrapper4ai.providers import BaseProvider, register_provider

print(list_providers())  # ['anthropic', 'bedrock', 'deepseek', 'google', ...]

class MyProvider(BaseProvider):
    def generate(self, messages): ...
    def stream(self, messages): ...
    def count_tokens(self, messages): ...

register_provider("my_backend", MyProvider)
client = connect("my_backend", "my-model")
```

---

## Testing

```bash
pip install -e ".[dev]"
PYTHONPATH=. pytest tests/ -v
```

Tests use a mock provider; no API keys required for the test suite.

---

## License

MIT.
