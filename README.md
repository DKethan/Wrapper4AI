# Wrapper4AI

**Multi-provider, pluggable LLM wrapper with token counting, history management, and seamless extensibility.**

> 🔧 Designed for developers building chat-based AI tools with OpenAI, Gemini, Bedrock, DeepSeek, Anthropic, and more.

---

## 🚀 Features

- 🔌 **Unified interface** for multiple LLM providers
- 💬 **Chat history tracking** with token trimming
- 🔢 **Token counting** with Tiktoken for OpenAI models
- 📚 **Extensible handler base** for adding new models
- ✅ **Testable client API** with clean abstraction

---

## 🛠️ Installation

```bash
  pip install git+https://github.com/DKethan/Wrap4AI.git
```

Or if you package it to PyPI:

```bash
  pip install wrapper4AI
```

---

## 🧩 Supported Providers

- ✅ OpenAI (`gpt-4o`, `gpt-3.5`, ... all OpenAI language models )
- ✅ Google Gemini -> yet to be released
- ✅ Amazon Bedrock -> yet to be released
- ✅ DeepSeek -> yet to be released
- ✅ Meta LLaMA -> yet to be released
- ✅ HuggingFace -> yet to be released
- ✅ Anthropic Claude -> yet to be released
- ✅ Perplexity AI -> yet to be released

---

## 🔧 Usage

### 1. Connect to a provider

```python
from wrapper4AI.wrap import connect

client = connect(provider="openai", model="gpt-4o", api_key="sk-xxx")
```

### 2. Basic Chat

```python
response = client.chat("Tell me a joke.")
print(response)
```

### 3. With History

```python
history = [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "A programming language."},
    {"role": "user", "content": "Who created it?"}
]
print(client.chat_with_history(history))
```


---

## 🧪 Testing

```bash
  python -m tests.test_openai_001
```
