# Examples

Provider test scripts using `.env`.

## Setup

```bash
pip install -e .
pip install python-dotenv
```

Create env file:

```bash
cp examples/.env.example .env
```

## Run tests

```bash
python examples/openai_test.py
python examples/gemini_test.py
```
# Examples

This folder contains runnable usage samples for `wrapper4ai`.

## UI4AI-style chat example

Run:

```bash
python examples/ui4ai_openai_chat.py
```

File:
- `examples/ui4ai_openai_chat.py` - Connects `wrapper4ai` to `UI4AI.run_chat`.
