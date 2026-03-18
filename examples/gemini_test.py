"""Simple Gemini test using environment variables from .env."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from wrapper4ai import connect


def main() -> None:
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY in .env")

    client = connect("google", "gemini-2.0-flash", api_key=api_key)
    response = client.chat("Say hello in one short line.")
    print("Gemini response:\n", response)


if __name__ == "__main__":
    main()
