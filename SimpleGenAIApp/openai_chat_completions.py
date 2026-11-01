"""A simple interactive chat using the OpenAI Chat Completions API."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


def get_client() -> tuple[OpenAI, str]:
    """Create an OpenAI or Azure AI Foundry OpenAI-compatible client."""
    model = os.getenv("FOUNDRY_MODEL") or os.getenv("AZURE_OPENAI_MODEL") or os.getenv("OPENAI_MODEL")
    if not model:
        raise RuntimeError("Set FOUNDRY_MODEL (your Foundry deployment name) in .env.")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if endpoint:
        if not api_key:
            raise RuntimeError("Set AZURE_OPENAI_API_KEY for the configured endpoint.")
        base_url = endpoint.rstrip("/")
        if not base_url.endswith("/openai/v1"):
            base_url += "/openai/v1"
        return OpenAI(api_key=api_key, base_url=base_url + "/"), model
    if api_key:
        return OpenAI(api_key=api_key), model
    raise RuntimeError("Set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY, or OPENAI_API_KEY.")


def main() -> None:
    client, model = get_client()
    print("Chat started. Type 'stop' or press Enter to finish.")

    while True:
        question = input("\nYou: ").strip()
        if not question or question.casefold() in {"stop", "quit", "exit"}:
            break

        result = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
        )
        print(f"Assistant: {result.choices[0].message.content or '(empty response)'}")

    print("Conversation ended.")


if __name__ == "__main__":
    main()
