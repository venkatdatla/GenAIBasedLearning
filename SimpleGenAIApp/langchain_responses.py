"""A simple interactive chat using LangChain + the OpenAI Responses API."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


def build_model() -> ChatOpenAI:
    """Create an OpenAI or Azure AI Foundry-compatible Responses model."""
    model = os.getenv("FOUNDRY_MODEL") or os.getenv("AZURE_OPENAI_MODEL") or os.getenv("OPENAI_MODEL")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not model:
        raise RuntimeError("Set FOUNDRY_MODEL, AZURE_OPENAI_MODEL, or OPENAI_MODEL in .env.")
    if endpoint:
        if not api_key:
            raise RuntimeError("Set AZURE_OPENAI_API_KEY for the configured endpoint.")
        base_url = endpoint.rstrip("/")
        if not base_url.endswith("/openai/v1"):
            base_url += "/openai/v1"
        return ChatOpenAI(
            model=model,
            base_url=base_url + "/",
            api_key=api_key,
            temperature=0.2,
            use_responses_api=True,
        )
    if api_key:
        return ChatOpenAI(model=model, api_key=api_key, temperature=0.2, use_responses_api=True)
    raise RuntimeError("Set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY, or OPENAI_API_KEY.")


def main() -> None:
    model = build_model()
    print("Chat started. Type 'stop' or press Enter to finish.")

    while True:
        question = input("\nYou: ").strip()
        if not question or question.casefold() in {"stop", "quit", "exit"}:
            break

        result = model.invoke(question)
        print(f"Assistant: {result.content or '(empty response)'}")

    print("Conversation ended.")


if __name__ == "__main__":
    main()
