"""A simple interactive chat using Microsoft Agent Framework + Chat Completions."""

from __future__ import annotations

import asyncio
import os

from agent_framework import Agent
from agent_framework.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()


def make_agent() -> Agent:
    model = os.getenv("FOUNDRY_MODEL") or os.getenv("AZURE_OPENAI_MODEL")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not model or not endpoint or not api_key:
        raise RuntimeError("Configure FOUNDRY_MODEL, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_API_KEY in .env.")
    base_url = endpoint.rstrip("/")
    if not base_url.endswith("/openai/v1"):
        base_url += "/openai/v1"
    client = OpenAIChatCompletionClient(model=model, base_url=base_url + "/", api_key=api_key)
    return Agent(
        client=client,
        name="SimpleChatAgent",
        instructions="You are a helpful assistant. Answer the user's question clearly and briefly.",
    )


async def main() -> None:
    agent = make_agent()
    print("Chat started. Type 'stop' or press Enter to finish.")
    try:
        while True:
            question = input("\nYou: ").strip()
            if not question or question.casefold() in {"stop", "quit", "exit"}:
                break
            result = await agent.run(question)
            print(f"Assistant: {result.text or '(empty response)'}")
    finally:
        await agent.client.close()
    print("Conversation ended.")


if __name__ == "__main__":
    asyncio.run(main())
