"""A simple interactive chat using Microsoft Agent Framework + Foundry Responses."""

from __future__ import annotations

import asyncio
import os

from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

load_dotenv()


def make_agent() -> Agent:
    endpoint = os.getenv("FOUNDRY_PROJECT_ENDPOINT")
    model = os.getenv("FOUNDRY_MODEL") or os.getenv("AZURE_OPENAI_MODEL")
    if not endpoint or not model:
        raise RuntimeError("Configure FOUNDRY_PROJECT_ENDPOINT and FOUNDRY_MODEL in .env.")
    client = FoundryChatClient(
        project_endpoint=endpoint,
        model=model,
        credential=AzureCliCredential(),
    )
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
