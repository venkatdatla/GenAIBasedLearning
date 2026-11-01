"""A simple interactive chat using LangGraph + the OpenAI Responses API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


class AnalysisState(TypedDict):
    system_message: str
    question: str
    response_id: str | None
    assistant_message: str


class AnalysisUpdate(TypedDict):
    response_id: str
    assistant_message: str


SYSTEM_MESSAGE = "You are a helpful assistant. Answer clearly and use the conversation context."


def build_model() -> ChatOpenAI:
    model = os.getenv("FOUNDRY_MODEL") or os.getenv("AZURE_OPENAI_MODEL")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not model or not endpoint or not api_key:
        raise RuntimeError("Configure FOUNDRY_MODEL, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_API_KEY in .env.")
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


def main() -> None:
    model = build_model()

    def call_model(state: AnalysisState) -> AnalysisUpdate:
        messages = [
            SystemMessage(content=state["system_message"]),
            HumanMessage(content=state["question"]),
        ]
        if state["response_id"] is None:
            result = model.invoke(messages)
        else:
            result = model.invoke(
                messages,
                previous_response_id=state["response_id"],
            )
        response_id = result.response_metadata.get("id")
        if not isinstance(response_id, str) or not response_id:
            raise RuntimeError("The Responses API did not return a response ID.")
        return {
            "assistant_message": str(result.content),
            "response_id": response_id,
        }

    graph = StateGraph(AnalysisState)
    graph.add_node("answer_question", call_model)
    graph.add_edge(START, "answer_question")
    graph.add_edge("answer_question", END)
    app = graph.compile()

    print("Chat started. Type 'stop' or press Enter to finish.")
    response_id: str | None = None
    while True:
        question = input("\nYou: ").strip()
        if not question or question.casefold() in {"stop", "quit", "exit"}:
            break
        state = app.invoke(
            {
                "system_message": SYSTEM_MESSAGE,
                "question": question,
                "response_id": response_id,
                "assistant_message": "",
            }
        )
        response_id = state["response_id"]
        print(f"Assistant: {state['assistant_message']}")

    print("Conversation ended.")


if __name__ == "__main__":
    main()
