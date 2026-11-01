"""A simple interactive chat using LangGraph + Chat Completions."""

from __future__  import annotations
import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


class AnalysisState(TypedDict):
    question: str
    answer: str
    history: list[tuple[str, str]]


def build_model() -> ChatOpenAI:
    model = os.getenv("FOUNDRY_MODEL") or os.getenv("AZURE_OPENAI_MODEL")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    print(f"Model: {model}, Endpoint: {endpoint}, API Key: {'set' if api_key else 'not set'}")
    if not model or not endpoint or not api_key:
        raise RuntimeError("Configure FOUNDRY_MODEL, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_API_KEY in .env.")
    base_url = endpoint.rstrip("/")
    if not base_url.endswith("/openai/v1"):
        base_url += "/openai/v1"
    return ChatOpenAI(model=model, base_url=base_url + "/", api_key=api_key, temperature=0.2)


def main() -> None:
    model = build_model()

    def call_model(state: AnalysisState) -> dict[str, str]:
        print(f"Calling model with question: {state['question']}")
        messages = [SystemMessage(content="You are a helpful assistant. Answer clearly and use the conversation context.")]
        for previous_question, previous_answer in state["history"]:
            messages.append(HumanMessage(content=previous_question))
            messages.append(AIMessage(content=previous_answer))
        messages.append(HumanMessage(content=state["question"]))
        result = model.invoke(messages)
        print(f"Model response: {result.content}")
        answer = str(result.content)
        return {
            "answer": answer,
            "history": [*state["history"], (state["question"], answer)],
        }

    graph = StateGraph(AnalysisState)
    graph.add_node("answer_question", call_model)
    graph.add_edge(START, "answer_question")
    graph.add_edge("answer_question", END)
    app = graph.compile()

    print("Chat started. Type 'stop' or press Enter to finish.")
    history: list[tuple[str, str]] = []
    while True:
        question = input("\nYou: ").strip()
        if not question or question.casefold() in {"stop", "quit", "exit"}:
            break
        state = app.invoke({"question": question, "answer": "", "history": history})
        history = state["history"]
        print(f"Assistant: {state['answer']}")

    print("Conversation ended.")


if __name__ == "__main__":
    main()
