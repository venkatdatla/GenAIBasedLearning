# SimpleGenAIApp

Eight small Python chat examples. Each one sends your question to a model, prints the answer, then waits for the next question. Type `stop` (or `quit`/`exit`) or press Enter on an empty line to end the program.

Each question is sent on its own. The examples do not keep prior chat messages, load stock data, or use external tools. The goal is to get a first model request working before adding memory, prompting techniques, agents, or workflows.

| API              | OpenAI SDK                   | Microsoft Agent Framework             | LangChain                          | LangGraph                       |
| ---------------- | ---------------------------- | ------------------------------------- | ---------------------------------- | ------------------------------- |
| Chat Completions | `openai_chat_completions.py` | `microsoft_agent_chat_completions.py` | `langchain_chat_completions.py`    | `langgraph_chat_completions.py` |
| Responses        | `openai_responses.py`        | `microsoft_agent_responses.py`        | `langchain_responses.py`           | `langgraph_responses.py`        |

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). From the workspace root in PowerShell:

```powershell
uv sync --project SimpleGenAIApp
Copy-Item SimpleGenAIApp/.env.example SimpleGenAIApp/.env
```

Edit `SimpleGenAIApp/.env` with your endpoint, deployment name, and credentials. Never commit `.env` or put real API keys in source code.

- The **OpenAI SDK**, **LangChain**, and **LangGraph** examples use `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and `AZURE_OPENAI_MODEL`. They can instead use OpenAI directly with `OPENAI_API_KEY` and `OPENAI_MODEL` when no Azure endpoint is configured.
- The **Microsoft Agent Framework Chat Completions** example uses the Azure OpenAI-compatible endpoint and API key.
- The **Microsoft Agent Framework Responses** example uses `FOUNDRY_PROJECT_ENDPOINT` and `FOUNDRY_MODEL`; run `az login` first to authenticate with `AzureCliCredential`.

## Run an example

From the workspace root, run one at a time:

```powershell
uv run --project SimpleGenAIApp python SimpleGenAIApp/openai_chat_completions.py
uv run --project SimpleGenAIApp python SimpleGenAIApp/openai_responses.py
uv run --project SimpleGenAIApp python SimpleGenAIApp/microsoft_agent_chat_completions.py
uv run --project SimpleGenAIApp python SimpleGenAIApp/microsoft_agent_responses.py
uv run --project SimpleGenAIApp python SimpleGenAIApp/langchain_chat_completions.py
uv run --project SimpleGenAIApp python SimpleGenAIApp/langchain_responses.py
uv run --project SimpleGenAIApp python SimpleGenAIApp/langgraph_chat_completions.py
uv run --project SimpleGenAIApp python SimpleGenAIApp/langgraph_responses.py
```

The examples make model requests and may incur API charges. `pyproject.toml` is the dependency source used by `uv`; a separate `requirements.txt` is not needed for this project.
