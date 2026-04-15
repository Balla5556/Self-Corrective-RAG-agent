# AI Project: Self-Corrective RAG Agent

This repository now contains one focused AI project only: a local self-corrective RAG agent built with `LangGraph`.

## What It Does

The agent takes a question and moves through an AI workflow:

1. Retrieve matching knowledge from a small local knowledge base.
2. Grade whether the retrieved documents are relevant.
3. Rewrite the query if the first retrieval is weak.
4. Generate a grounded answer or return a fallback response.

## Main File

- `agent_system.py` - the full AI workflow and command-line entry point

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 agent_system.py
```

Custom question:

```bash
python3 agent_system.py "What is Agentic RAG?"
```

## Notes

- This is a local AI demo project with no external API key required.
- The repository is intentionally kept minimal so it stays as one AI project only.
