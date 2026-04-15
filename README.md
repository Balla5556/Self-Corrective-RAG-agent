# Self-Corrective RAG Agent

This project is a local-first demonstration of a self-corrective Retrieval-Augmented Generation workflow built with `LangGraph`.

Instead of using a single linear pipeline, the agent:

1. Retrieves candidate documents from a small knowledge base.
2. Grades whether the retrieved evidence is relevant.
3. Rewrites the query when the first search is weak.
4. Generates a grounded answer or stops with a fallback message.

## Project highlights

- Stateful orchestration with `LangGraph`
- Retrieval, grading, rewrite, and generation nodes
- Local knowledge base with no external API dependency
- Graceful fallback for unsupported questions
- Small `unittest` suite for direct-hit, rewrite, and fallback behavior

## Files

- `agent_system.py` - main LangGraph workflow and CLI entry point
- `tests/test_agent_system.py` - automated tests
- `requirements.txt` - Python dependencies

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the agent

Default demo:

```bash
python3 agent_system.py
```

Ask a custom question:

```bash
python3 agent_system.py "Why does it rewrite the question?"
```

## Run tests

```bash
python3 -m unittest discover -s tests -v
```

## Example behavior

Direct retrieval:

```bash
python3 agent_system.py "What is Agentic RAG?"
```

Rewrite path:

```bash
python3 agent_system.py "Why does it rewrite the question?"
```

Fallback path:

```bash
python3 agent_system.py "What are transformer attention heads?"
```

## GitHub note

The repository should not track `.venv` or `.DS_Store`. A `.gitignore` is included now so the repo stays clean and pushes more reliably.
