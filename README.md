# Agentic-RAG: Self-Corrective LLM Orchestration 🤖

This repository implements a **Self-Corrective Retrieval-Augmented Generation (CRAG)** system using a state-graph architecture. Unlike standard linear RAG pipelines, this agent employs a "Reasoning Loop" to evaluate the quality of retrieved information before generating a response.

## 🌟 Why this is Top 1% Engineering
Standard RAG systems often suffer from "hallucinations" when the retriever returns irrelevant data. This project solves that by implementing:
* **Stateful Orchestration:** Using `LangGraph` to manage complex agentic transitions.
* **Self-Grading Logic:** A dedicated node that acts as a "critic" to verify document relevance.
* **Recursive Feedback Loops:** If the data is insufficient, the agent triggers a query-rewrite cycle to attempt a better retrieval.

---

## 🏗️ Architecture Overview

The system operates as a **State Machine**:

1.  **Retrieve:** Fetches context from a knowledge base (simulated/local).
2.  **Grade:** An LLM-based grader evaluates the `(Question, Document)` pair.
3.  **Decide:** * If **Relevant** ✅ → Move to **Generation**.
    * If **Irrelevant** ❌ → Move to **Query Transformation** (Rewrite) and re-retrieve.
4.  **Generate:** Synthesizes the final response using only verified context.

---

## 🚀 Technical Stack
* **Framework:** `LangGraph` (for the Agentic State Machine)
* **Logic:** Python 3.11+
* **Design Pattern:** Corrective RAG (CRAG)
* **Environment:** local-first configuration

---

## 🛠️ How to Run

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/Balla5556/Self-Corrective-RAG-agent.git](https://github.com/Balla5556/Self-Corrective-RAG-agent.git)
   cd Self-Corrective-RAG-agent