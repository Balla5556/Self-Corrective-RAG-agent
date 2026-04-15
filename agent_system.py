from __future__ import annotations

import argparse
import re
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph


class KnowledgeDocument(TypedDict):
    id: str
    title: str
    content: str


class GraphState(TypedDict):
    original_question: str
    question: str
    retrieved_documents: list[KnowledgeDocument]
    relevant_documents: list[KnowledgeDocument]
    generation: str
    rewrite_count: int
    search_history: list[str]
    decision: Literal["generate", "rewrite", "fallback"]


KNOWLEDGE_BASE: list[KnowledgeDocument] = [
    {
        "id": "agentic-rag-overview",
        "title": "Agentic RAG Overview",
        "content": (
            "Agentic RAG combines retrieval augmented generation with a control loop. "
            "The system retrieves evidence, checks whether that evidence is useful, "
            "and can re-run retrieval before producing the final answer."
        ),
    },
    {
        "id": "corrective-rag",
        "title": "Corrective RAG",
        "content": (
            "Corrective RAG, often shortened to CRAG, adds a critic stage that grades "
            "retrieved documents. When the evidence is weak, the agent rewrites the query "
            "or retries retrieval so the final answer is grounded in stronger context."
        ),
    },
    {
        "id": "query-rewriting",
        "title": "Query Transformation",
        "content": (
            "Query transformation improves search quality by expanding vague questions "
            "with synonyms, domain terms, or more precise intent. It is useful when the "
            "first retrieval attempt misses the right documents."
        ),
    },
    {
        "id": "hallucination-control",
        "title": "Hallucination Control",
        "content": (
            "Self-corrective retrieval pipelines reduce hallucinations by refusing to answer "
            "from low-quality evidence. They prefer verified context and surface uncertainty "
            "when supporting documents are missing."
        ),
    },
]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "with",
}

REWRITE_EXPANSIONS = {
    "agentic rag": "agentic rag corrective rag retrieval augmented generation control loop",
    "crag": "corrective rag self corrective rag retrieval critic",
    "self correcting": "self corrective corrective rag query rewrite relevance grading",
    "hallucinations": "hallucinations grounded evidence verified context",
    "query rewrite": "query rewrite query transformation better retrieval",
    "rewrite the question": "query rewrite query transformation better retrieval",
}

SPECIAL_RELEVANCE = {
    "agentic rag": ("agentic rag",),
    "crag": ("corrective rag",),
    "self correcting": ("self corrective", "corrective rag"),
    "hallucinations": ("hallucinations",),
    "query rewrite": ("query transformation", "rewrites the query"),
    "rewrite the question": ("query transformation", "rewrites the query"),
}

GENERIC_TERMS = {
    "agent",
    "answer",
    "bad",
    "context",
    "evidence",
    "grounded",
    "improve",
    "improved",
    "quality",
    "question",
    "retrieve",
    "retrieval",
    "search",
    "system",
    "work",
}

MAX_REWRITES = 2
TOP_K = 3


def normalize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [token for token in tokens if token not in STOPWORDS]


def score_document(question: str, document: KnowledgeDocument) -> int:
    question_terms = set(normalize(question))
    document_terms = set(normalize(document["title"] + " " + document["content"]))
    shared_terms = question_terms & document_terms
    score = len(shared_terms)

    if "agentic rag" in question.lower() and "agentic rag" in document["content"].lower():
        score += 2
    if "crag" in question.lower() and "corrective rag" in document["content"].lower():
        score += 3
    if "hallucination" in question.lower() and "hallucination" in document["content"].lower():
        score += 2
    if "query" in question.lower() and "transform" in question.lower():
        if "query transformation" in document["content"].lower():
            score += 2

    return score


def retrieve_documents(question: str, top_k: int = TOP_K) -> list[KnowledgeDocument]:
    ranked = sorted(
        KNOWLEDGE_BASE,
        key=lambda document: score_document(question, document),
        reverse=True,
    )
    return [document for document in ranked[:top_k] if score_document(question, document) > 0]


def rewrite_question(question: str, rewrite_count: int) -> str:
    rewritten_question = question.lower()

    for phrase, expansion in REWRITE_EXPANSIONS.items():
        if phrase in rewritten_question:
            rewritten_question = rewritten_question.replace(phrase, expansion)

    if rewritten_question == question.lower():
        rewritten_question = f"{question} clarify terminology"

    if rewrite_count == 0:
        return rewritten_question

    return f"{rewritten_question} improved search"


def build_answer(question: str, documents: list[KnowledgeDocument]) -> str:
    if not documents:
        return (
            f"I could not find strong evidence to answer '{question}'. "
            "The agent stopped after exhausting its rewrite attempts."
        )

    evidence_summary = " ".join(document["content"] for document in documents[:2])
    summary_lines = [
        f"Question: {question}",
        "",
        "Answer:",
        evidence_summary,
        "",
        "Supporting evidence:",
    ]

    for document in documents:
        summary_lines.append(f"- {document['title']}: {document['content']}")

    return "\n".join(summary_lines)


def is_document_relevant(original_question: str, question: str, document: KnowledgeDocument) -> bool:
    document_text = f"{document['title']} {document['content']}".lower()
    original_terms = set(normalize(original_question)) - GENERIC_TERMS
    overlap = original_terms & set(normalize(document_text))

    if len(overlap) >= 2:
        return True

    lowered_original_question = original_question.lower()
    for phrase, expected_matches in SPECIAL_RELEVANCE.items():
        if phrase in lowered_original_question and any(match in document_text for match in expected_matches):
            return True

    return score_document(question, document) >= 3 and len(overlap) >= 1


def retrieve_node(state: GraphState) -> GraphState:
    print(f"--- RETRIEVE: {state['question']} ---")
    documents = retrieve_documents(state["question"])
    return {
        **state,
        "retrieved_documents": documents,
    }


def grade_documents_node(state: GraphState) -> GraphState:
    print("--- GRADE DOCUMENTS ---")
    relevant_documents = [
        document
        for document in state["retrieved_documents"]
        if is_document_relevant(state["original_question"], state["question"], document)
    ]

    if relevant_documents:
        decision: Literal["generate", "rewrite", "fallback"] = "generate"
    elif state["rewrite_count"] < MAX_REWRITES:
        decision = "rewrite"
    else:
        decision = "fallback"

    return {
        **state,
        "relevant_documents": relevant_documents,
        "decision": decision,
    }


def rewrite_query_node(state: GraphState) -> GraphState:
    print("--- REWRITE QUERY ---")
    rewritten_question = rewrite_question(state["question"], state["rewrite_count"])
    history = [*state["search_history"], rewritten_question]
    return {
        **state,
        "question": rewritten_question,
        "rewrite_count": state["rewrite_count"] + 1,
        "search_history": history,
    }


def generate_node(state: GraphState) -> GraphState:
    print("--- GENERATE ANSWER ---")
    return {
        **state,
        "generation": build_answer(state["original_question"], state["relevant_documents"]),
    }


def fallback_node(state: GraphState) -> GraphState:
    print("--- FALLBACK ANSWER ---")
    return {
        **state,
        "generation": build_answer(state["original_question"], state["relevant_documents"]),
    }


def route_after_grading(state: GraphState) -> Literal["generate", "rewrite", "fallback"]:
    return state["decision"]


workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_docs", grade_documents_node)
workflow.add_node("rewrite", rewrite_query_node)
workflow.add_node("generate", generate_node)
workflow.add_node("fallback", fallback_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_docs")
workflow.add_conditional_edges(
    "grade_docs",
    route_after_grading,
    {
        "generate": "generate",
        "rewrite": "rewrite",
        "fallback": "fallback",
    },
)
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)
workflow.add_edge("fallback", END)

app = workflow.compile()


def run_agent(question: str) -> GraphState:
    initial_state: GraphState = {
        "original_question": question,
        "question": question,
        "retrieved_documents": [],
        "relevant_documents": [],
        "generation": "",
        "rewrite_count": 0,
        "search_history": [question],
        "decision": "rewrite",
    }
    return app.invoke(initial_state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local self-corrective RAG demo.")
    parser.add_argument(
        "question",
        nargs="?",
        default="What is Agentic RAG?",
        help="Question to send through the corrective RAG workflow.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    final_state = run_agent(args.question)

    print()
    print(final_state["generation"])
    print()
    print("Search history:")
    for query in final_state["search_history"]:
        print(f"- {query}")
