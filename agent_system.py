import os
from typing import List, TypedDict
from langgraph.graph import StateGraph, END

# --- 1. SETTING UP THE STATE ---
# This dictionary tracks the "memory" of the agent as it moves through the graph.
class GraphState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    is_relevant: str

# --- 2. THE NODES (THE "BRAIN" CELLS) ---

def retrieve_node(state: GraphState):
    """
    Simulates fetching data from a Vector Database.
    """
    print("--- NODE 1: RETRIEVING DATA ---")
    # Example: Imagine this was fetched from your local database
    retrieved_docs = ["Agentic RAG involves using loops to verify AI output accuracy."]
    return {"documents": retrieved_docs, "question": state["question"]}

def grade_documents_node(state: GraphState):
    """
    Determines if the retrieved data actually answers the user's question.
    """
    print("--- NODE 2: GRADING RELEVANCE ---")
    question = state["question"].lower()
    docs = "".join(state["documents"]).lower()

    # Logic: If the question asks for 'Agentic' and the docs have 'Agentic', it's relevant.
    if "agentic" in question and "agentic" in docs:
        return {"is_relevant": "yes"}
    else:
        return {"is_relevant": "no"}

def generate_node(state: GraphState):
    """
    Simulates generating a final answer based on the validated documents.
    """
    print("--- NODE 3: GENERATING FINAL ANSWER ---")
    return {"generation": "Agentic RAG is a loop-based system that self-corrects logic."}

def rewrite_query_node(state: GraphState):
    """
    If the data was bad, this node 'thinks' of a better way to ask the question.
    """
    print("--- NODE 4: REWRITING QUESTION ---")
    return {"question": f"Refined search for: {state['question']}"}

# --- 3. THE ARCHITECTURE (THE GRAPH) ---

workflow = StateGraph(GraphState)

# Add our functions as nodes in the graph
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_docs", grade_documents_node)
workflow.add_node("generate", generate_node)
workflow.add_node("rewrite", rewrite_query_node)

# Define the flow (Edges)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_docs")

# DECISION POINT: If relevant -> Generate. If not -> Rewrite and try again.
workflow.add_conditional_edges(
    "grade_docs",
    lambda x: x["is_relevant"],
    {
        "yes": "generate",
        "no": "rewrite"
    }
)
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)

# Compile the system
app = workflow.compile()

# --- 4. EXECUTION ---
if __name__ == "__main__":
    print("Starting Agentic Flow...")
    inputs = {"question": "What is Agentic RAG?"}
    for output in app.stream(inputs):
        print(output)
        print("-" * 20)