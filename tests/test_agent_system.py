import unittest

from agent_system import run_agent


class AgentSystemTests(unittest.TestCase):
    def test_direct_retrieval_answers_agentic_rag(self) -> None:
        result = run_agent("What is Agentic RAG?")
        self.assertIn("control loop", result["generation"].lower())
        self.assertEqual(result["rewrite_count"], 0)

    def test_rewrite_path_finds_better_context(self) -> None:
        result = run_agent("Why does it rewrite the question?")
        self.assertGreater(result["rewrite_count"], 0)
        self.assertIn("query transformation", result["generation"].lower())
        self.assertGreater(len(result["relevant_documents"]), 0)

    def test_unrelated_question_falls_back(self) -> None:
        result = run_agent("What are transformer attention heads?")
        self.assertIn("could not find strong evidence", result["generation"].lower())
        self.assertEqual(result["decision"], "fallback")


if __name__ == "__main__":
    unittest.main()
