import unittest

from crimsonvector.synthesizer.critiquer.OllamaCritiquer import OllamaCritiquer


class StubCritiquer(OllamaCritiquer):
    def call_llm(self, prompt: str) -> str:
        if "Context:" in prompt:
            return "Answer:::\nEvaluation: The context clearly supports the question.\nTotal rating: 5"
        if "context-independent" in prompt:
            return "Answer:::\nEvaluation: The question stands on its own.\nTotal rating: 4"
        return "Answer:::\nEvaluation: The question is useful for Hugging Face users.\nTotal rating: 4"


class OllamaCritiquerTest(unittest.TestCase):
    def test_sythesize_and_filter_scores_outputs(self) -> None:
        critiquer = StubCritiquer(
            ollama_url="http://localhost:11434",
            model_id="llama3",
            outputs=[
                {
                    "question": "What does the passage say about chunk overlap?",
                    "answer": "It uses 200 tokens of overlap.",
                    "passage": "The loader uses a chunk overlap of 200 tokens.",
                }
            ],
        )

        results = critiquer.sythesize()

        self.assertEqual(results[0]["groundedness_score"], 5)
        self.assertEqual(results[0]["relevance_score"], 4)
        self.assertEqual(results[0]["standalone_score"], 4)
        self.assertEqual(len(critiquer.filter()), 1)

    def test_parse_evaluation_rejects_invalid_format(self) -> None:
        critiquer = StubCritiquer(
            ollama_url="http://localhost:11434",
            model_id="llama3",
        )

        with self.assertRaises(RuntimeError):
            critiquer._parse_evaluation("missing fields")


if __name__ == "__main__":
    unittest.main()
