import unittest

from crimsonvector.synthesizer.critiquer.OllamaCritiquer import OllamaCritiquer


class StubCritiquer(OllamaCritiquer):
    def call_llm(self, prompt: str) -> str:
        if "Question: What type of cells does a Zeliox unit contain?" in prompt:
            if "Context:" in prompt:
                return "Answer:::\nEvaluation: The context explicitly answers the question.\nTotal rating: 5"
            if "self-contained" in prompt:
                return "Answer:::\nEvaluation: The question stands on its own within the corpus.\nTotal rating: 5"
            return "Answer:::\nEvaluation: This is a useful corpus-specific retrieval question.\nTotal rating: 3"

        if "Question: What does 'G' represent in the data server diagram?" in prompt:
            if "Context:" in prompt:
                return "Answer:::\nEvaluation: The context explicitly defines G.\nTotal rating: 5"
            if "self-contained" in prompt:
                return "Answer:::\nEvaluation: The question depends on a diagram-local reference.\nTotal rating: 1"
            return "Answer:::\nEvaluation: This is tied to a local figure label.\nTotal rating: 2"

        return "Answer:::\nEvaluation: Fallback.\nTotal rating: 1"


class OllamaCritiquerTest(unittest.TestCase):
    def test_critique_and_filter_keep_corpus_specific_question_with_medium_relevance(self) -> None:
        critiquer = StubCritiquer(
            ollama_url="http://localhost:11434",
            model_id="llama3",
            outputs=[
                {
                    "question": "What type of cells does a Zeliox unit contain?",
                    "answer": "lithium cells",
                    "passage": "A Zeliox unit contains lithium cells.",
                }
            ],
        )

        results = critiquer.critique()

        self.assertEqual(results[0]["groundedness_score"], 5)
        self.assertEqual(results[0]["relevance_score"], 3)
        self.assertEqual(results[0]["standalone_score"], 5)
        self.assertEqual(len(critiquer.filter(results=results)), 1)

    def test_filter_uses_only_scores(self) -> None:
        critiquer = StubCritiquer(
            ollama_url="http://localhost:11434",
            model_id="llama3",
        )
        outputs = [
            {
                "question": "What does 'G' represent in the data server diagram?",
                "groundedness_score": 5,
                "relevance_score": 4,
                "standalone_score": 4,
            }
        ]

        filtered = critiquer.filter(results=outputs)

        self.assertEqual(filtered, outputs)
        self.assertEqual(critiquer.metrics["filtered"], 1)
        self.assertEqual(critiquer.metrics["filtered_out"], 0)

    def test_parse_evaluation_rejects_invalid_format(self) -> None:
        critiquer = StubCritiquer(
            ollama_url="http://localhost:11434",
            model_id="llama3",
        )

        with self.assertRaises(RuntimeError):
            critiquer._parse_evaluation("missing fields")


if __name__ == "__main__":
    unittest.main()
