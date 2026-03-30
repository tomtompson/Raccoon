import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from raccoon.synthesizer.critiquer.OllamaCritiquer import OllamaCritiquer


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


class StubSynthesizer:
    def __init__(self, documents=None, results=None):
        self.documents = documents
        self.processed_documents = []
        self.results = results or []

    def synthesize(self):
        self.results = [
            {
                "question": "What type of cells does a Zeliox unit contain?",
                "answer": "lithium cells",
                "passage": self.documents[0].page_content,
                "metadata": self.documents[0].metadata,
            }
        ]
        return self.results


class PreSynthesizedStubSynthesizer:
    def __init__(self, documents):
        self.documents = documents
        self.processed_documents = [dict(document) for document in documents]
        self.results = []

    def synthesize(self):
        raise AssertionError("synthesize() should not be called for pre-synthesized rows")


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
            critiquer.parse_evaluation("missing fields")

    def test_critique_can_start_from_documents_via_synthesizer(self) -> None:
        documents = [
            Document(
                page_content="A Zeliox unit contains lithium cells.",
                metadata={"source": "unit"},
            )
        ]
        critiquer = StubCritiquer(
            ollama_url="http://localhost:11434",
            model_id="llama3",
            synthesizer=StubSynthesizer(documents=documents),
        )

        results = critiquer.critique()

        self.assertEqual(results[0]["question"], "What type of cells does a Zeliox unit contain?")
        self.assertEqual(results[0]["answer"], "lithium cells")
        self.assertEqual(results[0]["source"], "unit")

    def test_critique_requires_passage_in_input_rows(self) -> None:
        critiquer = StubCritiquer(
            ollama_url="http://localhost:11434",
            model_id="llama3",
            outputs=[
                {
                    "question": "What type of cells does a Zeliox unit contain?",
                    "answer": "lithium cells",
                    "content": "A Zeliox unit contains lithium cells.",
                }
            ],
        )

        with self.assertRaises(KeyError):
            critiquer.critique()

    def test_critique_uses_pre_synthesized_rows_without_running_synthesize(self) -> None:
        critiquer = StubCritiquer(
            ollama_url="http://localhost:11434",
            model_id="llama3",
            synthesizer=PreSynthesizedStubSynthesizer(
                documents=[
                    {
                        "passage": "A Zeliox unit contains lithium cells.",
                        "question": "What type of cells does a Zeliox unit contain?",
                        "answer": "lithium cells",
                        "metadata": {"source": "unit"},
                    }
                ]
            ),
        )

        results = critiquer.critique()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["question"], "What type of cells does a Zeliox unit contain?")
        self.assertEqual(results[0]["answer"], "lithium cells")
        self.assertEqual(results[0]["source"], "unit")

    def test_save_and_load_round_trip_preserves_results_and_metrics(self) -> None:
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
        critiquer.critique()
        critiquer.filter()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "critique.json"
            critiquer.save(output_path)

            loaded = StubCritiquer(
                ollama_url="http://localhost:11434",
                model_id="placeholder",
            )
            loaded.load(output_path)

        self.assertEqual(loaded.results, critiquer.results)
        self.assertEqual(loaded.metrics, critiquer.metrics)
        self.assertEqual(loaded.outputs, critiquer.outputs)
        self.assertEqual(loaded.model_id, "llama3")


if __name__ == "__main__":
    unittest.main()
