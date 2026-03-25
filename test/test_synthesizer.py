import unittest

from langchain_core.documents import Document

from raccoon.synthesizer.OllamaSynthesizer import OllamaSynthesizer


class StubSynthesizer(OllamaSynthesizer):
    def call_llm(self, passage: str) -> dict[str, str]:
        return {
            "response": (
                "Factoid question: What chunk overlap is configured?\n"
                "Answer: The chunk overlap is 200."
            )
        }


class OllamaSynthesizerTest(unittest.TestCase):
    def test_process_doc_returns_jsonl_ready_row(self) -> None:
        synthesizer = StubSynthesizer(
            ollama_url="http://localhost:11434",
            model_id="llama3",
        )

        row = synthesizer.process_doc(
            Document(
                page_content="The loader uses a chunk overlap of 200.",
                metadata={"source": "unit-test"},
            )
        )

        self.assertEqual(
            row,
            {
                "passage": "The loader uses a chunk overlap of 200.",
                "metadata": {"source": "unit-test"},
            },
        )

    def test_init_processes_pre_synthesized_rows(self) -> None:
        synthesizer = StubSynthesizer(
            ollama_url="http://localhost:11434",
            model_id="llama3",
            documents=[
                {
                    "page_content": "The loader uses a chunk overlap of 200.",
                    "question": "What chunk overlap is configured?",
                    "answer": "The chunk overlap is 200.",
                    "metadata": {"source": "unit-test"},
                }
            ],
        )

        self.assertEqual(
            synthesizer.processed_documents,
            [
                {
                    "passage": "The loader uses a chunk overlap of 200.",
                    "question": "What chunk overlap is configured?",
                    "answer": "The chunk overlap is 200.",
                    "metadata": {"source": "unit-test"},
                }
            ],
        )

    def test_synthesize_returns_question_answer_pairs(self) -> None:
        synthesizer = StubSynthesizer(
            ollama_url="http://localhost:11434",
            model_id="llama3",
            documents=[
                Document(
                    page_content="The loader uses a chunk overlap of 200.",
                    metadata={"source": "unit-test"},
                )
            ],
        )

        results = synthesizer.synthesize()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["question"], "What chunk overlap is configured?")
        self.assertEqual(results[0]["answer"], "The chunk overlap is 200.")
        self.assertEqual(results[0]["metadata"], {"source": "unit-test"})
        self.assertEqual(results[0]["passage"], "The loader uses a chunk overlap of 200.")

    def test_parse_generated_response_rejects_invalid_format(self) -> None:
        synthesizer = StubSynthesizer(
            ollama_url="http://localhost:11434",
            model_id="llama3",
        )

        with self.assertRaises(RuntimeError):
            synthesizer.parse_generated_response("not-json")


if __name__ == "__main__":
    unittest.main()
