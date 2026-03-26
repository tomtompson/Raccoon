import json
import unittest

from langchain_core.documents import Document

from raccoon.synthesizer.multi_query.MultiQuerySynthesizer import MultiQueryOllamaSynthesizer


class StubSynthesizer(MultiQueryOllamaSynthesizer):
    def call_llm(self, passage: str) -> dict[str, str]:
        return {
            "response": json.dumps(
                {
                    "entity_question": "What chunk overlap does the loader use?",
                    "entity_answer": "200",
                    "open_question": "How does the loader handle chunk overlap?",
                    "open_answer": "The loader uses a chunk overlap of 200.",
                    "abstract_question": "Does the loader use overlap between chunks?",
                    "abstract_answer": "Yes, it uses a chunk overlap of 200.",
                }
            )
        }


class MultiQueryOllamaSynthesizerTest(unittest.TestCase):
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
                    "entity_question": "What chunk overlap does the loader use?",
                    "entity_answer": "200",
                    "open_question": "How does the loader handle chunk overlap?",
                    "open_answer": "The loader uses a chunk overlap of 200.",
                    "abstract_question": "Does the loader use overlap between chunks?",
                    "abstract_answer": "Yes, it uses a chunk overlap of 200.",
                    "metadata": {"source": "unit-test"},
                }
            ],
        )

        self.assertEqual(
            synthesizer.processed_documents,
            [
                {
                    "passage": "The loader uses a chunk overlap of 200.",
                    "entity_question": "What chunk overlap does the loader use?",
                    "entity_answer": "200",
                    "open_question": "How does the loader handle chunk overlap?",
                    "open_answer": "The loader uses a chunk overlap of 200.",
                    "abstract_question": "Does the loader use overlap between chunks?",
                    "abstract_answer": "Yes, it uses a chunk overlap of 200.",
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

        self.assertEqual(results[0]["entity_question"], "What chunk overlap does the loader use?")
        self.assertEqual(results[0]["entity_answer"], "200")
        self.assertEqual(results[0]["open_question"], "How does the loader handle chunk overlap?")
        self.assertEqual(results[0]["open_answer"], "The loader uses a chunk overlap of 200.")
        self.assertEqual(results[0]["abstract_question"], "Does the loader use overlap between chunks?")
        self.assertEqual(results[0]["abstract_answer"], "Yes, it uses a chunk overlap of 200.")

        self.assertEqual(
            results[0]["questions"],
            [
                {
                    "type": "entity",
                    "question": "What chunk overlap does the loader use?",
                    "answer": "200",
                },
                {
                    "type": "open",
                    "question": "How does the loader handle chunk overlap?",
                    "answer": "The loader uses a chunk overlap of 200.",
                },
                {
                    "type": "abstract",
                    "question": "Does the loader use overlap between chunks?",
                    "answer": "Yes, it uses a chunk overlap of 200.",
                },
            ],
        )

        self.assertEqual(results[0]["metadata"], {"source": "unit-test"})
        self.assertEqual(results[0]["passage"], "The loader uses a chunk overlap of 200.")

    def test_parse_generated_response_rejects_invalid_format(self) -> None:
        synthesizer = StubSynthesizer(
            ollama_url="http://localhost:11434",
            model_id="llama3",
        )

        with self.assertRaises(RuntimeError):
            synthesizer.parse_generated_response("not-json")

    def test_parse_generated_response_accepts_valid_json(self) -> None:
        synthesizer = StubSynthesizer(
            ollama_url="http://localhost:11434",
            model_id="llama3",
        )

        parsed = synthesizer.parse_generated_response(
            json.dumps(
                {
                    "entity_question": "What chunk overlap does the loader use?",
                    "entity_answer": "200",
                    "open_question": "How does the loader handle chunk overlap?",
                    "open_answer": "The loader uses a chunk overlap of 200.",
                    "abstract_question": "Does the loader use overlap between chunks?",
                    "abstract_answer": "Yes, it uses a chunk overlap of 200.",
                }
            )
        )

        self.assertEqual(
            parsed,
            {
                "entity_question": "What chunk overlap does the loader use?",
                "entity_answer": "200",
                "open_question": "How does the loader handle chunk overlap?",
                "open_answer": "The loader uses a chunk overlap of 200.",
                "abstract_question": "Does the loader use overlap between chunks?",
                "abstract_answer": "Yes, it uses a chunk overlap of 200.",
            },
        )


if __name__ == "__main__":
    unittest.main()