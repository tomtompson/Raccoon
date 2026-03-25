from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

from langchain_core.documents import Document


class BaseSynthesizer(ABC):
    def __init__(
        self,
        config: dict | None = None,
        model_id: str | None = None,
        documents: list[Document | dict[str, Any]] | None = None,
        prompt: str | None = None,
    ):
        self.config = config or {}
        self.model_id = model_id
        self.documents = documents or []
        self.processed_documents: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []
        self.metrics: dict[str, Any] = {}
        self.prompt = prompt
        self.processed_documents = self.process_documents(self.documents)

    def process_doc(self, document: Document | dict[str, Any]) -> dict[str, Any]:
        if isinstance(document, Document):
            return {
                "passage": str(document.page_content),
                "metadata": dict(document.metadata or {}),
            }

        row = dict(document)
        passage = row.get("passage", row.get("content", row.get("page_content", "")))
        row["passage"] = str(passage)
        row["metadata"] = dict(row.get("metadata") or {})
        row.pop("page_content", None)
        return row

    def process_documents(
        self,
        documents: Iterable[Document | dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        source_documents = documents if documents is not None else self.documents
        processed_documents = [self.process_doc(document) for document in source_documents]
        self.processed_documents = processed_documents
        return processed_documents

    @abstractmethod
    def _load_llm(self):
        pass

    @abstractmethod
    def call_llm(self):
        pass
    @abstractmethod
    def synthesize(self) -> list[dict[str, Any]]:
        pass
