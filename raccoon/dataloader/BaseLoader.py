from __future__ import annotations

from langchain_core.documents import Document
from abc import ABC, abstractmethod

import json

from typing import Iterable
from pathlib import Path
import uuid

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

class BaseLoader(ABC):
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.data: list[Document] | None = None
    @abstractmethod
    def load_data(self):
        pass
    @abstractmethod
    def preprocess_data(self, data):
        pass
    def get_data(self):
        data = self.load_data()
        return self.preprocess_data(data)
    
    def _save_documents(
    self,
    documents: list[Document],
    output_path: str | Path,
    ensure_ascii: bool = False,
    pretty: bool = False,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            for doc in documents:
                record = {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                }

                if pretty:
                    f.write(json.dumps(record, ensure_ascii=ensure_ascii, indent=2))
                else:
                    f.write(json.dumps(record, ensure_ascii=ensure_ascii))

                f.write("\n")
    
    def save_chunked_documents(
    self,
    parent_output_path: str | Path,
    child_output_path: str | Path,
    ensure_ascii: bool = False,
    pretty: bool = False,
    ) -> None:
        if self.parent_data is None or self.child_data is None:
            raise ValueError("No chunked documents available. Run preprocess_data() first.")

        self._save_documents(self.parent_data, parent_output_path, ensure_ascii=ensure_ascii, pretty=pretty)
        self._save_documents(self.child_data, child_output_path, ensure_ascii=ensure_ascii, pretty=pretty)

    def load_documents(self, input_path: str | Path) -> list[Document]:
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")

        documents: list[Document] = []

        with input_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    documents.append(
                        Document(
                            page_content=record["page_content"],
                            metadata=record.get("metadata", {}),
                        )
                    )
                except Exception as e:
                    raise ValueError(
                        f"Failed to parse line {line_number} in {input_path}: {e}"
                    ) from e
        self.data = documents
        return documents
    
    def chunk(self, docs: list[Document], chunk_size: int, overlap: int) -> tuple[list[Document], list[Document]]:
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
        )

        parent_docs = parent_splitter.split_documents(docs)

        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max(1, chunk_size // 4),
            chunk_overlap=max(0, overlap // 4),
            length_function=len,
            is_separator_regex=False,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
        )

        child_docs: list[Document] = []

        for parent_doc in parent_docs:
            parent_id = str(uuid.uuid4())
            parent_doc.metadata = {
                **parent_doc.metadata,
                "id": parent_id,
                "doc_type": "parent",
            }

            split_children = child_splitter.split_documents([parent_doc])

            for child_doc in split_children:
                child_doc.metadata = {
                    **parent_doc.metadata,       
                    **child_doc.metadata,           
                    "parent_id": parent_id,
                    "child_id": str(uuid.uuid4()),
                    "doc_type": "child",
                }
                child_docs.append(child_doc)

        return child_docs, parent_docs