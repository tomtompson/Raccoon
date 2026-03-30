from __future__ import annotations

from langchain_core.documents import Document
from abc import ABC, abstractmethod

import json

from typing import Iterable
from pathlib import Path

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
    
    def save_documents(self,
    output_path: str | Path,
    ensure_ascii: bool = False,
    pretty: bool = False,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            for doc in self.data:
                record = {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                }

                if pretty:
                    f.write(json.dumps(record, ensure_ascii=ensure_ascii, indent=2))
                    f.write("\n")
                else:
                    f.write(json.dumps(record, ensure_ascii=ensure_ascii))
                    f.write("\n")

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