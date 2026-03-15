from __future__ import annotations

from langchain_core.documents import Document


class BaseLoader:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.data: list[Document] | None = None

    def load_data(self):
        raise NotImplementedError("Subclasses must implement this method")

    def preprocess_data(self, data):
        raise NotImplementedError("Subclasses must implement this method")

    def get_data(self):
        data = self.load_data()
        return self.preprocess_data(data)