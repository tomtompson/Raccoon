from __future__ import annotations

from langchain_core.documents import Document
from abc import ABC, abstractmethod

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