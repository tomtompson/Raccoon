from __future__ import annotations

from abc import ABC, abstractmethod
from langchain_core.documents import Document

class BaseSynthesizer(ABC):
    def __init__(self, 
                 config: dict | None = None, 
                 model_id: str | None = None,
                 documents: list[Document] | None = None,
                 prompt: str | None = None):
        self.config = config
        self.model_id = model_id
        self.documents = documents
        self.results = []
        self.metrics = {}
        self.prompt = prompt

    @abstractmethod
    def _load_llm(self):
        pass

    @abstractmethod
    def call_llm(self):
        pass
    
    @abstractmethod
    def synthesize(self) -> dict[str, str, str, str]:
        pass