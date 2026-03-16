from __future__ import annotations

from abc import ABC, abstractmethod
from ..BaseSynthesizer import BaseSynthesizer

class BaseCritiquer(ABC):
    def __init__(self,
                 config: dict | None = None,
                 sythesizer: BaseSynthesizer| None = None,
                 model_id: str | None = None,
                 
                ):
        self.config = config
        self.sythesizer = sythesizer
        self.model_id = model_id
        self.results = []
        self.metrics = {}        
    
    @abstractmethod
    def _load_llm(self):
        pass

    @abstractmethod
    def call_llm(self):
        pass
    
    @abstractmethod
    def load_prompt(self):
        pass

    @abstractmethod
    def critique(self):
        pass
    
    @abstractmethod
    def filter(self):
        pass