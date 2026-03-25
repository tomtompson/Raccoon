from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..BaseSynthesizer import BaseSynthesizer


class BaseCritiquer(ABC):
    def __init__(
        self,
        config: dict | None = None,
        synthesizer: BaseSynthesizer | None = None,
        model_id: str | None = None,
    ):
        self.config = config or {}
        self.synthesizer = synthesizer
        self.model_id = model_id
        self.results: list[dict[str, Any]] = []
        self.metrics: dict[str, Any] = {}

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
