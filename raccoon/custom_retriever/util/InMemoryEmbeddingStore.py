import numpy as np
from typing import List, Dict, Optional, Tuple
import faiss    


class EmbeddingStore:
    def __init__(self, model, batch_size: int = 64, backend: str = "memory"):
        if backend not in {"memory", "faiss"}:
            raise ValueError("backend must be 'memory' or 'faiss'")

        self.model = model
        self.batch_size = batch_size
        self.backend = backend

        self.ids: List[str] = []
        self.id_to_idx: Dict[str, int] = {}
        self.id_to_text: Dict[str, str] = {}

        self.embeddings: Optional[np.ndarray] = None
        self.index = None
        self.dim: Optional[int] = None

    def build(self, items: List[Tuple[str, str]], prefix_text: str = "passage"):
        items = sorted(items, key=lambda x: len(x[1]), reverse=True)

        self.ids = [item_id for item_id, _ in items]
        self.id_to_idx = {item_id: i for i, item_id in enumerate(self.ids)}
        self.id_to_text = {item_id: text for item_id, text in items}

        if not items:
            self.embeddings = np.zeros((0, 1), dtype=np.float32)
            self.index = None
            self.dim = 0
            return

        texts = [f"{prefix_text}: {text}".strip() for _, text in items]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        self.dim = embeddings.shape[1]

        if self.backend == "memory":
            self.embeddings = embeddings
            self.index = None

        elif self.backend == "faiss":
            self.embeddings = None
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        q = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        top_k = min(top_k, len(self.ids))

        if self.backend == "faiss":
            scores, indices = self.index.search(q, top_k)
            return indices[0], scores[0]

        if self.embeddings is None:
            raise ValueError("No embeddings available. Run build() first.")

        scores = np.dot(self.embeddings, q.T).flatten()
        indices = np.argsort(scores)[::-1][:top_k]

        return indices, scores[indices]