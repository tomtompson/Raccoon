from typing import List, Dict, Tuple
import faiss
import numpy as np

class FaissEmbeddingStore:
    def __init__(self, model, batch_size: int = 64):
        self.model = model
        self.batch_size = batch_size
        self.ids: List[str] = []
        self.id_to_idx: Dict[str, int] = {}
        self.id_to_text: Dict[str, str] = {}
        self.index = None
        self.dim = None

    def build(self, items: List[Tuple[str, str]], prefix_text: str = "passage", print_every: int = 5000):
        self.ids = [item_id for item_id, _ in items]
        self.id_to_idx = {item_id: i for i, item_id in enumerate(self.ids)}
        self.id_to_text = {item_id: text for item_id, text in items}

        if not items:
            self.index = None
            self.dim = 0
            return

        total = len(items)

        for start in range(0, total, self.batch_size):
            batch = items[start:start + self.batch_size]
            texts = [f"{prefix_text}: {text}" for _, text in batch]

            emb = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)

            if self.index is None:
                self.dim = emb.shape[1]
                self.index = faiss.IndexFlatIP(self.dim)

            self.index.add(emb)

            done = min(start + self.batch_size, total)
            if done % print_every == 0 or done == total:
                print(f"Embedding {prefix_text}s: {done}/{total}")

            del batch, texts, emb

    def search(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        q = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        top_k = min(top_k, len(self.ids))
        scores, indices = self.index.search(q, top_k)
        return indices[0], scores[0]