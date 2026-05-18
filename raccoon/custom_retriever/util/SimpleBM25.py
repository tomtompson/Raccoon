from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import math
from .utils import tokenize

class SimpleBM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.doc_len: Dict[str, int] = {}
        self.avgdl: float = 0.0
        self.doc_tf: Dict[str, Counter] = {}
        self.df: Counter = Counter()
        self.idf: Dict[str, float] = {}
        self.inverted: Dict[str, List[str]] = defaultdict(list)

    def build(self, doc_texts: Dict[str, str]) -> None:
        self.doc_ids = list(doc_texts.keys())
        total_len = 0

        for doc_id, text in doc_texts.items():
            terms = tokenize(text)
            tf = Counter(terms)
            self.doc_tf[doc_id] = tf
            self.doc_len[doc_id] = len(terms)
            total_len += len(terms)

            for term in tf.keys():
                self.df[term] += 1
                self.inverted[term].append(doc_id)

        n = max(1, len(self.doc_ids))
        self.avgdl = total_len / n if n else 0.0

        for term, df in self.df.items():
            self.idf[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        q_terms = tokenize(query)
        if not q_terms:
            return []

        scores = defaultdict(float)
        unique_terms = set(q_terms)

        for term in unique_terms:
            if term not in self.inverted:
                continue
            idf = self.idf.get(term, 0.0)
            for doc_id in self.inverted[term]:
                tf = self.doc_tf[doc_id].get(term, 0)
                dl = self.doc_len.get(doc_id, 0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / max(1e-9, self.avgdl))
                scores[doc_id] += idf * (tf * (self.k1 + 1.0)) / max(1e-9, denom)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
