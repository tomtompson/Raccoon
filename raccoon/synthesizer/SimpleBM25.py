from typing import List, Tuple
from collections import Counter
import math

from raccoon.synthesizer.helper import _tokenize

class SimpleBM25:
    def __init__(self, docs: List[dict], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.N = len(docs)

        self.doc_tokens: List[List[str]] = []
        self.doc_len: List[int] = []
        self.term_freqs: List[Counter] = []
        self.df: Counter = Counter()
        self.avgdl = 0.0

        for doc in docs:
            combined = f'{doc.get("title", "")}\n{doc.get("text", "")}'
            tokens = _tokenize(combined)
            self.doc_tokens.append(tokens)
            self.doc_len.append(len(tokens))

            tf = Counter(tokens)
            self.term_freqs.append(tf)

            for term in tf.keys():
                self.df[term] += 1

        self.avgdl = sum(self.doc_len) / self.N if self.N else 0.0

        self.idf = {}
        for term, df in self.df.items():
            self.idf[term] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10) -> List[Tuple[dict, float]]:
        q_terms = _tokenize(query)
        if not q_terms:
            return []

        scored = []
        for idx, doc in enumerate(self.docs):
            score = 0.0
            dl = self.doc_len[idx] or 1
            tf = self.term_freqs[idx]

            for term in q_terms:
                if term not in tf:
                    continue

                f = tf[term]
                idf = self.idf.get(term, 0.0)

                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                score += idf * ((f * (self.k1 + 1)) / denom)

            if score > 0:
                scored.append((doc, float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]