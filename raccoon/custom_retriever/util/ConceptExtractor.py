import spacy
from typing import Optional, Tuple, Dict, List, Set, Any
from collections import defaultdict

import re

class ConceptExtractor:
    def __init__(
        self,
        model_name: str,
        min_concept_len: int = 6,
        max_concept_words: int = 4,
        stop_concept: Optional[List[str]] = None,
        use_gpu: bool = False,
    ):

        if use_gpu:
            try:
                spacy.require_gpu()
                print("Using spaCy on GPU")
            except Exception:
                print("spaCy GPU unavailable; using CPU")
        else:
            print("Using spaCy on CPU")

        self.nlp = spacy.load(model_name)

        if "sentencizer" not in self.nlp.pipe_names and "parser" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")

        self.min_concept_len = min_concept_len
        self.max_concept_words = max_concept_words
        self.stop_concept = stop_concept or []


    def _clean_concept(self, text: str) -> Optional[str]:
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = text.strip(" .,:;()[]{}\"'")

        if len(text) < self.min_concept_len:
            return None
        if len(text.split()) > self.max_concept_words:
            return None
        if text in self.stop_concept:
            return None
        if text.isdigit():
            return None
        if re.fullmatch(r"[\d\W_]+", text):
            return None

        return text

    def extract_from_doc(self, doc) -> Tuple[Set[str], Dict[str, Set[str]], List[str]]:
        passage_concepts: Set[str] = set()
        sentence_to_concepts: Dict[str, Set[str]] = defaultdict(set)
        sentence_list: List[str] = []

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            sentence_list.append(sent_text)

            concepts = set()

            # Named entities are still useful, but no longer the only concept source.
            for ent in sent.ents:
                c = self._clean_concept(ent.text)
                if c:
                    concepts.add(c)

            # Noun chunks capture legal phrases much better than NER alone.
            if hasattr(sent, "noun_chunks"):
                try:
                    for chunk in sent.noun_chunks:
                        c = self._clean_concept(chunk.text)
                        if c:
                            concepts.add(c)
                except Exception:
                    pass

            # Add useful lemma tokens as fallback concepts.
            for token in sent:
                if token.is_stop or token.is_punct or token.like_num:
                    continue
                if token.pos_ in {"NOUN", "PROPN", "ADJ", "VERB"}:
                    lemma = token.lemma_.strip().lower()
                    c = self._clean_concept(lemma)
                    if c:
                        concepts.add(c)

            if concepts:
                sentence_to_concepts[sent_text].update(concepts)
                passage_concepts.update(concepts)

        return passage_concepts, sentence_to_concepts, sentence_list

    def extract_query_concepts(self, text: str) -> Set[str]:
        doc = self.nlp(text)
        passage_concepts, _, _ = self.extract_from_doc(doc)
        return passage_concepts