from raccoon.report.util.BaseText import BaseText


class EnglishText(BaseText):
    def __init__(self):
        super().__init__(
            intro_text= ("This report compares {retriever_count} {retriever_word} for retrieval-augmented generation. "
                "It summarizes the corpus and query set, compares ranking quality and runtime, then shows each "
                "retriever's configuration and sample hits. Use the comparison section to choose which retriever "
                "is most likely to put useful evidence into the generator's context window."),
            
            metric_guide_text=("<b>Metric guide:</b> These scores evaluate the retrieval stage before the language model writes an answer. "
                "In a RAG pipeline, better retrieval means the generator receives more relevant, better ranked evidence "
                "and has less need to rely on unsupported model knowledge."),

            metric_text=("<b>Recall@k</b> is coverage: how many relevant documents were found in the top k. "
                "<b>Precision@k</b> is focus: how much of the top k is relevant instead of distracting context. "
                "<b>MAP@k</b> rewards relevant documents appearing consistently early across queries. "
                "<b>NDCG@k</b> rewards rank order and graded relevance, so it is often the best single signal for "
                "whether the strongest evidence reaches the prompt first."),

            practical_interpretation_text=("<b>Practical interpretation:</b> Do not use these results solely to select the highest score, "
                "but to determine which retriever best suits the purpose of the RAG application."
                "A high Recall@k is particularly important when the system must miss as little relevant information as possible, "
                "for example, with legal or support queries. A high Precision@k is important when the "
                "context window is limited and irrelevant passages can distract the generator."
                "When reranking yields clearly better NDCG or MAP scores, this means that the correct documents are not only "
                "found, but also appear higher in the rankings. This is practically valuable because a language model "
                "usually relies more heavily on the first passages in the context."
                "Runtime must also be taken into account: a slightly lower score may be acceptable if the retriever is much "
                "faster or remains easier to maintain."),
        )