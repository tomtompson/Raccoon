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

            throughput_text=(
                "<b>Runtime and throughput guide:</b> These results evaluate the practical performance of the retrieval stage "
                "in addition to retrieval quality. Query time measures how long it takes to process queries, while index time "
                "measures how long it takes to build the full retrieval index. Queries/sec and documents/sec represent the "
                "throughput of the system and provide insight into scalability within larger RAG environments. Storage MB shows "
                "how much storage space the retrieval index requires, which can become important in production environments where "
                "infrastructure cost and memory usage matter. The graph combines retrieval quality with throughput by plotting "
                "Recall@10 against queries per second. Higher positions indicate better retrieval coverage, while positions "
                "further to the right indicate higher speed and lower latency."
            ),

            throughput_interpretation_text=(
                "<b>Practical interpretation:</b> Do not use these results only to select the fastest or most accurate retriever, "
                "but to determine which retrieval architecture best matches the practical requirements of the application. "
                "A retriever positioned in the upper-right region of the graph combines high retrieval quality with high throughput "
                "and therefore often provides the best overall balance for real-time RAG systems. However, some retrievers may "
                "intentionally use more query time to achieve stronger ranking quality or better retrieval coverage. "
                "Production environments should therefore also consider latency, scalability, hardware cost, and maintenance complexity. "
                "A slightly lower retrieval score may still be acceptable when the system remains substantially faster, cheaper, "
                "or easier to scale."
            ),
            rank_distribution_text=(
                "<b>Rank distribution guide:</b> Rank distribution shows where the first relevant document appears in the "
                "ranking for each query. Instead of only showing an overall retrieval score, this distribution makes visible "
                "how quickly useful information becomes available to the RAG pipeline. Rank 1 means that the first returned "
                "result is already relevant. Rank 2-3 means that relevant information is still found very early. Rank 4-10 "
                "indicates that the retriever needs to go deeper into the ranking before useful evidence appears. Rank 11+ "
                "means that relevant information is found relatively late, increasing the chance that the language model "
                "receives weaker or less focused context. The average first rank summarizes how early the first relevant "
                "document usually appears; lower values indicate stronger ranking performance."
            ),

            rank_distribution_interpretation_text=(
                "<b>Practical interpretation:</b> A retriever with many Rank 1 and Rank 2-3 results generally provides more "
                "immediately usable evidence for generation. This is especially valuable when the context window is limited "
                "or when the generator relies heavily on the first passages in the prompt. A larger Rank 4-10 share can still "
                "be useful, but means that relevant information is presented less prominently. A large Rank 11+ share often "
                "suggests weaker ranking quality or a stronger need for reranking, hybrid retrieval, or better query processing. "
                "Rank distribution therefore helps determine not only whether relevant documents are retrieved, but especially "
                "how quickly and how visibly those documents become available to the generator."
            ),
        )