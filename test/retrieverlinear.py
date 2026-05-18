from pathlib import Path
import torch

from raccoon.dataloader.utils import load_local_beir_dataset
from raccoon.custom_retriever.LinearRagRetriever import LinearRagRetriever

from beir.retrieval.evaluation import EvaluateRetrieval


INPUT_PATH = Path("data/processed/rechtspraken/beir_realistic_TEST")
TOP_K = 100


def main() -> None:
    corpus, queries, qrels, name = load_local_beir_dataset(INPUT_PATH, "test")

    config = {
        "embedding_model_name": "snowflake/snowflake-arctic-embed-l-v2.0",
        "spacy_model_name": "nl_core_news_sm",
        "dataset_name": name,

        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_seq_length": 256,

        "retrieval_top_k": TOP_K,
        "embed_batch_size": 192,
        "ner_batch_size": 64,

        "dense_candidate_k": 500,
        "bm25_candidate_k": 500,
        "graph_candidate_k": 500,

        "local_graph_dense_seed_k": 200,
        "local_graph_bm25_seed_k": 200,
        "local_graph_sentence_seed_k": 80,
        "local_graph_concept_seed_k": 80,

        "dense_rrf_weight": 0.0,
        "bm25_rrf_weight": 0.0,
        "graph_rrf_weight": 1.0,
        "rrf_k": 60,

        "min_concept_len": 4,
        "max_concept_words": 6,
        "min_concept_df": 1,
        "max_concept_df_ratio": 0.30,

        "stop_concept": [
            "artikel", "rechtbank", "zaak", "zaken", "eiser", "gedaagde",
            "verzoeker", "verweerster", "werknemer", "werkgever", "partij",
            "partijen", "overeenkomst", "arbeidsovereenkomst", "datum",
            "januari", "februari", "maart", "april", "mei", "juni",
            "juli", "augustus", "september", "oktober", "november",
            "december", "lid", "grond", "beroep", "besluit", "uitspraak",
            "rechter", "kantonrechter", "proces", "procedure", "verzoek",
            "vordering",
        ],

        "ppr_damping": 0.6,
        "ppr_max_result_docs": 500,
    }

    retriever = LinearRagRetriever(
        config=config,
        corpus=corpus,
        queries=queries,
    )

    retriever.index_corpus()
    retriever.search(top_k=TOP_K)



    evaluator = EvaluateRetrieval()
    eval_results = evaluator.evaluate(
        qrels=qrels,
        results=retriever.results,
        k_values=[1, 3, 5, 10, 50, 100],
    )

    print(eval_results)


if __name__ == "__main__":
    main()