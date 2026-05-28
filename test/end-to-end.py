from pathlib import Path

from raccoon.custom_retriever.DenseRetriever import DenseRetrieverSentenceBert
from raccoon.custom_retriever.HybridRetriever import HybridRetriever
from raccoon.custom_retriever.LinearRagRetriever import LinearRagRetriever
from raccoon.report.StaticRetrieverReport import StaticRetrieverReport
from raccoon.dataloader.FixedDocumentLoader import FixedDocumentLoader
from raccoon.synthesizer import OllamaSynthesizer
from pathlib import Path
from raccoon.dataloader.utils import load_local_beir_dataset
from raccoon.custom_retriever.BM25Retriever import BM25Retriever
from raccoon.custom_retriever.util.utils import append_results
from raccoon.custom_retriever.util.Reranker import Reranker 

from testcontainers.elasticsearch import ElasticSearchContainer
from beir.retrieval.evaluation import EvaluateRetrieval
import torch


SOURCE_PATH_RAW = Path("data/raw/rechtspraak")
OUTPUT_PATH_CHUNKS = Path("data/processed/chunks_600")

RERANKER_ID = "BAAI/bge-reranker-v2-m3"

QUERY_PROMPT_PATH = "prompts/query_scenarios/query_generation_ambiguous.txt"
OUTPUT_PATH_SYNTH = "data/processed/rechtspraken/beir_600_ambiguous"

RESULT_FILE_PATH = "data/processed/rechtspraken/beir_600_ambiguous/eval_results.json"

INPUT_PATH = Path("data/processed/rechtspraken/beir_600_ambiguous")

ELASTIC_IMAGE = "docker.elastic.co/elasticsearch/elasticsearch:8.13.4"
TOP_K = 20



TOP_K = 20
MODEL_ID = "snowflake/snowflake-arctic-embed-l-v2.0"
MAX_LENGHT = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUERY_PROMPT_NAME = "query"
PASSAGE_PROMPT_NAME = "document"
ENCODE_PATH = "data/processed/rechtspraken/beir_600_ambiguous/encode/"

RETRIEVERS = []


PDF_PATH = Path("data/processed/rechtspraken/report_ambiguous.pdf")


def main() -> None:
    #======================================================
    # Document loading and chunking
    #======================================================
    # loader = FixedDocumentLoader(
    #     path=SOURCE_PATH_RAW,
    #     chunk_size=4000, # Document will be split into chunks of 4000 characters then futher split into child chunks // 4
    #     chunk_overlap=200,
    # )
    # documents_parents , documents_child = loader.get_data()
    # loader.save_chunked_documents(OUTPUT_PATH_CHUNKS / "parent_chunks.json", OUTPUT_PATH_CHUNKS / "child_chunks.json")

    # print(f"Saved {len(documents_parents)} parent and {len(documents_child)} child documents to {OUTPUT_PATH_CHUNKS}")


    # #======================================================
    # # Synthesize dataset
    # #======================================================
    synth = OllamaSynthesizer(
    host="http://localhost:11434",
    prompt_paths={
        "query_generation": QUERY_PROMPT_PATH,
        "query_validation": "prompts/example_rechtspraak/query_validation.txt",
        "candidate_judging": "prompts/example_rechtspraak/candidate_judging.txt",
        "distribution_validation": "prompts/example_rechtspraak/distribution_validation.txt",
    },)
    # parent = synth.load_chunks(OUTPUT_PATH_CHUNKS / "parent_chunks.json")
    # child = synth.load_chunks(OUTPUT_PATH_CHUNKS / "chunks.json")

    # synth.synthesize_beir(
    #     parent_chunks=parent,
    #     child_chunks=child,
    #     output_dir=OUTPUT_PATH_SYNTH,
    #     embedding_id="Snowflake/snowflake-arctic-embed-l-v2.0",
    #     embedding_kwargs= {},
    #     query_model="qwen3:8b",
    #     query_validation_model="qwen3:8b",
    #     judge_model="qwen3:8b",
    #     qrel_validation_model="qwen3:8b",
    #     queries_per_parent_to_generate=2,
    #     max_queries_to_keep_per_parent=1,
    #     dense_k=25,
    #     bm25_k=15,
    #     rrf_top_k=15,
    #     same_topic_negative_k=3,
    #     random_negative_k=3,
    #     min_score_to_keep_in_qrels=2,
    #     max_qrels_per_query=3,
    #     include_source_parent_children=True,
    #     parent_text_limit_prompt=3000,
    #     candidate_text_limit_prompt=800,
    #     max_estimated_tokens= 5000,
    #     overwrite_corpus=False,
    #     max_parents=None,
    #     random_seed=42,
    #     target_queries_per_source=1,
    #     shuffle_parents=True,
    #     reranker_id= RERANKER_ID,
    #     #None,
    #     rerank_pool_size= 70,
    #     rerank_keep_top_k=50,
    # )



    #======================================================
    # Initialize Reranker
    #======================================================
    reranker = Reranker(
        model_id=RERANKER_ID,
        top_k=TOP_K,
        batch_size=16,
        max_length=MAX_LENGHT,
        device=DEVICE,
    )

    #======================================================
    # BM25 Retrieval and evaluation
    #======================================================
    
    corpus, queries, qrels, name = load_local_beir_dataset(INPUT_PATH, "test")

    description = synth.generate_description_of_ds(
        model="qwen3:8b",
        language="dutch",
        corpus=corpus,
        queries=queries,
        description_length=250,
    )


    with ElasticSearchContainer(ELASTIC_IMAGE) as container:
        elasticsearch_url = (
            f"http://{container.get_container_host_ip()}:{container.get_exposed_port(container.port)}"
        )
        retriever_bm25 = BM25Retriever(
            elasticsearch_url=elasticsearch_url,
            index_name="raccoon-bm25-test",
            corpus = corpus,
            queries = queries,
            topk=TOP_K,
            reranker=reranker,
        )
        retriever_bm25.index_corpus()
        retriever_bm25.search()
        eval = EvaluateRetrieval()
        eval_results = eval.evaluate(qrels=qrels, results=retriever_bm25.results, k_values=[1, 3, 5, 10, 20],)
        retriever_bm25.add_retrieval_result(eval_results)
        append_results(RESULT_FILE_PATH, retriever_bm25.retriever_type,retriever_bm25.retrieval_metrics, retriever_bm25.metrics)
        

        eval_results = eval.evaluate(qrels=qrels, results=retriever_bm25.rerank_results, k_values=[1, 3, 5, 10, 20],)
        retriever_bm25.add_rerank_retrieval_result(eval_results)
        append_results(
            RESULT_FILE_PATH,
            "reranker-bm25",    
            {
                "rerank": retriever_bm25.retrieval_metrics["rerank"],
                "rerank_time": retriever_bm25.rerank_metrics,
            }
        )

        RETRIEVERS.append(retriever_bm25)
    #======================================================
    # Dense Retrieval and evaluation
    #======================================================

    retriever_dense = DenseRetrieverSentenceBert(corpus=corpus, 
                                           queries=queries,
                                           model_id=MODEL_ID,
                                           max_length=MAX_LENGHT,
                                           device=DEVICE,
                                           query_prompt_name=QUERY_PROMPT_NAME,
                                           passage_prompt_name=PASSAGE_PROMPT_NAME,
                                           reranker=reranker,)
    retriever_dense.search(top_k=TOP_K, 
                     encode_output_path= ENCODE_PATH,
                     )
    eval = EvaluateRetrieval()
    eval_results = eval.evaluate(qrels=qrels, results=retriever_dense.results, k_values=[1, 3, 5, 10, 20],)
    retriever_dense.add_retrieval_result(eval_results)
    append_results(RESULT_FILE_PATH, retriever_dense.retriever_type, retriever_dense.retrieval_metrics, retriever_dense.metrics)
    
    eval_results = eval.evaluate(qrels=qrels, results=retriever_dense.rerank_results, k_values=[1, 3, 5, 10, 20],)
    retriever_dense.add_rerank_retrieval_result(eval_results)
    append_results(
    RESULT_FILE_PATH,
    "reranker-dense",
    {
        "rerank": retriever_dense.retrieval_metrics["rerank"],
        "rerank_time": retriever_dense.rerank_metrics,
    }
    )
    
    RETRIEVERS.append(retriever_dense)

    #======================================================
    # Hybrid Retrieval and evaluation
    #======================================================
    
    retriever_hybrid = HybridRetriever(
        corpus = corpus,
        queries = queries,
        retrievers=[
            (retriever_bm25, 0.3),
            (retriever_dense, 0.7),
        ],
        k=60,
        reranker=reranker,
    )
    retriever_hybrid.search(top_k=TOP_K)
    eval = EvaluateRetrieval()
    eval_results = eval.evaluate(qrels=qrels, results=retriever_hybrid.results, k_values=[1, 3, 5, 10, 20],)
    retriever_hybrid.add_retrieval_result(eval_results)
    append_results(RESULT_FILE_PATH, retriever_hybrid.retriever_type, retriever_hybrid.retrieval_metrics, retriever_hybrid.metrics)
    
    eval_results = eval.evaluate(qrels=qrels, results=retriever_hybrid.rerank_results, k_values=[1, 3, 5, 10, 20],)
    retriever_hybrid.add_rerank_retrieval_result(eval_results)
    append_results(
    RESULT_FILE_PATH,
    "reranker-hybrid",
    {
        "rerank": retriever_hybrid.retrieval_metrics["rerank"],
        "rerank_time": retriever_hybrid.rerank_metrics,
    }
)
    
    RETRIEVERS.append(retriever_hybrid)  
 
    #======================================================
    # LinearRAG Retrieval and evaluation
    #======================================================

    config = {
    "embedding_model_name": "snowflake/snowflake-arctic-embed-l-v2.0",
    "spacy_model_name": "nl_core_news_sm",
    "dataset_name": name,
    "cache_path": "data/processed/rechtspraken/beir_600_ambiguous/linear_rag_cache",

    "device": DEVICE,
    "max_seq_length": MAX_LENGHT,

    # Match DenseRetriever more closely
    "embed_batch_size": 128,
    "ner_batch_size": 64,

    "retrieval_top_k": TOP_K,

    # Candidate pools
    "dense_candidate_k": 500,
    "bm25_candidate_k": 500,
    "graph_candidate_k": 500,

    # Local graph seeds
    "local_graph_dense_seed_k": 200,
    "local_graph_bm25_seed_k": 200,
    "local_graph_sentence_seed_k": 50,
    "local_graph_concept_seed_k": 50,

    # Fusion
    "dense_rrf_weight": 0.0,
    "bm25_rrf_weight": 0.0,
    "graph_rrf_weight": 1.0,
    "rrf_k": 60,

    # Much stricter concept filtering
    "min_concept_len": 4,
    "max_concept_words": 4,
    "min_concept_df": 2,
    "max_concept_df_ratio": 0.10,

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

    retriever_linear = LinearRagRetriever(
        config=config,
        corpus=corpus,
        queries=queries,
        reranker=reranker,
    )

    retriever_linear.index_corpus()
    retriever_linear.search(top_k=TOP_K)



    evaluator = EvaluateRetrieval()
    eval_results = evaluator.evaluate(
        qrels=qrels,
        results=retriever_linear.results,
        k_values=[1, 3, 5, 10, 50, 100],
    )
    retriever_linear.add_retrieval_result(eval_results)
    append_results(RESULT_FILE_PATH, retriever_linear.retriever_type, retriever_linear.retrieval_metrics, retriever_linear.metrics)


    eval_results = eval.evaluate(qrels=qrels, results=retriever_linear.rerank_results, k_values=[1, 3, 5, 10, 20],)
    retriever_linear.add_rerank_retrieval_result(eval_results)
    append_results(
    RESULT_FILE_PATH,
    "reranker-linear",
    {
        "rerank": retriever_linear.retrieval_metrics["rerank"],
        "rerank_time": retriever_linear.rerank_metrics,
    }
)

    RETRIEVERS.append(retriever_linear)
    


    #======================================================
    # Generate Report for Retrievers
    #======================================================  
    report_config = {
    # Report samples
    "sample_queries": 1,
    "sample_results_per_query": 3,
    "sample_text_chars": 500,

    # Tables
    "metric_rows": 30,
    "retrieval_metric_rows": 40,

    # Main comparison charts
    "comparison_metric": ["NDCG@10", "Recall@10"],
    "rerank_comparison_metric": ["NDCG@10", "Recall@10"],

    "evaluation": {
        "k_values": [1, 3, 5, 10, 20],
        "metrics": [
            "NDCG",
            "MAP",
            "Recall",
            "Precision",
            "MRR",
        ],
    },
    }
    report = StaticRetrieverReport()
    report.generate_report(title = "Retriever Performance Report", ds_description = description, retrievers=RETRIEVERS, qrels=qrels, output_path=PDF_PATH, language="dutch", config=report_config)





if __name__ == "__main__":
    main()