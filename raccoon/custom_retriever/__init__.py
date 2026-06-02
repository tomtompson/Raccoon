__all__ = [
    "BaseRetriever",
    "BM25Retriever",
    "DenseRetrieverSentenceBert",
    "LinearRagRetriever",
    "HybridRetriever",
]

_EXPORTS = {
    "BaseRetriever": ".BaseRetriever",
    "BM25Retriever": ".BM25Retriever",
    "DenseRetrieverSentenceBert": ".DenseRetriever",
    "HybridRetriever": ".HybridRetriever",
    "LinearRagRetriever": ".LinearRagRetriever",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
