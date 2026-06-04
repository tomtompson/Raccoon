__all__ = [
    "StaticRetrieverReport",
]


def __getattr__(name):
    if name != "StaticRetrieverReport":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .StaticRetrieverReport import StaticRetrieverReport

    globals()[name] = StaticRetrieverReport
    return StaticRetrieverReport
