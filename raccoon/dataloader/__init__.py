__all__ = [
    "BaseLoader",
    "FixedDocumentLoader",
    "SQLDocumentLoader",
]

_EXPORTS = {
    "BaseLoader": ".BaseLoader",
    "FixedDocumentLoader": ".FixedDocumentLoader",
    "SQLDocumentLoader": ".SQLDocumentLoader",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
