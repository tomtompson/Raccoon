from __future__ import annotations

import logging
import os

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

QUERY_LEVEL = 25
JUDGE_LEVEL = 26
START_LEVEL = 27
END_LEVEL = 28
RERANK_LEVEL = 29

logging.addLevelName(QUERY_LEVEL, "QUERY")
logging.addLevelName(JUDGE_LEVEL, "JUDGE")
logging.addLevelName(START_LEVEL, "START")
logging.addLevelName(END_LEVEL, "END")
logging.addLevelName(RERANK_LEVEL, "RERANK")

console = Console(
    theme=Theme(
        {
            "logging.level.query": "green",
            "logging.level.judge": "orange1",
            "logging.level.start": "cyan3",
            "logging.level.end": "bright_magenta",
            "logging.level.rerank": "sky_blue2",
        }
    )
)


def query(self: logging.Logger, message, *args, **kwargs):
    if self.isEnabledFor(QUERY_LEVEL):
        kwargs.setdefault("stacklevel", 2)
        self._log(QUERY_LEVEL, message, args, **kwargs)


def judge(self: logging.Logger, message, *args, **kwargs):
    if self.isEnabledFor(JUDGE_LEVEL):
        kwargs.setdefault("stacklevel", 2)
        self._log(JUDGE_LEVEL, message, args, **kwargs)


def start(self: logging.Logger, message, *args, **kwargs):
    if self.isEnabledFor(START_LEVEL):
        kwargs.setdefault("stacklevel", 2)
        self._log(START_LEVEL, message, args, **kwargs)


def end(self: logging.Logger, message, *args, **kwargs):
    if self.isEnabledFor(END_LEVEL):
        kwargs.setdefault("stacklevel", 2)
        self._log(END_LEVEL, message, args, **kwargs)


def rerank(self: logging.Logger, message, *args, **kwargs):
    if self.isEnabledFor(RERANK_LEVEL):
        kwargs.setdefault("stacklevel", 2)
        self._log(RERANK_LEVEL, message, args, **kwargs)


def install_custom_levels() -> None:
    logging.Logger.query = query
    logging.Logger.judge = judge
    logging.Logger.start = start
    logging.Logger.end = end
    logging.Logger.rerank = rerank


def _resolve_level(level: int | str | None) -> int:
    value = level if level is not None else os.getenv("RACCOON_LOG_LEVEL", "DEBUG")
    if isinstance(value, int):
        return value

    resolved = logging.getLevelName(value.upper())
    if isinstance(resolved, int):
        return resolved
    return logging.INFO


def configure_logging(level: int | str | None = None, force: bool = False) -> logging.Logger:
    install_custom_levels()
    logger = logging.getLogger("raccoon")
    if getattr(logger, "_raccoon_configured", False) and level is None and not force:
        return logger

    resolved_level = _resolve_level(level)

    if force:
        logger.handlers.clear()

    if not logger.handlers:
        handler = RichHandler(rich_tracebacks=True, markup=True, console=console)
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        logger.addHandler(handler)

    logger.setLevel(resolved_level)
    logger.propagate = False
    logger._raccoon_configured = True
    return logger


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


def get_raccoon_logger(name: str) -> logging.Logger:
    return get_logger(name)


def silence_noisy_dependency_logs() -> None:
    for logger_name in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "huggingface",
        "transformers",
        "sentence_transformers",
        "urllib3",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
