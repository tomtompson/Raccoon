from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document


def save_langchain_documents(
    documents: Iterable[Document],
    output_path: str | Path,
    ensure_ascii: bool = False,
    pretty: bool = False,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for doc in documents:
            record = {
                "page_content": doc.page_content,
                "metadata": _make_json_safe(doc.metadata),
            }

            if pretty:
                f.write(json.dumps(record, ensure_ascii=ensure_ascii, indent=2))
                f.write("\n")
            else:
                f.write(json.dumps(record, ensure_ascii=ensure_ascii))
                f.write("\n")


def load_langchain_documents(input_path: str | Path) -> list[Document]:
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    documents: list[Document] = []

    with input_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                documents.append(
                    Document(
                        page_content=record["page_content"],
                        metadata=record.get("metadata", {}),
                    )
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to parse line {line_number} in {input_path}: {e}"
                ) from e

    return documents


def save_synthesized_results(
    rows: Iterable[dict],
    output_path: str | Path,
    ensure_ascii: bool = False,
    pretty: bool = False,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            record = _make_json_safe(dict(row))
            if pretty:
                f.write(json.dumps(record, ensure_ascii=ensure_ascii, indent=2))
                f.write("\n")
            else:
                f.write(json.dumps(record, ensure_ascii=ensure_ascii))
                f.write("\n")

def load_synthesized_results(input_path: str | Path) -> list[dict]:
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    rows: list[dict] = []

    with input_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rows.append(dict(json.loads(line)))
            except Exception as e:
                raise ValueError(
                    f"Failed to parse line {line_number} in {input_path}: {e}"
                ) from e

    return rows


def _make_json_safe(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_make_json_safe(v) for v in obj]
    return str(obj)
