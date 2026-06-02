from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from raccoon.dataloader import SQLDocumentLoader


def _sqlite_url(tmp_path: Path) -> str:
    db_path = tmp_path / "documents.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE documents (id INTEGER, title TEXT, body TEXT, status TEXT)"))
        conn.execute(
            text("INSERT INTO documents VALUES (:id, :title, :body, :status)"),
            [
                {"id": 1, "title": "One", "body": "First body", "status": "keep"},
                {"id": 2, "title": "Two", "body": "Second body", "status": "skip"},
            ],
        )
    return f"sqlite:///{db_path}"


def test_sql_document_loader_uses_bound_where_params(tmp_path):
    loader = SQLDocumentLoader(
        connection_string=_sqlite_url(tmp_path),
        table="documents",
        content_columns=["title", "body"],
        metadata_columns=["id"],
        where="status = :status",
        where_params={"status": "keep"},
        chunk_size=100,
        chunk_overlap=10,
    )

    parent_docs, child_docs = loader.get_data()

    assert len(parent_docs) == 1
    assert child_docs
    assert "One" in parent_docs[0].page_content
    assert parent_docs[0].metadata["source_id"] == 1
    assert parent_docs[0].metadata["doc_type"] == "parent"


def test_sql_document_loader_rejects_unsafe_identifiers(tmp_path):
    with pytest.raises(ValueError, match="Invalid SQL identifier"):
        SQLDocumentLoader(
            connection_string=_sqlite_url(tmp_path),
            table="documents; DROP TABLE documents",
            content_columns=["title"],
        )
