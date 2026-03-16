import os
from pathlib import Path

from crimsonvector.dataloader.SQLDocumentLoader import SQLDocumentLoader
from crimsonvector.dataloader.utils import save_langchain_documents


CONNECTION_STRING = os.getenv(
    "CRIMSONVECTOR_SQL_URL",
    "postgresql://myuser:mypassword@localhost:5432/mydatabase",
)
OUTPUT_PATH = Path("data/processed/sql_documents.jsonl")


def main() -> None:
    loader = SQLDocumentLoader(
        connection_string=CONNECTION_STRING,
        table="documents",
        content_columns=["title", "content"],
        metadata_columns=["id", "source", "created_at"],
    )
    documents = loader.get_data()
    save_langchain_documents(documents, OUTPUT_PATH)

    print(f"Saved {len(documents)} processed SQL documents to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
