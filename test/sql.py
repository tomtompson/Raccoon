import os
from pathlib import Path

from raccoon.dataloader.SQLDocumentLoader import SQLDocumentLoader


CONNECTION_STRING = os.getenv(
    "RACCOON_SQL_URL",
    "postgresql://myuser:mypassword@localhost:5432/mydatabase",
)
OUTPUT_PATH = Path("data/processed/chunks_sql/")


def main() -> None:
    loader = SQLDocumentLoader(
        connection_string=CONNECTION_STRING,
        table="documents",
        content_columns=["title", "content"],
        metadata_columns=["id", "source", "created_at"],
    )
    parent_documents, child_documents = loader.get_data()
    loader.save_chunked_documents(OUTPUT_PATH / "parent_chunks.json", OUTPUT_PATH / "child_chunks.json")

    print(f"Saved {len(parent_documents)} parent and {len(child_documents)} child documents to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
