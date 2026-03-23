from pathlib import Path

from crimsonvector.dataloader.FixedDocumentLoader import FixedDocumentLoader
from crimsonvector.dataloader.utils import save_langchain_documents


SOURCE_PATH = Path("data/raw")
OUTPUT_PATH = Path("data/processed/documents.jsonl")


def main() -> None:
    loader = FixedDocumentLoader(
        path=SOURCE_PATH,
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = loader.get_data()
    save_langchain_documents(documents, OUTPUT_PATH)

    print(f"Saved {len(documents)} processed documents to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
