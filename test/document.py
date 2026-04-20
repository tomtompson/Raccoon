from pathlib import Path

from raccoon.dataloader.FixedDocumentLoader import FixedDocumentLoader


SOURCE_PATH = Path("data/raw")
OUTPUT_PATH = Path("data/processed/chunks/")


def main() -> None:
    loader = FixedDocumentLoader(
        path=SOURCE_PATH,
        chunk_size=4000,
        chunk_overlap=200,
    )
    documents_parents , documents_child = loader.get_data()
    loader.save_chunked_documents(OUTPUT_PATH / "parent_chunks.json", OUTPUT_PATH / "child_chunks.json")

    print(f"Saved {len(documents_parents)} parent and {len(documents_child)} child documents to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
