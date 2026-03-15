from crimsonvector.dataloader.FixedDocumentLoader import FixedDocumentLoader as DocumentLoader
from crimsonvector.dataloader.utils import save_langchain_documents, load_langchain_documents

def main():
    loader = DocumentLoader(
    path="data/processed",
    chunk_size=2000,
    chunk_overlap=200,)

    docs_processed = loader.get_data()

    save_langchain_documents(docs_processed, "documents/output.jsonl")



if __name__ == "__main__":
    main()
