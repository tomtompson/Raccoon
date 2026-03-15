from crimsonvector.dataloader.SQLDocumentLoader import SQLDocumentLoader 
from crimsonvector.dataloader.utils import save_langchain_documents, load_langchain_documents
def main():
    loader = SQLDocumentLoader(
    connection_string="postgresql://myuser:mypassword@localhost:5432/mydatabase",
    table="documents",
    content_columns=["title", "content"],
    metadata_columns=["id", "source", "created_at"]
    )

    docs_processed = loader.get_data()

    save_langchain_documents(docs_processed, "data/processed/output.jsonl")



if __name__ == "__main__":
    main()
