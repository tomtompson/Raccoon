from typing import List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .BaseLoader import BaseLoader


class SQLDocumentLoader(BaseLoader):

    def __init__(
        self,
        connection_string: str,
        table: str,
        content_columns: List[str],
        metadata_columns: Optional[List[str]] = None,
        where: Optional[str] = None,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        super().__init__()

        self.connection_string = connection_string
        self.table = table
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns or []
        self.where = where
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parent_data: List[Document] | None = None
        self.child_data: List[Document] | None = None

        self.engine: Engine = create_engine(connection_string)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", ".", " ", ""],
            add_start_index=True,
        )

    def load_data(self) -> List[Document]:

        columns = self.content_columns + self.metadata_columns
        column_string = ", ".join(columns)

        query = f"SELECT {column_string} FROM {self.table}"

        if self.where:
            query += f" WHERE {self.where}"

        documents: List[Document] = []

        with self.engine.connect() as conn:

            result = conn.execute(text(query))

            for row in result:
                row_dict = dict(row._mapping)

                # combine content fields
                content = "\n\n".join(
                    str(row_dict[col])
                    for col in self.content_columns
                    if row_dict.get(col) is not None
                )

                metadata = {
                    col: row_dict.get(col)
                    for col in self.metadata_columns
                }

                documents.append(
                    Document(
                        page_content=content,
                        metadata=metadata
                    )
                )

        return documents

    def preprocess_data(self, data: List[Document]) -> List[Document]:
        parent_docs , child_docs = self.chunk(data, self.chunk_size, self.chunk_overlap)
        self.child_data = child_docs
        self.parent_data = parent_docs
        return parent_docs, child_docs
