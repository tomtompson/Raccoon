import os
from pathlib import Path
from typing import Iterable

from .BaseLoader import BaseLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader, 
    )



class FixedDocumentLoader(BaseLoader):
    SUPPORTED_EXTENSIONS = {".txt", ".docx", ".pdf"}

    def __init__(
        self,
        path: str | Path,
        config: dict | None = None,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
        recursive: bool = True,
        silent_errors: bool = True,
        text_encoding: str = "utf-8",
    ):
        super().__init__(config=config)
        self.path = Path(path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ".", " ", ""]
        self.recursive = recursive
        self.silent_errors = silent_errors
        self.text_encoding = text_encoding

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
            separators=self.separators,
        )

    def load_data(self) -> list[Document]:
        if not self.path.exists():
            raise FileNotFoundError(f"Path does not exist: {self.path}")

        if self.path.is_file():
            if self.path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported file type: {self.path.suffix}")
            documents = self._load_single_file(self.path)
        else:
            documents = self._load_directory(self.path)

        self.data = documents
        return documents

    def preprocess_data(self, data: list[Document]) -> list[Document]:
        return self.text_splitter.split_documents(data)

    def _load_single_file(self, file_path: Path) -> list[Document]:
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path), show_progress=True)
        elif suffix == ".docx":
            loader = Docx2txtLoader(str(file_path), show_progress=True)
        elif suffix == ".txt":
            loader = TextLoader(str(file_path), encoding=self.text_encoding, show_progress=True)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        return loader.load()

    def _load_directory(self, directory_path: Path) -> list[Document]:
        documents: list[Document] = []

        loader_configs = [
            {
                "glob": "**/*.pdf" if self.recursive else "*.pdf",
                "loader_cls": PyPDFLoader,
            },
            {
                "glob": "**/*.docx" if self.recursive else "*.docx",
                "loader_cls": Docx2txtLoader,
            },
            {
                "glob": "**/*.txt" if self.recursive else "*.txt",
                "loader_cls": TextLoader,
                "loader_kwargs": {"encoding": self.text_encoding},
            },
        ]

        for cfg in loader_configs:
            loader = DirectoryLoader(
                str(directory_path),
                glob=cfg["glob"],
                loader_cls=cfg["loader_cls"],
                loader_kwargs=cfg.get("loader_kwargs", {}),
                silent_errors=self.silent_errors,
                recursive=False,  
                show_progress=True,
            )
            documents.extend(loader.load())

        return documents