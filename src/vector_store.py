import os
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions
from langchain_core.documents import Document


load_dotenv()

CHROMA_PATH = Path(
    os.getenv("VECTOR_DB_DIRECTORY", "chromadb")
)
COLLECTION_NAME = "exam_materials"


def store_chunks(
    chunks: list[Document],
    persist_directory: str | Path = CHROMA_PATH
):
    """Generate embeddings and store chunks in ChromaDB."""

    client = chromadb.PersistentClient(path=str(persist_directory))

    embedding_function = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

    collection.upsert(
        ids=[str(i) for i in range(len(chunks))],
        documents=[chunk.page_content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks]
    )

    return collection


if __name__ == "__main__":

    from ingestion import load_pdfs, chunk_documents

    documents = load_pdfs()
    chunks = chunk_documents(documents)

    collection = store_chunks(chunks)

    print(f"Stored {collection.count()} chunks in ChromaDB.")