import chromadb
from chromadb.utils import embedding_functions
from langchain_core.documents import Document

from vector_store import CHROMA_PATH, COLLECTION_NAME


def retrieve_chunks(query: str,
                    max_candidates: int = 20,
                    distance_threshold: float | None = 1.5,
                    persist_directory=CHROMA_PATH) -> list[Document]:
    """Retrieve chunks whose distance is below the specified threshold."""

    client = chromadb.PersistentClient(path=str(persist_directory))

    embedding_function = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_collection(name=COLLECTION_NAME,
                                       embedding_function=embedding_function)

    results = collection.query(query_texts=[query],
                               n_results=min(max_candidates,
                                             collection.count()),
                               include=["documents", "metadatas", "distances"])

    documents = []

    for text, metadata, distance in zip(results["documents"][0],
                                        results["metadatas"][0],
                                        results["distances"][0]):

        if distance_threshold is not None and distance > distance_threshold:
            continue

        documents.append(
            Document(page_content=text,
                     metadata={
                         **metadata, "distance": distance
                     }))

    return documents


if __name__ == "__main__":

    question = "What is the epsilon-delta definition of a limit?"

    chunks = retrieve_chunks(question)

    print(f"Question: {question}")
    print(f"\nRetrieved {len(chunks)} chunks:\n")

    for i, chunk in enumerate(chunks, start=1):
        print(f"--- Result {i} ---")
        print(f"Source: {chunk.metadata['source']}")
        print(f"Page: {chunk.metadata['page']}")
        print(f"Distance: {chunk.metadata['distance']:.4f}")
        print(chunk.page_content)
        print()
