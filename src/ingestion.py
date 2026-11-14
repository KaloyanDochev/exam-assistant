import os
from dotenv import load_dotenv
from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DOCUMENTS_DIRECTORY = Path(os.getenv("DOCUMENTS_DIRECTORY", "documents"))


def load_pdf(pdf_path: str | Path) -> list[Document]:
    """Extract text from a PDF, creating one Document per non-empty page."""

    path = Path(pdf_path)
    documents = []

    with pymupdf.open(str(path)) as pdf:

        for page_number, page in enumerate(pdf, start=1):

            page_text = page.get_text("text", sort=True).strip()

            if not page_text:
                continue

            documents.append(
                Document(page_content=page_text,
                         metadata={
                             "source": path.name,
                             "page": page_number
                         }))

    return documents


def load_pdfs(directory: str | Path = DOCUMENTS_DIRECTORY) -> list[Document]:
    """Load all PDFs recursively from the documents directory."""

    directory = Path(directory)
    all_documents = []

    pdf_paths = sorted(path for path in directory.rglob("*")
                       if path.is_file() and path.suffix.lower() == ".pdf")

    for pdf_path in pdf_paths:

        documents = load_pdf(pdf_path)

        if not documents:
            print(f"No extractable text found in {pdf_path.name}. Skipping.")
            continue

        all_documents.extend(documents)

    return all_documents


def chunk_documents(documents: list[Document],
                    chunk_size: int = 1500,
                    chunk_overlap: int = 200) -> list[Document]:
    """Split documents into smaller, overlapping chunks."""

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)

    return text_splitter.split_documents(documents)


if __name__ == "__main__":

    documents = load_pdfs()
    chunks = chunk_documents(documents)

    print(f"Loaded {len(documents)} pages.")
    print(f"Created {len(chunks)} chunks.")

    for chunk in chunks[:6]:
        print(f"\nSource: {chunk.metadata['source']}")
        print(f"Page: {chunk.metadata['page']}")
        print(chunk.page_content[:500])
