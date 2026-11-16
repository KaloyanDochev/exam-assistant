# ExamAssistant

A lightweight, AI-powered command-line (CLI) study assistant that helps students prepare for exams using their own PDF materials. It uses Retrieval-Augmented Generation (RAG) to find relevant information in study documents and generate contextual answers with source references.

## Features

- **PDF processing:** Extracts text from PDF files, including documents stored in subdirectories.
- **Semantic search:** Finds relevant passages using vector embeddings.
- **AI-generated answers:** Uses an OpenAI model to generate answers based on retrieved study material.
- **Source references:** Includes the original PDF filename and page number to help verify information.
- **Persistent storage:** Stores document embeddings locally in ChromaDB, allowing them to be reused.
- **Answer saving:** Optionally saves generated answers as Markdown files.
- **Configurable settings:** Allows users to customize directories and retrieval parameters through environment variables.

## How It Works

ExamAssistant follows a Retrieval-Augmented Generation (RAG) workflow:

1. **Document loading:** Extracts text from PDF files using PyMuPDF.
2. **Text chunking:** Splits extracted text into smaller, overlapping chunks.
3. **Embedding and storage:** Converts chunks into vector embeddings and stores them in ChromaDB.
4. **Retrieval:** Searches the vector database for chunks relevant to the user's question.
5. **Answer generation:** Sends the question and retrieved context to an OpenAI model, which generates an answer with source references.

## Technologies

- **Python** – Main programming language
- **PyMuPDF** – PDF text extraction
- **LangChain** – Document processing and LLM integration
- **ChromaDB** – Vector database and similarity search
- **OpenAI API** – AI-generated answers
- **Rich** – Markdown formatting in the terminal

## Project Structure

```text
ExamAssistant/
├── documents/             # Default directory for PDF study materials
├── chroma_db/             # Default directory for the vector database
├── answers/               # Default directory for saved answers
├── src/
│   ├── ingestion.py       # PDF loading and text chunking
│   ├── vector_store.py    # Embedding storage
│   ├── retrieval.py       # Relevant chunk retrieval
│   ├── generation.py      # Answer generation
│   └── main.py            # Application entry point
├── .env.example           # Example environment configuration
├── .gitignore
├── requirements.txt
└── README.md
```



## Configuration

Create a `.env` file in the project root using `.env.example` as a template. Application settings can be adjusted through the `.env` file:

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key | — |
| `OPENAI_MODEL` | OpenAI model used for answer generation | `gpt-6-luna` |
| `REASONING_EFFORT` | Model reasoning effort | `none` |
| `SAVE_ANSWERS` | Whether to save generated answers | `true` |
| `DOCUMENTS_DIRECTORY` | Directory containing PDF materials | `documents` |
| `ANSWER_DIRECTORY` | Directory for saved answers | `answers` |
| `VECTOR_DB_DIRECTORY` | Persistent ChromaDB directory | `chromadb` |
| `MAX_CANDIDATES` | Maximum number of chunks retrieved | `20` |
| `DISTANCE_THRESHOLD` | Maximum retrieval distance for accepted chunks | `1.5` |

## Running the Application

**1. Configure environment settings**

Create a `.env` file in the project root using `.env.example` as a template and set your OpenAI API key. Optionally, you can also adjust the other settings.


**2. Index the study materials**

Run vector_store.py in order to extract and chunk the PDF content, generate embeddings, and store them in ChromaDB. This step only needs to be performed once for a given set of study materials.

**3. Start the assistant**

You can now run main.py in order to enter questions in the terminal and receive answers based on your indexed study materials.

## Notes

- The first run may take longer because the embedding model needs to be downloaded.
- PDF files must contain extractable text. Scanned documents may require OCR before processing.
- The vector database is stored locally and persists between runs.
- When adding, removing, or replacing study materials, the vector database may need to be rebuilt to reflect the changes.
- Answers depend on the quality of the source documents and retrieval results. Always verify important information against the original materials.
