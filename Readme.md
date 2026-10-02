# RAG AI Learning Project

This repository is a hands-on learning project focused on building a Retrieval-Augmented Generation (RAG) pipeline using Python and LangChain.

It was created to understand the end-to-end workflow of:

- loading documents
- splitting text into chunks
- creating embeddings
- storing vectors in a vector database
- retrieving relevant context from the database
- sending that context to an LLM for answer generation

This project is educational and experimental, not a production-ready application.

## Project Goals

The main purpose of this repo is to practice and understand:

- PDF document loading with LangChain
- text chunking strategies
- embeddings and vector databases
- similarity search
- retrievers for external sources like ArXiv
- basic RAG architecture and retrieval flow

## Tech Stack

- Python 3.12+
- LangChain
- LangChain Community
- Chroma / ChromaDB
- PyPDF
- ArXiv retriever
- Python-dotenv
- NVIDIA AI endpoints (used in some learning examples)

## Repository Structure

```text
rag-ai/
├── .env.example
├── .gitignore
├── notes.md
├── pyproject.toml
├── requirements.txt
├── src/
│   └── rag_project/
│       ├── __init__.py
│       ├── create_database.py
│       ├── document-loaders/
│       │   ├── main.py
│       │   ├── main copy.py
│       │   ├── notes.txt
│       │   ├── page.py
│       │   ├── pdf.py
│       │   ├── sample.txt
│       │   ├── test.py
│       │   └── test1.py
│       └── retrievers/
│           └── arixv.py
└── vector_store/
    └── DB.py
```

## Learning Workflow Covered

### 1. Document Loading

The project experiments with PDF and text loading using LangChain loaders such as:

- `PyPDFLoader`
- `TextLoader`

### 2. Chunking

Large documents are split into smaller chunks to reduce context-window issues and improve retrieval relevance.

Examples use:

- `RecursiveCharacterTextSplitter`
- `TokenTextSplitter`
- `CharacterTextSplitter`

### 3. Embeddings

The project creates embeddings for text chunks using LangChain embedding integrations.

This helps transform raw text into vector representations that can be compared semantically.

### 4. Vector Store

The repository stores chunk embeddings in Chroma to support fast similarity search.

This is the core database layer used in a typical RAG system.

### 5. Retrieval

The repo includes experiments with an ArXiv retriever to fetch relevant research papers based on a query.

This helps understand how retrievers work with external sources.

## Main Scripts

### `src/rag_project/create_database.py`

This script demonstrates creating a vector database from documents by:

- loading PDFs
- splitting text into chunks
- generating embeddings
- storing them in Chroma

### `src/rag_project/retrievers/arixv.py`

This script shows how to query the ArXiv retriever and print the top matching results.

### `src/rag_project/document-loaders/*.py`

These are smaller learning scripts that explore:

- PDF processing
- text splitting strategies
- prompt creation
- LLM invocation
- basic experimentation with RAG pipeline components

## Setup

1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

or with uv if preferred:

```bash
uv sync
```

3. Configure environment variables

Copy the example file if needed:

```bash
cp .env.example .env
```

Then add required API keys or services for your experiments.

## Notes

This repo contains notes and experiments from the learning process. It may include:

- incomplete scripts
- exploratory code
- commented examples
- temporary files for understanding concepts

That is intentional and part of the learning workflow.

## Important Reminder

This project is meant to understand the architecture behind RAG systems and is not meant to be treated as a polished production application.

## Useful Learning References

- LangChain document loaders
- Text splitters and chunking
- Embedding models
- Vector databases like Chroma
- Similarity search and ANN concepts
- Retriever-based systems
- LLM + retrieved context prompt pipelines

## Future Improvements

Possible next steps for this learning project include:

- adding a proper RAG query interface
- building a streamlit app
- adding a local FAISS or Chroma-based retrieval app
- connecting with OpenAI or other LLM providers
- cleaning up experimental files into a single structured project

## License

This project is for educational purposes and learning experimentation.
