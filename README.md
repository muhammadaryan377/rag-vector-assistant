# RAG Vector Assistant

A portfolio-ready **Retrieval-Augmented Generation (RAG)** project built with an **LLM + vector database**.

The API lets you upload PDF/TXT/Markdown documents, converts them into embeddings, stores them in **PostgreSQL + pgvector**, retrieves the most relevant chunks for a question, and sends only that retrieved context to an LLM to produce a grounded answer with citations.

## Tech stack

- Python
- FastAPI
- PostgreSQL
- pgvector
- Sentence Transformers
- Groq LLM API
- Docker
- Pytest

## Architecture

```text
PDF / TXT / MD
      |
      v
 Text Extraction
      |
      v
   Chunking
      |
      v
Sentence-Transformer Embeddings
      |
      v
PostgreSQL + pgvector
      |
User Question
      |
      v
 Query Embedding
      |
      v
Cosine Similarity Search
      |
      v
Top-k Relevant Chunks
      |
      v
      LLM
      |
      v
Grounded Answer + Citations
```

## Why this project is useful

This is more than a basic chatbot. It demonstrates the core engineering behind modern AI knowledge assistants:

- document ingestion
- text chunking
- embeddings
- vector similarity search
- HNSW vector indexing
- context construction
- LLM grounding
- source citations
- API design
- environment-based secret management

## Project structure

```text
rag-vector-assistant/
├── app/
│   ├── config.py
│   ├── db.py
│   ├── embeddings.py
│   ├── ingestion.py
│   ├── llm.py
│   ├── main.py
│   ├── rag.py
│   └── schemas.py
├── data/
│   └── sample.txt
├── tests/
│   └── test_chunking.py
├── .env.example
├── .gitignore
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Run locally

### 1. Start PostgreSQL + pgvector

```bash
docker compose up -d
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install packages

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env`.

```bash
cp .env.example .env
```

On Windows you can simply duplicate the file and rename the copy to `.env`.

Add your Groq API key and set `GROQ_MODEL` to a chat model currently available in your Groq account.

### 5. Run the API

```bash
uvicorn app.main:app --reload
```

Open the interactive API docs at:

```text
http://127.0.0.1:8000/docs
```

## API usage

### Health check

```bash
curl http://127.0.0.1:8000/health
```

### Upload a document

```bash
curl -X POST "http://127.0.0.1:8000/documents" \
  -F "file=@data/sample.txt"
```

Example response:

```json
{
  "source": "sample.txt",
  "chunks_inserted": 1
}
```

### Ask a question

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What is RAG?\",\"top_k\":5}"
```

Example response shape:

```json
{
  "answer": "RAG retrieves relevant information before asking the LLM to answer [1].",
  "sources": [
    {
      "citation": "[1]",
      "source": "sample.txt",
      "chunk_index": 0,
      "score": 0.8123
    }
  ]
}
```

## Run tests

```bash
pytest
```

## Resume / portfolio description

**RAG Vector Assistant** — Built a document question-answering API using FastAPI, sentence-transformer embeddings, PostgreSQL/pgvector semantic retrieval, and an LLM. Implemented document ingestion, chunking, HNSW vector search, grounded generation, and source citations.

## Good next upgrades

- hybrid BM25 + vector retrieval
- reranking
- conversation history
- metadata filters
- authentication
- streaming responses
- evaluation with Recall@K / MRR / NDCG
- simple React or Next.js frontend

## Security

Never commit `.env` or API keys to GitHub. The included `.gitignore` already excludes `.env`.
