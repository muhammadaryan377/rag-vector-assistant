# RAG Vector Assistant

I built this project to practice how a complete RAG pipeline works with an LLM and a vector database.

The idea is simple: upload a document, split it into chunks, create embeddings, store them in PostgreSQL with pgvector, retrieve the most relevant chunks for a question, and then send that context to the LLM so the answer stays grounded in the uploaded data.

## Tech I used

- Python
- FastAPI
- PostgreSQL
- pgvector
- Sentence Transformers
- Groq API
- Docker
- Pytest

## How it works

```text
Document
   ↓
Text extraction
   ↓
Chunking
   ↓
Embeddings
   ↓
PostgreSQL + pgvector
   ↓
User question
   ↓
Query embedding
   ↓
Vector similarity search
   ↓
Top relevant chunks
   ↓
LLM
   ↓
Answer with sources
```

## Main features

- Upload PDF, TXT, and Markdown files
- Split document text into smaller chunks
- Create embeddings using Sentence Transformers
- Store embeddings in PostgreSQL using pgvector
- Search relevant chunks with cosine similarity
- Use an HNSW index for vector search
- Send retrieved context to the LLM
- Return the answer with source references
- FastAPI endpoints for document upload and question answering

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

## Run the project

Start PostgreSQL with pgvector:

```bash
docker compose up -d
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file from `.env.example` and add your Groq API key and model name.

Then run the API:

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Example flow

First upload a document using the `/documents` endpoint.

Then ask a question using `/ask`:

```json
{
  "question": "What is RAG?",
  "top_k": 5
}
```

The system retrieves the closest matching chunks from pgvector and uses them as context for the LLM.

## Testing

```bash
pytest
```

## What I learned from this project

This project helped me understand the full RAG flow instead of only calling an LLM API. I worked with document ingestion, chunking, embeddings, vector similarity search, retrieval, context building, API design, and grounded generation.

## Next improvements

- Hybrid search using BM25 + vectors
- Reranking
- Conversation memory
- Metadata filtering
- Streaming responses
- RAG evaluation using Recall@K, MRR, and NDCG
- Simple frontend with React or Next.js

## Note

Do not commit your `.env` file or API keys to GitHub.