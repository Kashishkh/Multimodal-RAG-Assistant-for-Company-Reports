# Report Chat — Multimodal RAG (FastAPI + Streamlit)

Upload any PDF report, then ask questions. Text, tables, charts, diagrams and
scanned pages are all searchable; answers cite pages and show the page image.

## Run
```bash
cp .env.example .env            # secrets only: GROQ_API_KEY, PINECONE_API_KEY
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt -r frontend/requirements.txt

# terminal 1 (backend)
cd backend && uvicorn app.main:app --reload --port 8000
# terminal 2 (frontend)
cd frontend && streamlit run app.py
```
API docs: http://localhost:8000/docs

## Architecture
Streamlit → FastAPI → (BackgroundTasks ingestion) → Pinecone (namespace = user, filter = doc_id)
SQLite = doc status · local disk = PDFs/images (swap for Postgres + S3 later).

## Endpoints
POST /documents · GET /documents · GET /documents/{id} · DELETE /documents/{id}
GET /documents/{id}/pages/{n} · POST /chat · GET /health
(all except /health need an `X-User-Id` header)

## Next steps
Reranker + hybrid search · table→pandas for numeric questions · RAGAS eval set ·
Langfuse tracing · JWT auth · Postgres/S3 · Docker · SSE streaming · Next.js UI

## Secrets vs settings
- `.env` - API keys only. **Gitignored, never commit.** `.env.example` is the committed template.
- `backend/settings.toml` - non-secret config (limits, embedding model, **the list of selectable chat models**).
- If a key was ever committed, revoke it and create a new one - deleting the file is not enough.
- Deploying: set the same two variables in the host's environment settings instead of using a file.
- Requires Python 3.11+ (uses `tomllib`).

## Adding / changing chat models
Edit the `[[models]]` blocks in `backend/settings.toml` (any Groq model id). The UI picks them up automatically.
