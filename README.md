# Consiva AI Assistant

A RAG chatbot that answers questions about Consiva.ai and the DPDP Act using the content of consiva.ai.

```
consiva.ai ──scrape──> data/raw ──clean──> data/new_structured ──compare──> changes.json
                                                                                │
             data/vectorstore (Chroma) <──chunk + embed (bge-m3)── vector update ┘
                         │
                 snapshot.npz ──> FastAPI (/ask) ──> OpenAI (or Gemini) ──> React chat widget
```

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env     # then fill in OPENAI_API_KEY and FIRECRAWL_API_KEY
```

## Run

```powershell
# API on http://localhost:8000  (GET /health, POST /ask)
uvicorn app.main:app --reload

# Chat widget on http://localhost:5173 (proxies /api to the API)
cd frontend
npm install
npm run dev
```

Docker: `docker compose up --build` (API on 8000, widget on 5173; `data/` and the model cache are volumes).

`POST /ask` body: `{"question": "...", "history": [{"role": "user"|"assistant", "content": "..."}]}`
Response: `{"answer": "... [1]", "sources": [{"id": 1, "title", "section", "url"}], "query_type"}`.
Errors: 400 bad input, 429 rate limit, 503 knowledge base or LLM unavailable.

## Nightly pipeline

`run_pipeline.bat` runs `ingestion/pipeline.py`:
scraping → cleaning → comparison → vector update → manifest update → baseline promotion.

- Only changed chunks are embedded; IDs are content hashes.
- Changing chunking or cleaning? Bump `CHUNKER_VERSION` in `config.py`; the next run re-indexes every page.
- After a successful update the API reloads its index automatically (`data/vectorstore/index_version.json`).
- Status: `data/pipeline_logs/latest_run.json` (`running`, `success`, `failed`, `interrupted`), history in `run_history.jsonl`, log in `pipeline.log` (rotated).
- Failure alerts: set `ALERT_WEBHOOK_URL` (Slack/Teams incoming webhook).
- Windows task settings: run `scripts\update_scheduled_task.ps1` once as administrator.

Recovery: `python ingestion/full_rebuild.py` rebuilds the whole vector store from `data/new_structured` into a new folder and swaps it in (the old one is kept as a backup).

## Quality checks

```powershell
python -m unittest discover -s tests -v          # fast unit tests, no network
python evals/run_eval.py                          # retrieval on 42 test questions
python evals/run_eval.py --answers --out evals/results/run.json   # + LLM answers
```

Run the eval before and after any change to cleaning, chunking, retrieval or the prompt.

## Configuration

All settings are in `config.py` and can be overridden in `.env` (see `.env.example`):
chunk sizes, `TOP_K`, `MIN_RELEVANCE` (the "not found" threshold), rate limit, allowed origins, `LLM_PROVIDER` and model.
