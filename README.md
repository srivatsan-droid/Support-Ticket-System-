# TaskFlow Support RAG

FastAPI support assistant over a 40-entry fictional TaskFlow FAQ corpus. Retrieval uses Qdrant cosine search, an optional indexed category filter, a documented lexical reranker, and a cosine relevance cutoff. Answers stream as newline-delimited JSON (NDJSON); the final event follows the `QueryResponse` Pydantic schema.

## Run locally

Requires Python 3.11+, Docker, and an API key for the OpenAI-compatible Groq endpoint used by the default configuration.

```sh
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your provider key to OPENAI_API_KEY in .env
docker compose up -d qdrant
python ingest.py
python raw_search.py
uvicorn main:app --reload
```

The model downloads on first use. `python ingest.py` may be run repeatedly: source entries are replaced by ID and deterministic UUID5 chunk IDs prevent duplicate points. To start with a fresh collection, remove the local Qdrant volume while Qdrant is stopped, then restart it and ingest again.

## API

OpenAPI docs are at `http://localhost:8000/docs`.

```sh
curl -N -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"How do I reset my password?"}'
```

The `/query` NDJSON stream starts with a `sources` event (`retrieved_sources`: chunks given to the model), emits one or more `delta` events, and ends with `final`, whose `answer` and `cited_sources` form a `QueryResponse`. `cited_sources` are the IDs the model actually cited in `[brackets]`, intersected with the retrieved IDs. If no result meets `MIN_RELEVANCE`, it emits the fixed abstention and does not call the LLM. `/ingest` accepts `id`, `title`, `body`, `category`, and ISO `last_updated` and uses the same idempotent ingestion path.

## Review artifacts

- Corpus and the three deliberately unanswerable questions: `corpus/faqs.json`, `corpus/corpus.md`.
- Retrieval comparison: `retrieval-notes.md`.
- Evaluation inputs and offline retrieval scorer: `evaluation/questions.json`, `evaluate.py`, `evaluation.md`.
- Tests: `pytest`.

`python evaluate.py` scores the real pipeline against live Qdrant (context recall/precision, abstention); `--generate --judge` adds citation precision and LLM-judged faithfulness. The pytest suite uses an in-memory Qdrant and an offline stand-in encoder, so retrieval, filtering, idempotency and the abstention gate run for real; only the LLM is mocked. Score calibration against MiniLM is checked by `python evaluate.py --check`, not by pytest.
