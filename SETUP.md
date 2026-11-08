# Setup and verification guide

Use these steps from the project directory. First run will download the embedding model; allow a few minutes and ensure Hugging Face is reachable.

## 1. Create the environment

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY` to your Groq API key (the default LLM endpoint is Groq's OpenAI-compatible API). Keep `.env` local; it is ignored by Git. Qdrant and embedding/retrieval tests do not need this key.

## 2. Start Qdrant and build the collection

```sh
docker compose up -d qdrant
curl http://localhost:6333/healthz
python ingest.py
```

The first ingestion creates `taskflow_support` and the category payload index. It should report 40 entries and 40 chunks. Run `python ingest.py` again; the point count must stay unchanged. Verify the count directly:

```sh
curl -s -X POST http://localhost:6333/collections/taskflow_support/points/count \
  -H 'Content-Type: application/json' -d '{"exact":true}'
```

The JSON `result.count` should be 40 after each run. `python raw_search.py` prints the top three raw similarity results with their payload metadata.

## 3. Run automated checks and evaluation

```sh
pytest -q
python evaluate.py --label before --check
```

The pytest suite is offline: it uses an in-memory Qdrant and a stand-in encoder (no Docker, no model download) and mocks only the LLM. `evaluate.py` needs Qdrant running with the corpus ingested; it uses the real embedding model and writes `evaluation/results_<label>.json`. Copy the numbers from that file into `evaluation.md`.

## 4. Start and exercise the service

```sh
uvicorn main:app --reload
```

Open `http://localhost:8000/docs` and confirm `POST /query`, `POST /ingest`, and `GET /health` are listed. A streamed answer can also be checked with:

```sh
curl -N -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"How do I reset my password?"}'
```

Expect NDJSON events: one `sources`, one or more `delta`, and a final `final` event containing `answer` and `cited_sources`. Try the three questions in `corpus/corpus.md`; each should produce the fixed abstention and no cited sources. Empty/whitespace questions and incomplete `/ingest` bodies should return 4xx JSON errors.

Test live ingestion with an entry whose `id` is not already in the corpus:

```sh
curl -i -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"id":"demo-001","title":"Demo question","body":"Demo answer.","category":"account","last_updated":"2026-11-08"}'
```

It should return 201. Repeating the same request should replace that source's points rather than increase the count. Stop services with `docker compose down`.
