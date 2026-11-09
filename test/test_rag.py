import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import generator
import ingest
import retrieval
from main import app

client = TestClient(app)

UNANSWERABLE = [
    "Can I connect my TaskFlow account to a Nintendo Switch or PlayStation console?",
    "How do I bake a sourdough loaf of bread using the mobile app timer?",
    "What is the airspeed velocity of an unladen swallow in TaskFlow's enterprise settings?",
    # In-domain but absent from the corpus: the harder abstention case.
    "Can I change my billing currency from USD to EUR?",
    "Does TaskFlow have a Gantt chart view?",
]


def _ndjson(response):
    return [json.loads(line) for line in response.text.splitlines()]


def _count(qdrant):
    return qdrant.count(collection_name=ingest.COLLECTION_NAME, exact=True).count


def _entry(entry_id="test-1", **overrides):
    base = {"id": entry_id, "title": "How?", "body": "Do this.", "category": "account",
            "last_updated": "2026-01-01"}
    return {**base, **overrides}


# ---------- validation ----------
def test_empty_question_is_clean_4xx():
    assert client.post("/query", json={"question": "   "}).status_code == 422


def test_malformed_ingest_request_is_clean_4xx():
    response = client.post("/ingest", json={"id": "bad-entry"})
    assert response.status_code == 422
    assert "traceback" not in response.text.lower()


def test_ingest_rejects_invalid_date():
    assert client.post("/ingest", json=_entry(last_updated="not-a-date")).status_code == 422


# ---------- ingestion (real Qdrant semantics, in memory) ----------
def test_ingest_corpus_is_idempotent_and_prunes_orphans(memory_qdrant, tmp_path):
    corpus = tmp_path / "faqs.json"
    corpus.write_text(json.dumps([_entry("a"), _entry("b", title="Other")]))
    ingest.ingest_corpus(corpus)
    first = _count(memory_qdrant)
    ingest.ingest_corpus(corpus)
    assert _count(memory_qdrant) == first == 2  # re-run: no duplicates
    corpus.write_text(json.dumps([_entry("a")]))  # entry "b" removed from the file
    ingest.ingest_corpus(corpus)
    assert _count(memory_qdrant) == 1  # orphan pruned


def test_edited_entry_replaces_old_chunks(memory_qdrant):
    long_body = " ".join(f"word{i}" for i in range(600))  # forces several chunks
    assert ingest.upsert_entry(_entry(body=long_body)) > 1
    assert ingest.upsert_entry(_entry(body="Short now.")) == 1
    assert _count(memory_qdrant) == 1  # stale chunks removed


def test_chunks_carry_required_metadata(memory_qdrant):
    ingest.upsert_entry(_entry(title="Reset password", category="billing"))
    payload = memory_qdrant.scroll(ingest.COLLECTION_NAME, limit=1, with_payload=True)[0][0].payload
    assert {"entry_id", "title", "category", "last_updated"} <= payload.keys()
    assert payload["entry_id"] == "test-1" and payload["category"] == "billing"


def test_chunker_short_entry_is_single_chunk_and_long_has_overlap():
    assert ingest.chunk_text("short answer") == ["short answer"]
    words = [f"w{i}" for i in range(400)]
    chunks = ingest.chunk_text(" ".join(words), max_chars=200, overlap=40)
    assert len(chunks) > 1 and all(len(c) <= 200 for c in chunks)
    assert set(chunks[0].split()) & set(chunks[1].split())  # overlap present


# ---------- retrieval ----------
def test_category_filter_restricts_results(memory_qdrant):
    ingest.upsert_entries([
        _entry("acct", title="Reset password", body="Use the forgot password link.", category="account"),
        _entry("bill", title="Reset password invoice", body="Password for invoices.", category="billing"),
    ])
    both = {r.payload["entry_id"] for r in retrieval.retrieve("reset password", top_k=5, min_relevance=0.0)}
    only = {r.payload["entry_id"] for r in
            retrieval.retrieve("reset password", category="billing", top_k=5, min_relevance=0.0)}
    assert both == {"acct", "bill"} and only == {"bill"}


def test_gate_accepts_high_confidence_without_lexical_overlap():
    hit = SimpleNamespace(score=0.8, payload={"title": "Refund policy", "chunk_text": "money returned"})
    assert retrieval.passes_gate("zzz qqq", hit)
    weak = SimpleNamespace(score=0.4, payload={"title": "Refund policy", "chunk_text": "money returned"})
    assert not retrieval.passes_gate("zzz qqq", weak)  # mid-score needs lexical evidence
    assert not retrieval.passes_gate("refund", SimpleNamespace(score=0.2, payload=weak.payload))


# ---------- abstention (retrieval is REAL; only the LLM is mocked) ----------
@pytest.fixture()
def loaded_corpus(memory_qdrant):
    ingest.upsert_entries(ingest.load_entries("corpus/faqs.json"))


@pytest.mark.parametrize("question", UNANSWERABLE)
def test_unanswerable_questions_abstain_without_llm(loaded_corpus, question):
    with patch("main.generate_support_answer") as generate:
        response = client.post("/query", json={"question": question})
    assert response.status_code == 200
    events = _ndjson(response)
    assert events[-1] == {"type": "final", "answer": generator.ABSTENTION, "cited_sources": []}
    generate.assert_not_called()


def test_answerable_question_is_retrieved(loaded_corpus):
    ids = [r.payload["entry_id"] for r in retrieval.retrieve("How do I reset my password?")]
    assert "faq-001" in ids


# ---------- streaming + citations ----------
def test_final_event_reports_only_ids_actually_cited():
    hits = [SimpleNamespace(payload={"entry_id": "faq-001"}), SimpleNamespace(payload={"entry_id": "faq-002"})]
    with patch("main.retrieve", return_value=hits), patch(
        "main.generate_support_answer", return_value=iter(["Click Forgot Password ", "[faq-001]. ", "See [faq-999]."])
    ):
        response = client.post("/query", json={"question": "How do I reset my password?"})
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = _ndjson(response)
    assert events[0] == {"type": "sources", "retrieved_sources": ["faq-001", "faq-002"]}
    assert events[-1]["cited_sources"] == ["faq-001"]  # faq-002 not cited, faq-999 hallucinated


def test_generation_error_is_reported_as_error_event():
    hit = SimpleNamespace(payload={"entry_id": "faq-001"})
    with patch("main.retrieve", return_value=[hit]), patch("main.generate_support_answer", side_effect=RuntimeError("x")):
        events = _ndjson(client.post("/query", json={"question": "Reset?"}))
    assert events[-1]["type"] == "error" and "RuntimeError" not in json.dumps(events[-1])


def test_extract_cited_ids_handles_formats():
    allowed = ["faq-001", "faq-0011", "faq-002"]
    assert generator.extract_cited_ids("A [faq-001, faq-002].", allowed) == ["faq-001", "faq-002"]
    assert generator.extract_cited_ids("A [Source ID: faq-0011].", allowed) == ["faq-0011"]
    assert generator.extract_cited_ids(generator.ABSTENTION, allowed) == []


# ---------- generator (OpenAI client mocked) ----------
def test_generator_streams_and_skips_empty_choice_chunks(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    chunk = lambda t: SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=t))])
    stream = [chunk("Hi "), SimpleNamespace(choices=[]), chunk(None), chunk("there [faq-001].")]
    fake = MagicMock()
    fake.chat.completions.create.return_value = iter(stream)
    hit = SimpleNamespace(payload={"entry_id": "faq-001", "title": "T", "chunk_text": "body"})
    with patch("generator.OpenAI", return_value=fake):
        assert "".join(generator.generate_support_answer("q", [hit])) == "Hi there [faq-001]."
    assert "answer only" in fake.chat.completions.create.call_args.kwargs["messages"][0]["content"].lower()


# ---------- evaluation scorer (pure logic, no model) ----------
def test_score_retrieval_metrics_and_leak_reporting():
    import evaluate
    cases = [
        {"question": "q1", "expected_source_ids": ["a", "b"], "answerable": True},
        {"question": "q2", "expected_source_ids": [], "answerable": False},
        {"question": "q3", "expected_source_ids": [], "answerable": False},
    ]
    fake = {"q1": ["a", "x"], "q2": [], "q3": ["z"]}
    result = evaluate.score_retrieval(cases, fake.__getitem__)
    assert result["context_recall"] == 0.5 and result["context_precision"] == 0.5
    assert result["abstention_rate"] == 0.5
    assert result["leaked_unanswerables"] == [{"question": "q3", "leaked_ids": ["z"]}]


# ---------- hybrid retrieval (Part 4) ----------
def _hybrid_corpus():
    """One entry whose only strong signal is a rare code, buried among 30 look-alike noise entries."""
    target = _entry("target", title="Sync stalls", category="bugs",
                    body="Code ERR4021 means the calendar token expired. Reauthorize the calendar under Settings.")
    noise = [_entry(f"noise-{i}", title=f"Error report {i}", category="bugs",
                    body="An error happened. Please try again later.") for i in range(30)]
    return [target, *noise]


def test_hybrid_surfaces_entry_that_plain_vector_search_misses(memory_qdrant):
    ingest.upsert_entries(_hybrid_corpus())
    query = "what is error ERR4021"
    query_vector = retrieval.encoder.encode(query).tolist()
    pool = retrieval.vector_search(query_vector, 12)
    assert "target" not in {h.payload["entry_id"] for h in pool}  # BEFORE: not even a candidate
    assert retrieval.retrieve(query, hybrid=False) == []          # BEFORE: plain search abstains
    hits = retrieval.retrieve(query)                              # AFTER: BM25 finds it
    assert [h.payload["entry_id"] for h in hits] == ["target"]
    assert hits[0].sources == ["bm25"] and hits[0].lexical_rank == 1


def test_lexical_only_hit_keeps_its_true_cosine_score(memory_qdrant):
    ingest.upsert_entries(_hybrid_corpus())
    query = "what is error ERR4021"
    hit = retrieval.retrieve(query)[0]
    vec = retrieval.encoder.encode(query)
    stored = memory_qdrant.retrieve(ingest.COLLECTION_NAME, ids=[hit.id], with_vectors=True)[0].vector
    assert hit.score == pytest.approx(float(vec @ stored), abs=1e-5)  # gate sees real cosine, not RRF


def test_rescue_tier_needs_both_cosine_floor_and_strong_overlap():
    p = {"title": "Webhook failures", "chunk_text": "webhook delivery retries"}
    low_cos = SimpleNamespace(score=0.10, payload=p)   # below RESCUE_MIN_COSINE: never rescued
    mid_cos = SimpleNamespace(score=0.30, payload=p)   # between rescue floor and MIN_RELEVANCE
    assert not retrieval.passes_gate("webhook delivery retries", low_cos)
    assert retrieval.passes_gate("webhook delivery retries", mid_cos)       # overlap 1.0 >= 0.6
    assert not retrieval.passes_gate("webhook totally unrelated words", mid_cos)  # overlap too low


def test_idf_overlap_penalises_terms_absent_from_the_corpus():
    index = retrieval.LexicalIndex({
        "1": {"title": "Enterprise settings", "chunk_text": "Open settings for the enterprise plan."},
        "2": {"title": "Settings", "chunk_text": "Change settings in the app."},
        "3": {"title": "Billing", "chunk_text": "Pay by card in settings."},
    })
    payload = index.payloads["1"]
    plain = index.weighted_overlap("enterprise settings", payload)
    padded = index.weighted_overlap("airspeed velocity unladen swallow enterprise settings", payload)
    assert plain > 0.9 and padded < 0.35  # unseen words count fully against the match


def test_category_filter_applies_to_the_bm25_side_too(memory_qdrant):
    ingest.upsert_entries([
        _entry("acct", title="Reset password", body="Use the forgot password link.", category="account"),
        _entry("bill", title="Reset password invoice", body="Password for invoices.", category="billing"),
    ])
    only = {h.payload["entry_id"] for h in
            retrieval.retrieve("password invoice", category="account", top_k=5, min_relevance=0.0)}
    assert only <= {"acct"}


# ---------- index freshness + live ingest ----------
def test_ingest_endpoint_entry_is_immediately_retrievable(memory_qdrant):
    ingest.upsert_entries(_hybrid_corpus())
    assert retrieval.retrieve("dark mode appearance") == []  # primes (and caches) the BM25 index
    response = client.post("/ingest", json=_entry("demo-001", title="Is there dark mode?",
                           body="Yes. Open Settings > Appearance and choose Dark.", category="account"))
    assert response.status_code == 201
    with patch("main.generate_support_answer", return_value=iter(["Use Appearance [demo-001]."])):
        events = _ndjson(client.post("/query", json={"question": "dark mode appearance"}))
    assert "demo-001" in events[0]["retrieved_sources"]  # index was invalidated by the write
    assert events[-1]["cited_sources"] == ["demo-001"]


def test_retrieval_failure_is_a_clean_503():
    with patch("main.retrieve", side_effect=RuntimeError("qdrant down")):
        response = client.post("/query", json={"question": "anything"})
    assert response.status_code == 503 and "qdrant down" not in response.text