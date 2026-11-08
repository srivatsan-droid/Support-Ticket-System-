"""Vector retrieval with optional metadata filtering, lexical reranking, and abstention.

Pipeline: vector search (wide candidate pool, optional category filter) ->
lexical-bonus rerank -> relevance gate.

Gate design (two tiers, so paraphrases are not rejected for lacking shared words):
  * score >= HIGH_CONFIDENCE                -> accept on semantics alone
  * MIN_RELEVANCE <= score < HIGH_CONFIDENCE -> accept only with lexical evidence
  * score < MIN_RELEVANCE                   -> reject
The gate always uses the ORIGINAL cosine score, never the boosted rerank score, so a
keyword match cannot rescue a semantically unrelated result.

NOTE: all three thresholds were set for all-MiniLM-L6-v2 and must be re-calibrated
(run evaluate.py) whenever the embedding model or corpus changes.
"""
from __future__ import annotations

import os
import re
from typing import Optional

from qdrant_client.http import models

from ingest import COLLECTION_NAME, client, encoder

MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.42"))
HIGH_CONFIDENCE = float(os.getenv("HIGH_CONFIDENCE", "0.55"))
MIN_LEXICAL_OVERLAP = float(os.getenv("MIN_LEXICAL_OVERLAP", "0.20"))
RERANK_WEIGHT = 0.08

GENERIC_STOP_WORDS = {
    "a", "an", "and", "are", "can", "do", "does", "for", "how", "i", "in", "is", "it",
    "my", "of", "on", "the", "to", "what", "when", "where", "why", "with", "you",
}
# Words that appear in nearly every entry of THIS product's corpus, so overlap on them
# is not evidence of relevance. Corpus-specific by design; revisit if the corpus changes.
DOMAIN_STOP_WORDS = {"taskflow", "connect", "connecting", "connection", "connections"}
STOP_WORDS = GENERIC_STOP_WORDS | DOMAIN_STOP_WORDS


def _stem(word: str) -> str:
    """Tiny suffix stripper so 'integration'/'integrations', 'sync'/'syncing' overlap."""
    for suffix in ("ing", "ions", "ion", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _tokens(text: str) -> set[str]:
    return {
        _stem(word)
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if len(word) > 1 and word not in STOP_WORDS
    }


def lexical_overlap(query: str, result) -> float:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    payload = result.payload or {}
    text_tokens = _tokens(f"{payload.get('title', '')} {payload.get('chunk_text', '')}")
    return len(query_tokens & text_tokens) / len(query_tokens)


def keyword_boost_rerank(query: str, results: list, top_k: int = 3) -> list:
    """Re-order the vector candidates with a small lexical bonus (no extra model).

    Chosen over a hosted/cross-encoder reranker because the corpus is tiny, it adds no
    latency or API dependency, and it is fully explainable. Limitation: it can only
    re-order what vector search already returned.
    """
    if not _tokens(query):
        return sorted(results, key=lambda r: r.score, reverse=True)[:top_k]
    ranked = sorted(
        results,
        key=lambda r: r.score + RERANK_WEIGHT * lexical_overlap(query, r),
        reverse=True,
    )
    return ranked[:top_k]


def passes_gate(query: str, result, min_relevance: float | None = None) -> bool:
    cutoff = MIN_RELEVANCE if min_relevance is None else min_relevance
    if result.score < cutoff:
        return False
    if result.score >= max(HIGH_CONFIDENCE, cutoff):
        return True
    return lexical_overlap(query, result) >= MIN_LEXICAL_OVERLAP


def vector_search(query_vector: list[float], limit: int, query_filter=None) -> list:
    """Use query_points on newer qdrant-client, fall back to search on the pinned 1.8.x."""
    if hasattr(client, "query_points"):
        return client.query_points(
            collection_name=COLLECTION_NAME, query=query_vector, query_filter=query_filter,
            limit=limit, with_payload=True,
        ).points
    return client.search(
        collection_name=COLLECTION_NAME, query_vector=query_vector, query_filter=query_filter,
        limit=limit, with_payload=True,
    )


def retrieve(query: str, category: Optional[str] = None, top_k: int = 3,
             min_relevance: float | None = None) -> list:
    query_vector = encoder.encode(query).tolist()
    query_filter = None
    if category:
        query_filter = models.Filter(
            must=[models.FieldCondition(key="category", match=models.MatchValue(value=category))]
        )
    candidates = vector_search(query_vector, max(top_k * 4, 12), query_filter)
    ranked = keyword_boost_rerank(query, candidates, top_k=top_k)
    return [r for r in ranked if passes_gate(query, r, min_relevance)]
