from __future__ import annotations

import math
import os
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from qdrant_client.http import models

from ingest import COLLECTION_NAME, client, encoder

MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.42"))
HIGH_CONFIDENCE = float(os.getenv("HIGH_CONFIDENCE", "0.55"))
MIN_LEXICAL_OVERLAP = float(os.getenv("MIN_LEXICAL_OVERLAP", "0.20"))
RESCUE_MIN_COSINE = float(os.getenv("RESCUE_MIN_COSINE", "0.25"))
RESCUE_MIN_OVERLAP = float(os.getenv("RESCUE_MIN_OVERLAP", "0.60"))
RRF_K = 60  # standard RRF constant; damps the influence of any single list's top rank

# ---------------------------------------------------------------------------------------------
# Lexical half of hybrid search: in-process BM25 over every chunk in the collection.
# Why: the old rerank only re-ordered the vector top-12, so it could never surface an entry the
# embedding ranked badly. BM25 scores the WHOLE collection, so a rare keyword (error code, acronym)
# still finds its entry. Scale note: this holds all chunk text in memory (fine for ~40-40,000 short
# chunks); beyond that move it into Qdrant sparse vectors and keep the same fusion code.
# Freshness: main.py's /ingest calls invalidate_index(); a TTL covers writes by another process
# (e.g. `python ingest.py` while the API is running).
# ---------------------------------------------------------------------------------------------
GENERIC_STOP_WORDS = {
    "a", "an", "and", "are", "can", "do", "does", "for", "how", "i", "in", "is", "it",
    "my", "of", "on", "the", "to", "what", "when", "where", "why", "with", "you",
}
# Words in nearly every entry of THIS product's corpus. With IDF weighting below they are already
# down-weighted, but dropping them also keeps BM25 scores clean. Revisit if the corpus changes.
DOMAIN_STOP_WORDS = {"taskflow", "connect", "connecting", "connection", "connections"}
STOP_WORDS = GENERIC_STOP_WORDS | DOMAIN_STOP_WORDS

K1, B = 1.5, 0.75
INDEX_TTL_SECONDS = 30.0


def _stem(word: str) -> str:
    """Tiny suffix stripper so 'integration'/'integrations', 'sync'/'syncing' overlap."""
    for suffix in ("ing", "ions", "ion", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    """Stemmed, stop-word-free tokens, KEEPING duplicates (BM25 needs term frequency)."""
    return [
        _stem(word)
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if len(word) > 1 and word not in STOP_WORDS
    ]


def _tokens(text: str) -> set[str]:
    """Distinct tokens (set form of tokenize); also used by the offline test encoder."""
    return set(tokenize(text))


class LexicalIndex:
    def __init__(self, docs: dict[str, dict]):
        """docs: point id -> payload. Only title + chunk_text are indexed."""
        self.payloads = docs
        self.tf: dict[str, Counter] = {}
        self.lengths: dict[str, int] = {}
        df: Counter = Counter()
        for pid, payload in docs.items():
            tokens = tokenize(f"{payload.get('title', '')} {payload.get('chunk_text', '')}")
            self.tf[pid] = Counter(tokens)
            self.lengths[pid] = len(tokens)
            df.update(set(tokens))
        self.n = len(docs)
        self.avg_len = (sum(self.lengths.values()) / self.n) if self.n else 0.0
        # Smoothed BM25 IDF (always > 0). Unseen terms get the maximum IDF: a query word that
        # appears nowhere in the corpus is the strongest sign the question is out of scope.
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.max_idf = math.log(1 + (self.n + 0.5) / 0.5) if self.n else 1.0

    def idf_of(self, token: str) -> float:
        return self.idf.get(token, self.max_idf)

    def search(self, query: str, limit: int, allowed_ids: set[str] | None = None) -> list[tuple[str, float]]:
        """Top-`limit` (point_id, bm25_score) with score > 0, best first."""
        terms = set(tokenize(query))
        if not terms or not self.n:
            return []
        scored = []
        for pid, tf in self.tf.items():
            if allowed_ids is not None and pid not in allowed_ids:
                continue
            norm = K1 * (1 - B + B * self.lengths[pid] / (self.avg_len or 1.0))
            score = sum(
                self.idf[t] * tf[t] * (K1 + 1) / (tf[t] + norm)
                for t in terms if t in tf
            )
            if score > 0:
                scored.append((pid, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def weighted_overlap(self, query: str, payload: dict) -> float:
        """IDF-weighted share of the query's terms present in the chunk (0..1).

        Unlike a plain word count, matching a common word ("settings") earns little, and every
        query word that is absent from the corpus ("airspeed", "unladen") counts fully against.
        """
        terms = set(tokenize(query))
        if not terms:
            return 0.0
        text_terms = set(tokenize(f"{payload.get('title', '')} {payload.get('chunk_text', '')}"))
        total = sum(self.idf_of(t) for t in terms)
        return sum(self.idf_of(t) for t in terms & text_terms) / total if total else 0.0


_lock = threading.Lock()
_cache: dict[str, tuple[float, LexicalIndex]] = {}


def invalidate_index() -> None:
    """Drop the cached BM25 index (called by /ingest after a write; a TTL covers other processes)."""
    with _lock:
        _cache.clear()


def get_index(client, collection: str) -> LexicalIndex:
    """Build (or reuse) the index by scrolling payloads from Qdrant."""
    with _lock:
        hit = _cache.get(collection)
        if hit and time.monotonic() - hit[0] < INDEX_TTL_SECONDS:
            return hit[1]
        docs: dict[str, dict] = {}
        offset = None
        while True:
            records, offset = client.scroll(
                collection_name=collection, limit=256, offset=offset,
                with_payload=True, with_vectors=False,
            )
            docs.update({str(r.id): (r.payload or {}) for r in records})
            if offset is None:
                break
        index = LexicalIndex(docs)
        _cache[collection] = (time.monotonic(), index)
        return index


@dataclass
class Hit:
    """One retrieved chunk. `score` is ALWAYS the true cosine; `fused` is the ranking score."""
    id: str
    score: float
    payload: dict
    fused: float = 0.0
    vector_rank: Optional[int] = None
    lexical_rank: Optional[int] = None
    sources: list[str] = field(default_factory=list)


def lexical_overlap(query: str, result, index: LexicalIndex | None = None) -> float:
    """Share of query terms found in the chunk; IDF-weighted when an index is supplied."""
    payload = result.payload or {}
    if index is not None:
        return index.weighted_overlap(query, payload)
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    text_tokens = _tokens(f"{payload.get('title', '')} {payload.get('chunk_text', '')}")
    return len(query_tokens & text_tokens) / len(query_tokens)


def passes_gate(query: str, result, min_relevance: float | None = None,
                index: LexicalIndex | None = None) -> bool:
    cutoff = MIN_RELEVANCE if min_relevance is None else min_relevance
    overlap = lexical_overlap(query, result, index)
    if result.score >= max(HIGH_CONFIDENCE, cutoff):
        return True
    if result.score >= cutoff:
        return overlap >= MIN_LEXICAL_OVERLAP
    return result.score >= RESCUE_MIN_COSINE and overlap >= RESCUE_MIN_OVERLAP


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


def _category_filter(category: Optional[str]):
    if not category:
        return None
    return models.Filter(
        must=[models.FieldCondition(key="category", match=models.MatchValue(value=category))]
    )


def _cosines_for(point_ids: list[str], query_vector: list[float]) -> dict[str, float]:
    """True cosine for points that only the lexical side found (their score isn't known yet)."""
    if not point_ids:
        return {}
    q = np.asarray(query_vector, dtype=np.float32)
    q_norm = np.linalg.norm(q) or 1.0
    out: dict[str, float] = {}
    for record in client.retrieve(COLLECTION_NAME, ids=point_ids, with_vectors=True, with_payload=False):
        v = np.asarray(record.vector, dtype=np.float32)
        out[str(record.id)] = float(q @ v / (q_norm * (np.linalg.norm(v) or 1.0)))
    return out


def hybrid_candidates(query: str, query_vector: list[float], category: Optional[str],
                      pool: int, index: LexicalIndex) -> list[Hit]:
    """Fuse vector and BM25 rankings with RRF. Returns hits best-first with true cosine scores."""
    vec_hits = vector_search(query_vector, pool, _category_filter(category))
    allowed = None
    if category:
        allowed = {pid for pid, p in index.payloads.items() if p.get("category") == category}
    lex_hits = index.search(query, pool, allowed)

    hits: dict[str, Hit] = {}
    for rank, h in enumerate(vec_hits, 1):
        pid = str(h.id)
        hits[pid] = Hit(id=pid, score=float(h.score), payload=h.payload or {}, vector_rank=rank)
    missing = [pid for pid, _ in lex_hits if pid not in hits]
    cosines = _cosines_for(missing, query_vector)
    for rank, (pid, _) in enumerate(lex_hits, 1):
        if pid in hits:
            hits[pid].lexical_rank = rank
        elif pid in cosines:
            hits[pid] = Hit(id=pid, score=cosines[pid], payload=index.payloads[pid], lexical_rank=rank)
    for hit in hits.values():
        if hit.vector_rank:
            hit.fused += 1.0 / (RRF_K + hit.vector_rank)
            hit.sources.append("vector")
        if hit.lexical_rank:
            hit.fused += 1.0 / (RRF_K + hit.lexical_rank)
            hit.sources.append("bm25")
    return sorted(hits.values(), key=lambda h: h.fused, reverse=True)


def retrieve(query: str, category: Optional[str] = None, top_k: int = 3,
             min_relevance: float | None = None, hybrid: bool = True) -> list[Hit]:
    """hybrid=False reproduces plain gated vector search, for before/after comparisons."""
    query_vector = encoder.encode(query).tolist()
    pool = max(top_k * 4, 12)
    index = get_index(client, COLLECTION_NAME)
    if hybrid:
        ranked = hybrid_candidates(query, query_vector, category, pool, index)
    else:
        ranked = [
            Hit(id=str(h.id), score=float(h.score), payload=h.payload or {}, vector_rank=i)
            for i, h in enumerate(vector_search(query_vector, pool, _category_filter(category)), 1)
        ]
    # Ranking decides ORDER, the gate decides RELEVANCE: gate the whole fused pool, then keep top_k.
    # (Truncating first would let high-consensus but irrelevant chunks fill the slots and then be
    # rejected, hiding a lexical hit sitting just below them.) Both modes use the same rule so
    # hybrid=False vs hybrid=True differ only in how candidates are found and ordered.
    return [h for h in ranked if passes_gate(query, h, min_relevance, index)][:top_k]