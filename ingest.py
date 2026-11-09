"""Load TaskFlow FAQ entries, chunk them, and upsert deterministic points into Qdrant.

Chunking choice (defended): FAQ/ticket entries are short (~160-230 characters), far
below the embedding model's ~256-token window. Splitting them would separate the
question from its answer, so a normal entry is exactly ONE chunk and overlap is
irrelevant. Because /ingest accepts arbitrary bodies, long bodies fall back to a
word-boundary sliding window (MAX_CHUNK_CHARS with CHUNK_OVERLAP_CHARS overlap),
and the title is prefixed to every chunk so each one stays self-describing.

Idempotency: point IDs are UUID5(entry_id, chunk_index), so re-running overwrites
instead of duplicating. Stale chunks of an edited entry, and entries removed from
the corpus file, are deleted after the new points are written (upsert first, delete
second, so an entry is never missing from the index mid-run).
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer


load_dotenv()

logger = logging.getLogger(__name__)

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "taskflow_support")
ENTRY_NAMESPACE = uuid.UUID("f44ad2a7-bafb-490f-8e14-73dc5ed96084")
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", 800))  # ~200 tokens, under MiniLM's 256 limit
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", 100))

client = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", 6333)),
)


class LazyEncoder:
    """Defer model download/loading until an embedding is actually requested."""
    _model = None

    def _get_model(self):
        if self._model is None:
            self._model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
        return self._model

    def encode(self, *args, **kwargs):
        return self._get_model().encode(*args, **kwargs)

    def get_sentence_embedding_dimension(self):
        return self._get_model().get_sentence_embedding_dimension()


encoder = LazyEncoder()


def clean_text(text: str) -> str:
    """Normalize whitespace while retaining useful punctuation and words."""
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, max_chars: int | None = None, overlap: int | None = None) -> list[str]:
    """One chunk for normal entries; word-boundary sliding window for oversized text."""
    max_chars = MAX_CHUNK_CHARS if max_chars is None else max_chars
    overlap = CHUNK_OVERLAP_CHARS if overlap is None else overlap
    cleaned = clean_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]
    overlap = min(overlap, max_chars // 2)  # guarantees forward progress
    words, chunks, start = cleaned.split(" "), [], 0
    while start < len(words):
        end, length = start, 0
        while end < len(words) and length + len(words[end]) + 1 <= max_chars:
            length += len(words[end]) + 1
            end += 1
        end = max(end, start + 1)  # a single over-long word still becomes a chunk
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        back, back_len = end, 0  # step back ~overlap characters for context continuity
        while back > start + 1 and back_len + len(words[back - 1]) + 1 <= overlap:
            back -= 1
            back_len += len(words[back]) + 1
        start = back
    return chunks


def validate_entry(entry: dict[str, Any]) -> dict[str, str]:
    required = ("id", "title", "body", "category", "last_updated")
    if not isinstance(entry, dict):
        raise ValueError("Each entry must be a JSON object.")
    result: dict[str, str] = {}
    for field in required:
        value = entry.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Entry field '{field}' is required and must be non-empty text.")
        result[field] = value.strip()
    try:
        date.fromisoformat(result["last_updated"])
    except ValueError as exc:
        raise ValueError("Entry field 'last_updated' must use YYYY-MM-DD format.") from exc
    return result


def _point_id(entry_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(ENTRY_NAMESPACE, f"{entry_id}:chunk:{chunk_index}"))


def init_collection() -> None:
    names = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=encoder.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE,
            ),
        )
    # Re-creating an existing index is a no-op; this also repairs older collections.
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="category",
        field_schema=models.PayloadSchemaType.KEYWORD,
        wait=True,
    )


def _build_chunks(valid: dict[str, str]) -> list[str]:
    prefix = f"Question: {valid['title']} Answer: "
    # Chunk the body only, then re-attach the title so every chunk is self-describing.
    return [clean_text(prefix + part) for part in chunk_text(valid["body"])]


def upsert_entries(entries: list[dict[str, Any]]) -> int:
    """Embed all entries in one batch, upsert once, then drop stale chunks. Idempotent."""
    valid_entries = [validate_entry(e) for e in entries]
    per_entry = [(v, _build_chunks(v)) for v in valid_entries]
    flat = [chunk for _, chunks in per_entry for chunk in chunks]
    if not flat:
        return 0
    vectors = iter(encoder.encode(flat).tolist())  # single batched encode call
    points: list[models.PointStruct] = []
    keep_ids: dict[str, list[str]] = {}
    for valid, chunks in per_entry:
        for index, chunk in enumerate(chunks):
            point_id = _point_id(valid["id"], index)
            keep_ids.setdefault(valid["id"], []).append(point_id)
            points.append(models.PointStruct(
                id=point_id,
                vector=next(vectors),
                payload={
                    "entry_id": valid["id"],
                    "title": valid["title"],
                    "category": valid["category"],
                    "last_updated": valid["last_updated"],
                    "chunk_text": chunk,
                    "chunk_index": index,
                },
            ))
    client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
    # Upsert first, delete second: remove only chunk IDs that this version no longer produces.
    for entry_id, ids in keep_ids.items():
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(filter=models.Filter(
                must=[models.FieldCondition(key="entry_id", match=models.MatchValue(value=entry_id))],
                must_not=[models.HasIdCondition(has_id=ids)],
            )),
            wait=True,
        )
    return len(points)


def upsert_entry(entry: dict[str, Any]) -> int:
    """Replace all chunks for one source entry; repeat calls are idempotent."""
    return upsert_entries([entry])


def remove_orphans(valid_ids: set[str]) -> int:
    """Delete points whose entry_id no longer exists in the corpus file."""
    orphans: set[str] = set()
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=COLLECTION_NAME, limit=256, offset=offset,
            with_payload=["entry_id"], with_vectors=False,
        )
        orphans.update(r.payload["entry_id"] for r in records if r.payload["entry_id"] not in valid_ids)
        if offset is None:
            break
    if orphans:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(filter=models.Filter(
                must=[models.FieldCondition(key="entry_id", match=models.MatchAny(any=sorted(orphans)))]
            )),
            wait=True,
        )
    return len(orphans)


def load_entries(file_path: str | Path) -> list[dict[str, str]]:
    try:
        raw = json.loads(Path(file_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read corpus JSON: {exc}") from exc
    if not isinstance(raw, list):
        raise ValueError("Corpus file must contain a JSON array.")
    entries = [validate_entry(item) for item in raw]
    ids = [entry["id"] for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Corpus entry IDs must be unique.")
    return entries


def ingest_corpus(file_path: str | Path = "corpus/faqs.json", prune: bool = True) -> int:
    entries = load_entries(file_path)
    init_collection()
    total = upsert_entries(entries)
    removed = remove_orphans({e["id"] for e in entries}) if prune else 0
    print(f"Upserted {len(entries)} entries as {total} chunks into '{COLLECTION_NAME}'"
          f" (removed {removed} orphaned entries).")
    return total


if __name__ == "__main__":
    ingest_corpus()