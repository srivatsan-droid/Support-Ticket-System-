"""Shared fixtures: in-memory Qdrant + a deterministic offline encoder (no model download)."""
import re
import zlib

import numpy as np
import pytest
from qdrant_client import QdrantClient

import ingest
import retrieval

DIM = 256


class HashingEncoder:
    """Bag-of-stemmed-words hashing encoder. Deterministic and offline.

    It stands in for MiniLM so tests exercise the REAL ingest/retrieve/gate code paths.
    It does not reproduce MiniLM's score calibration; that is evaluate.py's job.
    """

    def get_sentence_embedding_dimension(self):
        return DIM

    def _vec(self, text: str) -> np.ndarray:
        vec = np.zeros(DIM, dtype=np.float32)
        for token in retrieval._tokens(text):
            vec[zlib.crc32(token.encode()) % DIM] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def encode(self, texts, **_):
        if isinstance(texts, str):
            return self._vec(texts)
        return np.stack([self._vec(t) for t in texts])


@pytest.fixture()
def memory_qdrant(monkeypatch):
    memory, encoder = QdrantClient(":memory:"), HashingEncoder()
    for module in (ingest, retrieval):
        monkeypatch.setattr(module, "client", memory)
        monkeypatch.setattr(module, "encoder", encoder)
    # Sparse bag-of-words cosines run lower than MiniLM's (a 2-word query vs a ~20-token
    # entry scores ~0.3), so the stand-in needs its own cutoff. Real thresholds are
    # validated against MiniLM by `python evaluate.py --check`, not by unit tests.
    monkeypatch.setattr(retrieval, "MIN_RELEVANCE", 0.25)
    ingest.init_collection()
    return memory
