"""Before/after on any query: raw vector top-3 vs. the shipped pipeline (no oracle filter).

Usage: python retrieval_compare.py "my integration keeps throwing errors when syncing events"
Paste the printed output into retrieval-notes.md; do not write results by hand.
"""
import sys

from ingest import encoder
from retrieval import lexical_overlap, retrieve, vector_search

DEFAULT_QUERY = "My integration keeps throwing errors when syncing events."


def show(label: str, hits: list, query: str) -> None:
    print(f"\n{label}:")
    if not hits:
        print("  (nothing above the relevance gate -> abstain)")
    for hit in hits:
        p = hit.payload or {}
        print(f"  {p.get('entry_id')} [{p.get('category')}] cosine={hit.score:.4f} "
              f"overlap={lexical_overlap(query, hit):.2f}  {p.get('title')}")


def main(query: str) -> None:
    print(f"Query: {query!r}")
    show("BEFORE - raw vector top 3", vector_search(encoder.encode(query).tolist(), 3), query)
    show("AFTER  - rerank + gate (no category filter)", retrieve(query, top_k=3), query)


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]) or DEFAULT_QUERY)
