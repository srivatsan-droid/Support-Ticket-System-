"""Before/after on any query: raw vector top-3 vs. gated vector vs. the shipped hybrid pipeline.

Usage: python retrieval_compare.py "my integration keeps throwing errors when syncing events"
Paste the printed output into retrieval-notes.md; do not write results by hand.
Columns: cosine = embedding similarity, bm25 = keyword rank (None = not in BM25 pool),
vec = vector rank, ov = IDF-weighted lexical overlap used by the gate.
"""
import sys

from retrieval import COLLECTION_NAME, client, encoder, get_index, lexical_overlap, retrieve, vector_search

DEFAULT_QUERY = "My integration keeps throwing errors when syncing events."


def show(label: str, hits: list, query: str, index) -> None:
    print(f"\n{label}:")
    if not hits:
        print("  (nothing above the relevance gate -> abstain)")
    for hit in hits:
        p = hit.payload or {}
        print(f"  {p.get('entry_id')} [{p.get('category')}] cosine={hit.score:.4f} "
              f"vec={getattr(hit, 'vector_rank', None)} bm25={getattr(hit, 'lexical_rank', None)} "
              f"ov={lexical_overlap(query, hit, index):.2f}  {p.get('title')}")


def main(query: str) -> None:
    index = get_index(client, COLLECTION_NAME)
    print(f"Query: {query!r}")
    show("BEFORE 1 - raw vector top 3 (no gate)", vector_search(encoder.encode(query).tolist(), 3), query, index)
    show("BEFORE 2 - vector + gate (hybrid=False)", retrieve(query, hybrid=False), query, index)
    show("AFTER    - hybrid BM25+vector (RRF) + gate", retrieve(query), query, index)


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]) or DEFAULT_QUERY)