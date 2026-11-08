"""Part 3 deliverable: one raw similarity search, top 3 results with full payload."""
import json

from ingest import COLLECTION_NAME, client, encoder
from retrieval import vector_search


def run_raw_search(query: str = "How do I reset my password?") -> None:
    results = vector_search(encoder.encode(query).tolist(), limit=3)
    print(f"Query: {query!r}\nTop 3 results from '{COLLECTION_NAME}':")
    for rank, hit in enumerate(results, 1):
        print(f"\n{rank}. score={hit.score:.4f}")
        print(json.dumps(hit.payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run_raw_search()
