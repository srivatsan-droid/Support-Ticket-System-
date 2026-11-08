# Retrieval notes: before/after

## What the pipeline adds over plain top-k vector search
1. **Wider candidate pool** (12) instead of top-3, so reranking has something to reorder.
2. **Lexical-bonus rerank**: `cosine + 0.08 * query-token overlap` (light stemming, stop-words removed).
   *Why this and not a hosted/cross-encoder reranker:* 40 entries, no extra latency or API dependency, fully explainable.
   *Limitation:* it only reorders what the vector search already returned, so it is **not** true hybrid search.
3. **Two-tier relevance gate** (abstention): reject below `MIN_RELEVANCE`; accept above `HIGH_CONFIDENCE`; in between require lexical overlap >= `MIN_LEXICAL_OVERLAP`.
4. **Optional category filter** (indexed payload field), used by callers who know the category. It is *not* part of the before/after below, because supplying the right category is an oracle the real user doesn't have.

## Correction to the previous version of this file
The earlier note claimed that a `category="integrations"` filter surfaced `faq-034`. `faq-034` is in category **bugs**, so that filter can never return it, and the "before" results were never measured. That claim has been removed.

## Reproducible before/after
Run (Qdrant up, corpus ingested):

```sh
python retrieval_compare.py "My integration keeps throwing errors when syncing events."
```

Pick a query where BEFORE is poor and AFTER is better, and paste the **unedited** output here:

```
<paste output of retrieval_compare.py>
```

Then state in two sentences which signal changed the ranking (cosine vs. overlap columns are printed for exactly this purpose). If no query shows a genuine improvement with the 0.08 weight, say so and consider a cross-encoder or Qdrant sparse vectors for real hybrid search.

## Known gate weakness
"What is the airspeed velocity of an unladen swallow in TaskFlow's enterprise settings?" can clear the mid-score tier on overlap of the in-vocabulary words *enterprise* and *settings* (observed against `faq-032` with the offline test encoder). Confirm on real embeddings with `python evaluate.py --check`.
