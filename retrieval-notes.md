# Retrieval notes: before/after

## What the pipeline does (and what changed)
| Stage | Before (v2.1) | Now (hybrid) |
|---|---|---|
| Candidate generation | Vector top-12 only | Vector top-12 **plus** BM25 top-12 over the *whole* collection (`lexical.py`) |
| Ordering | `cosine + 0.08 * overlap` re-sort of the 12 vector hits | Reciprocal Rank Fusion (`k=60`) of the two rankings |
| Gate | cosine tiers + plain word-overlap | Same tiers, **IDF-weighted** overlap, plus a *rescue* tier for lexical hits |
| Category filter | Qdrant payload filter | Same filter applied to **both** sides |

**Why the old rerank was not enough.** It could only re-order what vector search already returned, so an entry the
embedding model ranked 13th or lower could never surface. BM25 scores every chunk independently, so a rare keyword
(an error code, an acronym) finds its entry regardless of embedding quality.

**Why BM25 + RRF instead of a hosted cross-encoder.** 40 chunks, no extra latency, API key, or cost, and every score is
explainable in the review. RRF needs no score calibration between cosine and BM25 (it fuses ranks, not scores).
Trade-off: a cross-encoder understands paraphrase better; at ~40k chunks I would move BM25 into Qdrant sparse vectors
and add a cross-encoder over the fused top-20.

**Why the gate still uses cosine.** The fused score only says "ranked high in the lists", not "relevant". For lexical-only
hits the true cosine is fetched from Qdrant and the gate decides on that. A keyword match below `RESCUE_MIN_COSINE`
(0.25) is never accepted, so word overlap cannot rescue a semantically unrelated chunk.

## Before/after (reproducible, deterministic, runs in CI)
Test: `test_hybrid_surfaces_entry_that_plain_vector_search_misses` (offline hashing encoder, in-memory Qdrant).
Corpus: one entry whose only strong signal is the code `ERR4021`, buried among 30 look-alike "Error report N" entries.
Query: `what is error ERR4021`.

```
BEFORE  vector top-12 pool contains 'target'?  -> False
BEFORE  retrieve(hybrid=False)                 -> []            (abstains: right answer never retrieved)
AFTER   retrieve()                             -> [('target', cosine=0.18, sources=['bm25'])]
```
Which signal changed the result: `target` is BM25 rank 1 (the rare token `err4021` has IDF 3.06 vs 0.05 for `error`) but is
not in the vector pool at all. Fusion adds it to the shortlist, the true cosine (0.18) clears the rescue floor, and the
IDF-weighted overlap (0.98) clears the rescue overlap bar.

Honest limits of this evidence:
* It is a synthetic regression test, not a measurement on MiniLM. The 40-entry FAQ corpus is small enough that
  MiniLM already ranks most answers in its top 12, so real-embedding gains are expected to be modest and are
  **not yet measured**.
* To measure on the real model (Qdrant up, corpus ingested) and paste the unedited output below:
  `python retrieval_compare.py "<query>"` prints raw vector, gated vector, and hybrid side by side with `vec`/`bm25`
  ranks. Good candidates are short keyword queries ("2FA", "CSV backup", "SAML", "webhook retries").

```
<paste output of python retrieval_compare.py "<your query>" once run against real embeddings>
```

## Earlier finding kept for the record
`test_runs/03_retrieval_compare.txt` (old rerank) showed the pipeline *removing* hits relative to raw top-3 rather than
recovering a missed one, e.g. "I can't log in, locked out of my account" re-ordered `faq-003` above `faq-001`. That is why
the lexical side was rebuilt as an independent BM25 retriever instead of a score bonus. (An earlier version of this file
claimed `category="integrations"` surfaced `faq-034`; `faq-034` is in `bugs`, so that claim was wrong and removed.)

## Known gate weakness: resolved
The "airspeed velocity of an unladen swallow in TaskFlow's enterprise settings" question previously leaked because plain
overlap counted `enterprise` and `settings` as 2/6 matches. IDF weighting makes those common words cheap and the four
absent words (`airspeed`, `velocity`, `unladen`, `swallow`) expensive, so overlap drops below 0.20 and the question
abstains. The test is no longer `xfail`. Re-check on MiniLM with `python evaluate.py --check` (thresholds were tuned
for MiniLM and a change to the gate must be re-verified, see below).