# Evaluation

Set: `evaluation/questions.json`, 14 questions (9 answerable with expected source IDs, 5 unanswerable, 2 of them
in-domain). `evaluate.py` scores the shipped pipeline (`retrieval.retrieve` against live Qdrant), not a re-implementation.
Embeddings: all-MiniLM-L6-v2. Generator and judge: `openai/gpt-oss-20b` via Groq.

## Metrics
| Metric | Layer it diagnoses |
|---|---|
| context_recall | candidate generation (did we find the right entries?) |
| context_precision | ranking + gate (did we avoid noise?) |
| abstention_rate | relevance gate (unanswerables return nothing) |
| citation_precision | prompt / generation (did the model cite the right IDs?) |
| faithfulness (LLM judge) | generation (are all claims supported by context?) |

## Round 1: recorded results (real MiniLM + Groq; source: `evaluation/results_before.json`, `results_after.json`)
| Metric | Raw vector top-3 | Before | After |
|---|---:|---:|---:|
| context_recall | 1.000 | 1.000 | 1.000 |
| context_precision | 0.444 | 0.759 | **0.907** |
| abstention_rate | 0.000 | 0.600 | **1.000** |
| leaked unanswerables | 5 | 2 | **0** |
| citation_precision | n/a | 0.944 | 1.000 |
| faithfulness (judge) | n/a | 1.000 | 1.000 |

**Weakest metric before:** abstention_rate (0.60). The two leaks were "Nintendo Switch or PlayStation" returning `faq-015` and
"billing currency USD to EUR" returning `faq-025`; context_precision (0.759) was the second-weakest, and the same two leaked
chunks account for the gap (they do not touch recall, which stayed 1.0).

**Change made:** gate calibration (relevance / lexical-overlap thresholds). The exact prior values were not recorded in the
repo, which is itself a gap; from now on use `evaluate.py --label <name>` before touching any threshold.

**Why this layer:** recall was already 1.0 and faithfulness 1.0, so the right entries were being found and the generator was
faithful; the failures were *irrelevant context passing through*, which only the gate controls. `evaluate.py --diagnose` prints
cosine, overlap, and gate verdict per hit to confirm this. Evidence it was not a generation problem: `test_runs/10_eval_after.txt`
(same retrieval as "before") has the same leaks, while only citation_precision moved (0.944 → 1.0, a prompt-layer effect).

## Round 2: hybrid retrieval + IDF-weighted gate (this change)
What changed: `lexical.py` (BM25 over the whole collection), RRF fusion, IDF-weighted overlap, rescue tier, gate-then-truncate
(see `retrieval-notes.md`). `evaluate.py` now also reports `vector_only_gated` so hybrid's effect is isolated:

| Report key | Meaning |
|---|---|
| `baseline_raw_top3` | raw vector top-3, no gate |
| `vector_only_gated` | vector candidates + same gate (`hybrid=False`) |
| `pipeline` | hybrid BM25+vector (RRF) + gate |

Verified offline (no model needed): `pytest` 27 passed, 0 xfailed. The required swallow unanswerable previously `xfail` and
now abstains; a synthetic regression shows plain vector search missing an entry that hybrid recovers.

**Not yet measured:** MiniLM-based Round 2 numbers. The thresholds (`MIN_RELEVANCE`, `HIGH_CONFIDENCE`, `MIN_LEXICAL_OVERLAP`,
`RESCUE_*`) were calibrated for MiniLM and the weighted overlap changes their meaning, so do not assume Round 1 numbers carry over.
Run, in order:
```sh
python evaluate.py --label hybrid --generate --judge
python evaluate.py --check          # must exit 0: no unanswerable returns context, recall >= 0.8
python evaluate.py --sweep          # if abstention or recall regressed, tune MIN_LEXICAL_OVERLAP / RESCUE_MIN_OVERLAP
```
Then add a "Round 2" column from `evaluation/results_hybrid.json`. Expected risk to watch: answerable paraphrases lose recall
if `MIN_LEXICAL_OVERLAP` is too strict under IDF weighting.

## Caveats
* 14 questions is small, and the gate thresholds were tuned on the same questions that are scored, so Round 1 abstention of 1.0
  is optimistic. Hold out new questions before the next tuning round.
* Faithfulness is judged by the same model family that generates answers (9 answerable cases), which is weak evidence.
  Ragas (or a different judge model) would be a stronger check; this repo uses a custom scorer, which the brief allows.
* Eval-set unanswerables are shorter paraphrases of the Part 1 wording; the exact Part 1 strings are covered by pytest and
  `test_runs/06_unanswerables.txt`.