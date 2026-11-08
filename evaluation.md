# Evaluation

Set: `evaluation/questions.json`, 14 questions (9 answerable with expected source IDs, 5 unanswerable, 2 of them in-domain).
`evaluate.py` scores the shipped pipeline (`retrieval.retrieve` against live Qdrant), not a re-implementation.

## How to produce the before/after
```sh
python evaluate.py --label before --generate --judge   # BEFORE the change
# ... make ONE change (e.g. MIN_LEXICAL_OVERLAP 0.20 -> 0.34, a prompt edit, a chunking tweak) ...
python evaluate.py --label after  --generate --judge   # AFTER the change
python evaluate.py --check                              # exit 1 if any unanswerable returns context
```
Raw results are saved to `evaluation/results_before.json` and `results_after.json`.

## Metrics
| Metric | Layer it diagnoses |
|---|---|
| context_recall | retrieval (did we find the right entries?) |
| context_precision | retrieval + rerank (did we avoid noise?) |
| abstention_rate | relevance gate |
| citation_precision | prompt / generation (did the model cite the right IDs?) |
| faithfulness (LLM judge) | generation (are all claims supported by context?) |

Baseline in the output = raw vector top-3 with no rerank and no gate. Its abstention rate is 0 by construction, so compare precision and abstention *of the pipeline* across before/after runs, not baseline vs pipeline.

## Results (fill from the saved JSON; do not hand-edit numbers)
| Metric | Before | After |
|---|---:|---:|
| context_recall | | |
| context_precision | | |
| abstention_rate | | |
| citation_precision | | |
| faithfulness | | |

**Weakest metric before the change:** _fill in_
**Change made (one):** _fill in_
**Why this layer:** _fill in, referencing the table above_

Caveats: 14 questions is small, and thresholds should not be tuned on the same set you report on. Hold out a few questions before tuning.
