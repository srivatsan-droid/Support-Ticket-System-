from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from generator import ABSTENTION, extract_cited_ids, generate_support_answer


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def score_retrieval(cases: list[dict], retrieve_fn) -> dict:
    """retrieve_fn(question) -> list of entry IDs. Pure function, unit-testable."""
    recall, precision, abstained, details = [], [], [], []
    for case in cases:
        found = list(dict.fromkeys(retrieve_fn(case["question"])))
        expected = set(case["expected_source_ids"])
        if case["answerable"]:
            hit = len(expected & set(found))
            recall.append(hit / len(expected))
            precision.append(hit / len(found) if found else 0.0)
        else:
            abstained.append(1.0 if not found else 0.0)
            if found:
                details.append({"question": case["question"], "leaked_ids": found})
    return {
        "context_recall": _mean(recall),
        "context_precision": _mean(precision),
        "abstention_rate": _mean(abstained),
        "leaked_unanswerables": details,
    }


def judge_faithfulness(question: str, context: str, answer: str) -> bool:
    from openai import OpenAI
    provider = OpenAI(base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
                      api_key=os.environ["OPENAI_API_KEY"])
    prompt = (
        "You are a strict fact-checker. Given SNIPPETS and an ANSWER, decide whether EVERY factual "
        'claim in the ANSWER is directly supported by the SNIPPETS. Reply with JSON only: '
        '{"faithful": true|false, "unsupported": ["..."]}\n\n'
        f"SNIPPETS:\n{context}\n\nQUESTION: {question}\n\nANSWER: {answer}"
    )
    out = provider.chat.completions.create(
        model=os.getenv("LLM_MODEL", "openai/gpt-oss-20b"), temperature=0,
        messages=[{"role": "user", "content": prompt}],
    ).choices[0].message.content
    try:
        return bool(json.loads(out[out.index("{"): out.rindex("}") + 1])["faithful"])
    except (ValueError, KeyError):
        return False  # unparsable judge output counts as a failure, never a pass


def score_generation(cases: list[dict], judge: bool) -> dict:
    from retrieval import retrieve
    cite_precision, faithful, correct_abstain = [], [], []
    for case in cases:
        hits = retrieve(case["question"])
        answer = "".join(generate_support_answer(case["question"], hits))
        if not case["answerable"]:
            correct_abstain.append(1.0 if answer.strip() == ABSTENTION else 0.0)
            continue
        cited = extract_cited_ids(answer, [h.payload["entry_id"] for h in hits])
        expected = set(case["expected_source_ids"])
        cite_precision.append(len(set(cited) & expected) / len(cited) if cited else 0.0)
        if judge:
            context = "\n\n".join(f"[{h.payload['entry_id']}] {h.payload['chunk_text']}" for h in hits)
            faithful.append(1.0 if judge_faithfulness(case["question"], context, answer) else 0.0)
    result = {"citation_precision": _mean(cite_precision), "generated_abstention_rate": _mean(correct_abstain)}
    if judge:
        result["faithfulness"] = _mean(faithful)
    return result


def diagnose(cases: list[dict]) -> None:
    """Print cosine, lexical overlap and gate verdict for the top-5 raw hits of every question."""
    import retrieval
    from ingest import encoder
    index = retrieval.get_index(retrieval.client, retrieval.COLLECTION_NAME)
    for case in cases:
        tag = "ANSWERABLE" if case["answerable"] else "UNANSWERABLE"
        print(f"\n[{tag}] {case['question']}  expected={case['expected_source_ids']}")
        for hit in retrieval.vector_search(encoder.encode(case["question"]).tolist(), 5):
            entry_id = hit.payload["entry_id"]
            mark = "*" if entry_id in case["expected_source_ids"] else " "
            verdict = "PASS" if retrieval.passes_gate(case["question"], hit, index=index) else "rej "
            print(f"  {mark}{entry_id} cos={hit.score:.3f} "
                  f"ov={retrieval.lexical_overlap(case['question'], hit, index):.2f} gate={verdict}  "
                  f"{hit.payload['title'][:45]}")


def sweep(cases: list[dict], relevance=(0.32, 0.35, 0.38, 0.40, 0.42, 0.45),
          overlap=(0.20, 0.25, 0.34, 0.50)) -> None:
    """Score the real pipeline under different gate thresholds (HIGH_CONFIDENCE unchanged)."""
    import retrieval
    defaults = (retrieval.MIN_RELEVANCE, retrieval.MIN_LEXICAL_OVERLAP)
    print("MIN_REL  MIN_OV   recall  precision  abstain  leaked")
    try:
        for rel in relevance:
            for ov in overlap:
                retrieval.MIN_RELEVANCE, retrieval.MIN_LEXICAL_OVERLAP = rel, ov
                r = score_retrieval(
                    cases, lambda q: [h.payload["entry_id"] for h in retrieval.retrieve(q)])
                note = "  <- current" if (rel, ov) == defaults else ""
                print(f"{rel:<8} {ov:<8} {r['context_recall']:<7} {r['context_precision']:<10} "
                      f"{r['abstention_rate']:<8} {len(r['leaked_unanswerables'])}{note}")
    finally:
        retrieval.MIN_RELEVANCE, retrieval.MIN_LEXICAL_OVERLAP = defaults


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default=None, help="save results to evaluation/results_<label>.json")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--check", action="store_true", help="CI gate: non-zero exit on regressions")
    parser.add_argument("--min-recall", type=float, default=0.8)
    parser.add_argument("--diagnose", action="store_true", help="print per-question scores, then exit")
    parser.add_argument("--sweep", action="store_true", help="sweep gate thresholds, then exit")
    args = parser.parse_args()

    from ingest import encoder
    from retrieval import retrieve, vector_search
    cases = json.loads(Path("evaluation/questions.json").read_text(encoding="utf-8"))

    if args.diagnose or args.sweep:
        if args.diagnose:
            diagnose(cases)
        if args.sweep:
            print()
            sweep(cases)
        return 0

    def raw_top3(question: str) -> list[str]:
        hits = vector_search(encoder.encode(question).tolist(), 3)
        return [h.payload["entry_id"] for h in hits]

    def vector_gated(question: str) -> list[str]:
        return [h.payload["entry_id"] for h in retrieve(question, hybrid=False)]

    def pipeline(question: str) -> list[str]:
        return [h.payload["entry_id"] for h in retrieve(question)]

    report = {"label": args.label, "baseline_raw_top3": score_retrieval(cases, raw_top3),
              "vector_only_gated": score_retrieval(cases, vector_gated),
              "pipeline": score_retrieval(cases, pipeline)}
    if args.generate:
        report["generation"] = score_generation(cases, judge=args.judge)
    if args.label:
        Path(f"evaluation/results_{args.label}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if args.check:
        p = report["pipeline"]
        failures = []
        if p["leaked_unanswerables"]:
            failures.append(f"unanswerable questions returned context: {p['leaked_unanswerables']}")
        if p["context_recall"] < args.min_recall:
            failures.append(f"context_recall {p['context_recall']} < {args.min_recall}")
        for failure in failures:
            print("CHECK FAILED:", failure, file=sys.stderr)
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())