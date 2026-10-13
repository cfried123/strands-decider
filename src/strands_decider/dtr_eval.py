"""Score answers on DTR-Bench with the review-queue abstention policy.

A prediction is `(decision, top_probability)`: `decision` is an option name, or
"abstain" when the model's answer falls inside the review band. Used for the local
strands-decider runs, for Snowflake batch outputs, and for the frontier-LLM baseline,
so all three are scored identically.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

NOUL_BAND = 0.25  # abstain when |p(true) - 0.5| < NOUL_BAND
CHOICE_FLOOR = 0.5  # abstain when the top option probability is below this


def decide(
    answer: dict[str, Any], *, noul_band: float = NOUL_BAND, choice_floor: float = CHOICE_FLOOR
) -> tuple[str, str, float]:
    """(decision, argmax option, top probability) for one API answer dict."""
    if answer["type"] == "noul":
        p = float(answer["noul"])
        top = "true" if p >= 0.5 else "false"
        conf = max(p, 1 - p)
        return ("abstain" if abs(p - 0.5) < noul_band else top), top, conf
    probs = answer["probabilities"]
    # Choice probabilities are keyed by option name, score probabilities by level index.
    top = max(probs, key=probs.get)
    conf = float(probs[top])
    return ("abstain" if conf < choice_floor else top), top, conf


def score(items: list[dict], preds: dict[str, dict]) -> dict[str, Any]:
    """`preds[item_id]` = {"decision", "argmax", "confidence"[, "latency_ms"]}."""
    groups: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for it in items:
        p = preds.get(it["item_id"])
        if p is None:
            continue
        for g in ("all", it["split"], it["kind"], it["task"]):
            groups[g].append((it, p))
    return {g: _metrics(rows) for g, rows in sorted(groups.items())}


def _metrics(rows: list[tuple[dict, dict]]) -> dict[str, Any]:
    definite = [(it, p) for it, p in rows if it["expected"] != "abstain"]
    ambiguous = [(it, p) for it, p in rows if it["expected"] == "abstain"]
    decided = [(it, p) for it, p in definite if p["decision"] != "abstain"]
    strict = sum(
        (p["decision"] == "abstain")
        if it["expected"] == "abstain"
        else (p["decision"] == it["expected"])
        for it, p in rows
    )
    m: dict[str, Any] = {
        "n": len(rows),
        "strict_accuracy": strict / len(rows) if rows else None,
        "forced_accuracy": (
            sum(p["argmax"] == it["expected"] for it, p in definite) / len(definite)
        )
        if definite
        else None,
        "coverage": len(decided) / len(definite) if definite else None,
        "selective_accuracy": (
            sum(p["decision"] == it["expected"] for it, p in decided) / len(decided)
        )
        if decided
        else None,
        "abstain_recall": (sum(p["decision"] == "abstain" for _, p in ambiguous) / len(ambiguous))
        if ambiguous
        else None,
        "ece": _ece([(p["confidence"], p["argmax"] == it["expected"]) for it, p in definite]),
    }
    lat = [p["latency_ms"] for _, p in rows if p.get("latency_ms") is not None]
    if lat:
        m["mean_latency_ms"] = sum(lat) / len(lat)
    return m


def _ece(pairs: list[tuple[float, bool]], bins: int = 10) -> float | None:
    if not pairs:
        return None
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(c, ok) for c, ok in pairs if lo < c <= hi or (b == 0 and c == 0)]
        if sel:
            conf = sum(c for c, _ in sel) / len(sel)
            acc = sum(ok for _, ok in sel) / len(sel)
            total += len(sel) / len(pairs) * abs(conf - acc)
    return total


def run_local(checkpoint: str, bench: str, out: str, device: str = "cpu") -> dict[str, Any]:
    """Ask every DTR-Bench item of `bench` with checkpoint and write predictions + metrics."""
    import time

    from pydantic import TypeAdapter

    from .infer import load_engine
    from .schema import Question

    items = [json.loads(line) for line in open(bench, encoding="utf-8")]
    eng = load_engine(checkpoint, device=device)
    qa = TypeAdapter(Question)
    preds: dict[str, dict] = {}
    for it in items:
        t = time.perf_counter()
        r = eng.ask(it["state"], {"q": qa.validate_python(it["question"])})
        ans = r.answers["q"].model_dump()
        d, top, conf = decide(ans)
        preds[it["item_id"]] = {
            "decision": d,
            "argmax": top,
            "confidence": conf,
            "latency_ms": (time.perf_counter() - t) * 1000,
            "answer": ans,
        }
    metrics = score(items, preds)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            {"checkpoint": checkpoint, "metrics": metrics, "predictions": preds}, fh, indent=1
        )
    return metrics


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on DTR-Bench.")
    ap.add_argument("checkpoint")
    ap.add_argument("--bench", default="data/dtr_bench.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    m = run_local(a.checkpoint, a.bench, a.out, a.device)
    for g in ("all", "in_domain", "holdout", "noul", "choice", "score"):
        if g in m:
            print(g, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m[g].items()})


if __name__ == "__main__":
    main()
