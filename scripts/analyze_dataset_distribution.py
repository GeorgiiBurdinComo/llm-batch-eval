"""Analyze dataset distribution and baseline disagreement stats.

Outputs: dataset size/prevalence, baseline coverage, disagreement buckets,
per-model accuracy, pairwise delta/q, optional sample-size helpers.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from itertools import combinations
from math import ceil, floor, log, sqrt
from statistics import NormalDist, median
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except Exception:
    pass

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
for _d in (_scripts_dir, os.path.join(_scripts_dir, "lib")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from load_dataset import load_dataset_rows


# ── Helpers ──────────────────────────────────────────────────────────────────

def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * log(p) + (1.0 - p) * log(1.0 - p))


def _load_truth_csv(path: str) -> Dict[str, bool]:
    truth: Dict[str, bool] = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = str(row.get("custom_id", "")).strip()
            if cid:
                truth[cid] = str(row.get("campaign_relevant", "")).strip().lower() == "true"
    return truth


def compute_disagreement_scores(baseline: Dict[str, Dict[str, bool]]) -> Dict[str, float]:
    """std(predictions) * entropy(p_true) per example. No scipy dependency."""
    scores: Dict[str, float] = {}
    for cid, preds in baseline.items():
        if not preds:
            scores[cid] = 0.0
            continue
        vals = [1.0 if v else 0.0 for v in preds.values()]
        mu = sum(vals) / len(vals)
        std = sqrt(sum((x - mu) ** 2 for x in vals) / len(vals))
        scores[cid] = std * (_binary_entropy(mu) + 1e-10)
    return scores


def _accuracy(truth: Dict[str, bool], preds: Dict[str, bool]) -> Tuple[int, Optional[float]]:
    overlap = [c for c in preds if c in truth]
    if not overlap:
        return 0, None
    correct = sum(1 for c in overlap if bool(preds[c]) == bool(truth[c]))
    return len(overlap), correct / len(overlap)


def _pairwise_disagreement(p1: Dict[str, bool], p2: Dict[str, bool]) -> Optional[Dict[str, Any]]:
    overlap = [c for c in p1 if c in p2]
    if not overlap:
        return None
    n = len(overlap)
    return {"n": n, "disagree_frac": sum(1 for c in overlap if bool(p1[c]) != bool(p2[c])) / n}


def _pairwise_correctness(
    truth: Dict[str, bool], p1: Dict[str, bool], p2: Dict[str, bool],
) -> Optional[Dict[str, Any]]:
    overlap = [c for c in p1 if c in p2 and c in truth]
    if not overlap:
        return None
    a_right = b_right = p01 = p10 = 0
    for c in overlap:
        y = bool(truth[c])
        a_ok, b_ok = bool(p1[c]) == y, bool(p2[c]) == y
        a_right += int(a_ok)
        b_right += int(b_ok)
        p01 += int(not a_ok and b_ok)
        p10 += int(a_ok and not b_ok)
    n = len(overlap)
    return {
        "n": n, "acc_a": a_right / n, "acc_b": b_right / n,
        "delta": b_right / n - a_right / n,
        "p01": p01 / n, "p10": p10 / n, "q": (p01 + p10) / n,
    }


def _z(p: float) -> float:
    return float(NormalDist().inv_cdf(p))


def _n_for_ci(h: float, alpha: float = 0.05) -> int:
    z = _z(1.0 - alpha / 2.0)
    return int(ceil(z * z * 0.25 / (h * h)))


def _n_for_delta(delta: float, q: float, alpha: float = 0.05, power: float = 0.8) -> int:
    return int(ceil((_z(1.0 - alpha / 2.0) + _z(power)) ** 2 * q / (delta * delta)))


def _percentile(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    if p <= 0:
        return float(vals[0])
    if p >= 100:
        return float(vals[-1])
    idx = (p / 100.0) * (len(vals) - 1)
    lo, hi = int(floor(idx)), int(ceil(idx))
    if lo == hi:
        return float(vals[lo])
    return float(vals[lo] + (idx - lo) * (vals[hi] - vals[lo]))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Dataset distribution & baseline analysis")
    p.add_argument("--csv", default=None)
    p.add_argument("--langfuse-dataset", default=None, dest="langfuse_dataset_name")
    p.add_argument("--body-template", default=None, dest="body_template_path")
    p.add_argument("--truth-csv", default=None)
    p.add_argument("--baseline", default="data/baseline_predictions.json")
    p.add_argument("--out", default=None)
    p.add_argument("--top-pairs", type=int, default=10)
    p.add_argument("--ci-halfwidth", type=float, default=None)
    p.add_argument("--delta", type=float, default=None)
    p.add_argument("--power", type=float, default=0.8)
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    rows: List[Dict[str, Any]] = []
    truth_by_id: Dict[str, bool] = {}
    load_error: Optional[str] = None

    try:
        rows = load_dataset_rows(
            csv_path=args.csv, langfuse_dataset_name=args.langfuse_dataset_name,
            body_template_path=args.body_template_path,
        )
        truth_by_id = {str(r["custom_id"]): bool(r["campaign_relevant"]) for r in rows}
    except Exception as e:
        load_error = f"{type(e).__name__}: {e}"
        if args.truth_csv:
            path = os.path.join(ROOT, args.truth_csv) if not os.path.isabs(args.truth_csv) else args.truth_csv
            truth_by_id = _load_truth_csv(path)
            rows = [{"custom_id": c, "campaign_relevant": y} for c, y in truth_by_id.items()]

    n_total = len(rows)
    n_pos = sum(1 for r in rows if r["campaign_relevant"])

    report: Dict[str, Any] = {
        "dataset": {
            "n": n_total,
            "campaign_relevant": {"true": n_pos, "false": n_total - n_pos,
                                  "p_true": (n_pos / n_total) if n_total else None},
            "load_error": load_error,
        }
    }

    baseline_abs = os.path.join(ROOT, args.baseline) if not os.path.isabs(args.baseline) else args.baseline
    baseline: Optional[Dict[str, Dict[str, bool]]] = None
    if os.path.isfile(baseline_abs):
        with open(baseline_abs, "r", encoding="utf-8") as f:
            baseline = json.load(f)

    if baseline:
        report["baseline"] = _analyze_baseline(
            baseline, truth_by_id, n_total, baseline_abs, args,
        )

    _print_report(report, args)

    if args.out:
        out_abs = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
        os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
        with open(out_abs, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote JSON report to: {out_abs}")

    return 0


def _analyze_baseline(
    baseline: Dict[str, Dict[str, bool]],
    truth_by_id: Dict[str, bool],
    n_total: int,
    baseline_path: str,
    args,
) -> Dict[str, Any]:
    scores = compute_disagreement_scores(baseline)
    covered = [c for c in (truth_by_id or scores) if c in scores]
    uncovered = (n_total - len(covered)) if truth_by_id else None

    score_vals = sorted(scores[c] for c in covered)
    p80 = _percentile(score_vals, 80.0)
    p50 = _percentile(score_vals, 50.0)

    high = [c for c in covered if scores[c] >= p80]
    med = [c for c in covered if p50 <= scores[c] < p80]
    low = [c for c in covered if scores[c] < p50]

    models = sorted({m for preds in baseline.values() for m in preds})
    by_model: Dict[str, Dict[str, bool]] = {m: {} for m in models}
    for cid, preds in baseline.items():
        for m, v in preds.items():
            by_model[m][cid] = bool(v)

    if truth_by_id:
        acc_by_model = {m: dict(zip(["n", "accuracy"], _accuracy(truth_by_id, by_model[m]))) for m in models}
    else:
        acc_by_model = {}
        for m in models:
            ps = by_model[m]
            acc_by_model[m] = {"n": len(ps), "pred_true_rate": (sum(1 for v in ps.values() if v) / len(ps)) if ps else None}

    pairwise: List[Dict[str, Any]] = []
    for a, b in combinations(models, 2):
        if truth_by_id:
            s = _pairwise_correctness(truth_by_id, by_model[a], by_model[b])
        else:
            s = _pairwise_disagreement(by_model[a], by_model[b])
        if s:
            pairwise.append({"a": a, "b": b, **s})

    sort_key = (lambda x: (x["n"], abs(x.get("delta", 0)))) if truth_by_id else (lambda x: (x["n"], x.get("disagree_frac", 0)))
    pairwise.sort(key=sort_key, reverse=True)

    q_vals = [x["q"] for x in pairwise if "q" in x and x["n"] > 0]
    d_vals = [x["disagree_frac"] for x in pairwise if "disagree_frac" in x and x["n"] > 0]
    q_median = float(median(q_vals)) if q_vals else None
    pred_disagree_median = float(median(d_vals)) if d_vals else None

    section: Dict[str, Any] = {
        "path": baseline_path,
        "coverage": {
            "covered_n": len(covered), "uncovered_n": uncovered,
            "covered_frac": (len(covered) / n_total) if n_total and uncovered is not None else None,
        },
        "disagreement_buckets": {
            "thresholds": {"p50": p50, "p80": p80},
            "counts": {"high_ge_p80": len(high), "medium_p50_to_p80": len(med),
                       "low_lt_p50": len(low), "uncovered_no_baseline": uncovered},
            "fractions_of_dataset": {
                "high": len(high) / n_total if n_total else None,
                "medium": len(med) / n_total if n_total else None,
                "low": len(low) / n_total if n_total else None,
                "uncovered": uncovered / n_total if n_total and uncovered is not None else None,
            },
        },
        "accuracy_by_model": acc_by_model,
        "pairwise": {"pairs": pairwise, "q_median": q_median, "pred_disagree_median": pred_disagree_median},
    }

    sizing: Dict[str, Any] = {}
    if args.ci_halfwidth is not None:
        sizing["ci_halfwidth"] = {
            "halfwidth": args.ci_halfwidth, "alpha": args.alpha, "p_assumed": 0.5,
            "n_required": _n_for_ci(args.ci_halfwidth, alpha=args.alpha),
        }
    if args.delta is not None:
        q_used = q_median if q_median is not None else 0.2
        sizing["delta_power"] = {
            "delta": args.delta, "alpha": args.alpha, "power": args.power,
            "q_used": q_used, "n_required": _n_for_delta(args.delta, q=q_used, alpha=args.alpha, power=args.power),
        }
    if sizing:
        section["sizing"] = sizing

    return section


def _print_report(report: Dict[str, Any], args) -> None:
    print(json.dumps(report["dataset"], indent=2))

    if "baseline" not in report:
        print("\n(no baseline found; pass --baseline or generate data/baseline_predictions.json)")
        return

    b = report["baseline"]
    print("\nBaseline coverage:")
    print(json.dumps(b["coverage"], indent=2))
    print("\nDisagreement buckets (counts):")
    print(json.dumps(b["disagreement_buckets"]["counts"], indent=2))

    print("\nPer-model stats:")
    items = sorted(b["accuracy_by_model"].items(),
                   key=lambda kv: (kv[1].get("n", 0), kv[1].get("accuracy") or kv[1].get("pred_true_rate") or -1),
                   reverse=True)
    for m, v in items[:20]:
        if "accuracy" in v:
            print(f"- {m}: n={v['n']}, acc={None if v['accuracy'] is None else round(v['accuracy'], 4)}")
        else:
            print(f"- {m}: n={v['n']}, pred_true_rate={None if v['pred_true_rate'] is None else round(v['pred_true_rate'], 4)}")

    pairs = b["pairwise"]["pairs"]
    if pairs:
        print(f"\nPairwise (top {args.top_pairs}):")
        for row in pairs[:args.top_pairs]:
            if "delta" in row:
                print(f"- {row['a']} vs {row['b']}: n={row['n']}, delta={row['delta']:+.4f}, q={row['q']:.4f}")
            else:
                print(f"- {row['a']} vs {row['b']}: n={row['n']}, disagree={row['disagree_frac']:.4f}")
        if b["pairwise"].get("q_median") is not None:
            print(f"\nMedian q: {b['pairwise']['q_median']:.4f}")
        if b["pairwise"].get("pred_disagree_median") is not None:
            print(f"\nMedian prediction disagreement: {b['pairwise']['pred_disagree_median']:.4f}")

    if "sizing" in b:
        print("\nSizing helpers:")
        print(json.dumps(b["sizing"], indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
