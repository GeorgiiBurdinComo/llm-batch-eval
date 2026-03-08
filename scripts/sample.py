"""Disagreement-based subset selection for cost-efficient drift monitoring."""

import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_lib_dir = os.path.join(_scripts_dir, "lib")
if _lib_dir not in sys.path:
    sys.path.insert(0, _lib_dir)
from load_dataset import load_dataset_rows, load_csv_rows


def load_baseline(path: str) -> Dict[str, Dict[str, bool]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_disagreement_scores(baseline: Dict[str, Dict[str, bool]]) -> Dict[str, float]:
    """Per-example score = std(predictions) * entropy(proportion_true)."""
    from scipy.stats import entropy

    scores = {}
    for cid, preds in baseline.items():
        if not preds:
            scores[cid] = 0.0
            continue
        vals = [1.0 if v else 0.0 for v in preds.values()]
        std = float(np.std(vals))
        p = sum(vals) / len(vals)
        scores[cid] = std * (entropy([p, 1.0 - p]) if 0 < p < 1 else 0.0 + 1e-10)
    return scores


def sample_subset(
    n: int,
    csv_path: Optional[str] = None,
    langfuse_dataset_name: Optional[str] = None,
    body_template_path: Optional[str] = None,
    baseline_path: Optional[str] = "data/baseline_predictions.json",
    high_weight: float = 0.67,
    medium_weight: float = 0.23,
    baseline_weight: float = 0.10,
    random_state: int = 42,
) -> List[Dict]:
    """Sample n examples with optional disagreement-weighted sampling from baseline."""
    rows = load_dataset_rows(
        csv_path=csv_path, langfuse_dataset_name=langfuse_dataset_name,
        body_template_path=body_template_path,
    )
    if not rows:
        return []

    if baseline_path and os.path.isfile(baseline_path):
        baseline = load_baseline(baseline_path)
        scores = compute_disagreement_scores(baseline)
        rows_scored = [r for r in rows if r["custom_id"] in scores]
        if not rows_scored:
            return _stratified_fallback(rows, n, random_state)

        score_vals = [scores[r["custom_id"]] for r in rows_scored]
        p80, p50 = np.percentile(score_vals, 80), np.percentile(score_vals, 50)

        pools = {
            "high": [r for r in rows_scored if scores[r["custom_id"]] >= p80],
            "medium": [r for r in rows_scored if p50 <= scores[r["custom_id"]] < p80],
            "low": [r for r in rows_scored if scores[r["custom_id"]] < p50],
        }
        counts = {"high": int(n * high_weight), "medium": int(n * medium_weight)}
        counts["low"] = n - counts["high"] - counts["medium"]

        rng = np.random.default_rng(random_state)
        out = []
        for key in ("high", "medium", "low"):
            pool, size = pools[key], counts[key]
            if pool and size > 0:
                idx = rng.choice(len(pool), size=min(size, len(pool)), replace=False)
                out.extend(pool[i] for i in idx)

        selected = {r["custom_id"] for r in out}
        for source in [rows_scored, rows]:
            if len(out) >= n:
                break
            rem = [r for r in source if r["custom_id"] not in selected]
            k = min(n - len(out), len(rem))
            if k > 0:
                out.extend(rem[i] for i in rng.choice(len(rem), size=k, replace=False))
                selected = {r["custom_id"] for r in out}

        rng.shuffle(out)
        return out[:n]

    return _stratified_fallback(rows, n, random_state)


def _stratified_fallback(rows: List[Dict], n: int, random_state: int = 42) -> List[Dict]:
    if not rows or n >= len(rows):
        return rows[:n] if rows else []

    by_class: Dict[bool, List[Dict]] = {}
    for r in rows:
        by_class.setdefault(r["campaign_relevant"], []).append(r)

    rng = np.random.default_rng(random_state)
    total = len(rows)
    out = []
    for pool in by_class.values():
        k = min(int(round(n * len(pool) / total)), len(pool), n - len(out))
        if k > 0:
            out.extend(pool[i] for i in rng.choice(len(pool), size=k, replace=False))

    if len(out) < n:
        selected = {r["custom_id"] for r in out}
        rem = [r for r in rows if r["custom_id"] not in selected]
        k = min(n - len(out), len(rem))
        if k > 0:
            out.extend(rem[i] for i in rng.choice(len(rem), size=k, replace=False))

    rng.shuffle(out)
    return out[:n]


def aggregate_baseline(results_dir: str, output_path: str = "data/baseline_predictions.json") -> str:
    """Aggregate result JSONLs into baseline_predictions.json."""
    aggregated: Dict[str, Dict[str, bool]] = {}
    for name in sorted(os.listdir(results_dir)):
        if not name.endswith(".jsonl"):
            continue
        model = name[:-6].split("_", 1)[-1] if "_" in name[:-6] else name[:-6]
        with open(os.path.join(results_dir, name), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cid = obj.get("custom_id") or obj.get("key")
                if not cid:
                    continue
                aggregated.setdefault(cid, {})[model] = _extract_bool(obj.get("response", {}))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)
    print(f"[aggregate_baseline] {len(aggregated)} examples from {results_dir} → {output_path}")
    return output_path


def _extract_bool(response: Dict) -> bool:
    """Extract campaign_relevant from any provider response shape."""
    # OpenAI
    if "text" in response:
        text = response["text"]
        if isinstance(text, dict):
            return bool(text.get("campaign_relevant", False))
        if isinstance(text, str):
            try:
                return bool(json.loads(text).get("campaign_relevant", False))
            except Exception:
                return False
        return False

    # Claude
    try:
        for block in response.get("content") or []:
            if block.get("type") == "text":
                t = block.get("text", "")
                if isinstance(t, str):
                    return bool(json.loads(t).get("campaign_relevant", False))
                if isinstance(t, dict):
                    return bool(t.get("campaign_relevant", False))
    except Exception:
        pass

    # Gemini
    try:
        parts = ((response.get("candidates") or [{}])[0].get("content", {}).get("parts") or [])
        if parts:
            t = parts[0].get("text", "")
            if isinstance(t, str):
                return bool(json.loads(t).get("campaign_relevant", False))
            return bool(t.get("campaign_relevant", False))
    except Exception:
        pass

    return False


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Sample subset for drift eval")
    p.add_argument("--csv", default=None)
    p.add_argument("--langfuse-dataset", default=None)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--baseline", default="data/baseline_predictions.json")
    p.add_argument("--out")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--results-dir", default="data/results")
    args = p.parse_args()

    if args.aggregate:
        aggregate_baseline(args.results_dir, args.baseline)
    else:
        subset = sample_subset(
            args.n, csv_path=args.csv, langfuse_dataset_name=args.langfuse_dataset,
            baseline_path=args.baseline if os.path.isfile(args.baseline) else None,
        )
        print(f"Sampled {len(subset)} examples")
        if args.out:
            with open(args.out, "w") as f:
                for r in subset:
                    f.write(r["custom_id"] + "\n")
