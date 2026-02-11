"""
Disagreement-based subset selection for cost-efficient drift monitoring.

- Load baseline_predictions.json (custom_id -> { model_id: bool }) when present.
- Compute per-example disagreement score: std(predictions) * entropy.
- sample_subset(n, high_weight=0.67, medium_weight=0.23, baseline_weight=0.10) by percentiles.
- Stratified fallback when no baseline.
"""

import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

# Lib (load_dataset) is not a direct entrypoint
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_lib_dir = os.path.join(_scripts_dir, "lib")
if _lib_dir not in sys.path:
    sys.path.insert(0, _lib_dir)
from load_dataset import load_dataset_rows, load_csv_rows


def load_baseline(baseline_path: str) -> Dict[str, Dict[str, bool]]:
    """
    Load baseline_predictions.json.
    Returns: custom_id -> { model_id: bool }
    """
    with open(baseline_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_disagreement_scores(
    baseline: Dict[str, Dict[str, bool]],
) -> Dict[str, float]:
    """
    Per-example score = std(numeric_predictions) * entropy(proportion_true).
    Higher = more disagreement across models.
    """
    from scipy.stats import entropy

    scores = {}
    for custom_id, preds in baseline.items():
        if not preds:
            scores[custom_id] = 0.0
            continue
        vals = [1.0 if v else 0.0 for v in preds.values()]
        std = float(np.std(vals))
        p_true = sum(vals) / len(vals)
        p_false = 1.0 - p_true
        if p_true <= 0 or p_false <= 0:
            ent = 0.0
        else:
            ent = entropy([p_true, p_false])
        scores[custom_id] = std * (ent + 1e-10)
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
    """
    Sample n examples. If baseline_path exists, use disagreement-weighted sampling.
    Weights: high (≥80th pctl), medium (50–80th), baseline (random from bottom 50%).
    Returns list of dicts with custom_id, body, campaign_relevant for batch scripts.
    Data source: csv_path if set, else Langfuse dataset (langfuse_dataset_name or env/default).
    """
    rows = load_dataset_rows(
        csv_path=csv_path,
        langfuse_dataset_name=langfuse_dataset_name,
        body_template_path=body_template_path,
    )
    if not rows:
        return []

    if baseline_path and os.path.isfile(baseline_path):
        baseline = load_baseline(baseline_path)
        scores = compute_disagreement_scores(baseline)
        # Only rows that have a score
        rows_with_score = [r for r in rows if r["custom_id"] in scores]
        if not rows_with_score:
            return stratified_fallback(rows, n, random_state)

        id_to_row = {r["custom_id"]: r for r in rows}
        score_vals = [scores[r["custom_id"]] for r in rows_with_score]
        high_p = np.percentile(score_vals, 80)
        medium_p = np.percentile(score_vals, 50)

        n_high = int(n * high_weight)
        n_medium = int(n * medium_weight)
        n_baseline = n - n_high - n_medium

        high_pool = [r for r in rows_with_score if scores[r["custom_id"]] >= high_p]
        medium_pool = [
            r for r in rows_with_score
            if medium_p <= scores[r["custom_id"]] < high_p
        ]
        low_pool = [r for r in rows_with_score if scores[r["custom_id"]] < medium_p]

        rng = np.random.default_rng(random_state)
        out = []
        for pool, size in [(high_pool, n_high), (medium_pool, n_medium), (low_pool, n_baseline)]:
            if not pool or size <= 0:
                continue
            k = min(size, len(pool))
            indices = rng.choice(len(pool), size=k, replace=False)
            out.extend([pool[i] for i in indices])

        # Top up to n without duplicates:
        # 1) from remaining scored rows, 2) from rows without baseline coverage.
        selected_ids = {r["custom_id"] for r in out}
        if len(out) < n:
            remaining_scored = [r for r in rows_with_score if r["custom_id"] not in selected_ids]
            k = min(n - len(out), len(remaining_scored))
            if k > 0:
                idx = rng.choice(len(remaining_scored), size=k, replace=False)
                out.extend([remaining_scored[i] for i in idx])
                selected_ids = {r["custom_id"] for r in out}

        if len(out) < n:
            remaining_all = [r for r in rows if r["custom_id"] not in selected_ids]
            k = min(n - len(out), len(remaining_all))
            if k > 0:
                idx = rng.choice(len(remaining_all), size=k, replace=False)
                out.extend([remaining_all[i] for i in idx])

        rng.shuffle(out)
        return out[:n]
    return stratified_fallback(rows, n, random_state)


def stratified_fallback(
    rows: List[Dict],
    n: int,
    random_state: int = 42,
) -> List[Dict]:
    """Stratified sample by campaign_relevant to preserve class balance."""
    if not rows or n >= len(rows):
        return rows[:n] if rows else []

    by_class: Dict[bool, List[Dict]] = {}
    for r in rows:
        key = r["campaign_relevant"]
        by_class.setdefault(key, []).append(r)

    rng = np.random.default_rng(random_state)
    total = len(rows)
    out = []
    for cls, pool in by_class.items():
        prop = len(pool) / total
        k = min(int(round(n * prop)), len(pool), n - len(out))
        if k <= 0:
            continue
        indices = rng.choice(len(pool), size=k, replace=False)
        out.extend([pool[i] for i in indices])

    # Rounding can leave us short; top up from remaining rows.
    if len(out) < n:
        selected_ids = {r["custom_id"] for r in out}
        remaining = [r for r in rows if r["custom_id"] not in selected_ids]
        k = min(n - len(out), len(remaining))
        if k > 0:
            idx = rng.choice(len(remaining), size=k, replace=False)
            out.extend([remaining[i] for i in idx])

    rng.shuffle(out)
    return out[:n]


def aggregate_baseline(
    results_dir: str,
    output_path: str = "data/baseline_predictions.json",
) -> str:
    """
    Aggregate result JSONL files into baseline_predictions.json.
    Expects files like results/openai_gpt-5-mini.jsonl, results/gemini_gemini-2.5-flash.jsonl.
    Output: { custom_id: { model_id: bool }, ... }
    """
    aggregated: Dict[str, Dict[str, bool]] = {}
    for name in sorted(os.listdir(results_dir)):
        if not name.endswith(".jsonl"):
            continue
        # openai_gpt-5-mini.jsonl -> gpt-5-mini; gemini_gemini-2.5-flash.jsonl -> gemini-2.5-flash
        stem = name[:-6]
        model = stem.split("_", 1)[-1] if "_" in stem else stem
        path = os.path.join(results_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # OpenAI batch: "custom_id"; Gemini native batch: "key"
                cid = obj.get("custom_id") or obj.get("key")
                if not cid:
                    continue
                response = obj.get("response", {})
                pred = _extract_bool(response, path)
                if cid not in aggregated:
                    aggregated[cid] = {}
                aggregated[cid][model] = pred
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)
    print(f"[aggregate_baseline] {len(aggregated)} examples from {results_dir} → {output_path}")
    return output_path


def _extract_bool(response: Dict, _path: str = "") -> bool:
    """Extract campaign_relevant from OpenAI Responses or Gemini native batch response."""
    # OpenAI Responses API: response.text (dict or JSON string)
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
    # Gemini native batch: response.candidates[0].content.parts[0].text
    try:
        candidates = response.get("candidates") or []
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts") or []
            if parts:
                text = parts[0].get("text", "")
                if isinstance(text, str):
                    return bool(json.loads(text).get("campaign_relevant", False))
                return bool(text.get("campaign_relevant", False))
    except Exception:
        pass
    return False


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Sample subset for drift eval")
    p.add_argument("--csv", default=None, help="Use CSV; if omitted, use Langfuse dataset")
    p.add_argument("--langfuse-dataset", default=None, help="Langfuse dataset name (default from env or campaign_relevance_02e1a68ccb0f)")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--baseline", default="data/baseline_predictions.json", help="Path or none to skip")
    p.add_argument("--out", help="Write selected custom_ids to file (one per line)")
    p.add_argument("--aggregate", action="store_true", help="Aggregate results dir into baseline")
    p.add_argument("--results-dir", default="data/results", help="For --aggregate")
    args = p.parse_args()

    if args.aggregate:
        aggregate_baseline(args.results_dir, args.baseline)
    else:
        baseline_path = args.baseline if os.path.isfile(args.baseline) else None
        subset = sample_subset(
            args.n,
            csv_path=args.csv,
            langfuse_dataset_name=args.langfuse_dataset,
            baseline_path=baseline_path,
        )
        print(f"Sampled {len(subset)} examples")
        if args.out:
            with open(args.out, "w") as f:
                for r in subset:
                    f.write(r["custom_id"] + "\n")
