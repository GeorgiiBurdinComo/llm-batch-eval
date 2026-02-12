"""
Simple accuracy report from local result JSONLs + ground truth CSV.
Optionally write a chart to reports/.
"""

import json
import os
import sys
from typing import Dict, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ingest import (
    _compute_cost_breakdown,
    _extract_prediction,
    _get_effective_response,
    _load_pricing,
    _normalize_usage,
    get_ground_truth,
)


def accuracy_and_cost_per_model(
    results_dir: str = "data/results",
    ground_truth_csv: Optional[str] = "input/dataset.csv",
    langfuse_dataset_name: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute accuracy (0–1) and total cost USD per model from JSONL files."""
    if not os.path.isdir(results_dir):
        return {}, {}
    truth = get_ground_truth(ground_truth_csv=ground_truth_csv, langfuse_dataset_name=langfuse_dataset_name)
    pricing = _load_pricing()
    acc_by_model: Dict[str, float] = {}
    cost_by_model: Dict[str, float] = {}

    for name in os.listdir(results_dir):
        if not name.endswith(".jsonl"):
            continue
        stem = name[:-6]
        parts = stem.split("_", 1)
        model = parts[-1] if len(parts) > 1 else stem
        provider = "openai" if stem.startswith("openai") else "gemini"
        path = os.path.join(results_dir, name)
        correct, total = 0, 0
        cost_usd = 0.0
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cid = obj.get("custom_id") or obj.get("key")
                resp = obj.get("response", {})
                effective = _get_effective_response(resp, provider)
                # Accuracy
                expected = truth.get(str(cid)) if cid is not None else None
                if expected is not None:
                    pred = _extract_prediction(effective, provider)
                    total += 1
                    if pred == expected:
                        correct += 1
                # Cost
                prompt_tok, completion_tok, _ = _normalize_usage(effective, provider)
                breakdown = _compute_cost_breakdown(model, prompt_tok, completion_tok, pricing)
                if breakdown is not None:
                    cost_usd += breakdown["total_cost"]
        if total:
            acc_by_model[model] = correct / total
        cost_by_model[model] = cost_usd
    return acc_by_model, cost_by_model


def generate_report(
    results_dir: str = "data/results",
    ground_truth_csv: Optional[str] = "input/dataset.csv",
    langfuse_dataset_name: Optional[str] = None,
    report_dir: str = "reports",
) -> str:
    """Print accuracy + cost table and optionally save a dual-axis bar chart. Returns report path."""
    acc, costs = accuracy_and_cost_per_model(results_dir, ground_truth_csv, langfuse_dataset_name)
    if not acc:
        print("[report] No result JSONLs found")
        return ""

    models = sorted(acc)
    lines = ["model,accuracy,cost_usd"]
    for model in models:
        lines.append(f"{model},{acc[model]:.4f},{costs.get(model, 0):.6f}")
    text = "\n".join(lines)
    print(text)

    os.makedirs(report_dir, exist_ok=True)
    out_path = os.path.join(report_dir, "accuracy_report.csv")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"[report] Wrote {out_path}")

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax1 = plt.subplots(figsize=(max(8, len(models) * 0.5), 5))
        x = np.arange(len(models))
        width = 0.35
        bars1 = ax1.bar(x - width / 2, [acc[m] for m in models], width, color="steelblue", edgecolor="navy", label="Accuracy")
        ax1.set_ylabel("Accuracy", color="steelblue")
        ax1.set_ylim(0.8, 1.02)
        ax1.tick_params(axis="y", labelcolor="steelblue")
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width / 2, [costs.get(m, 0) for m in models], width, color="darkorange", alpha=0.8, edgecolor="chocolate", label="Cost (USD)")
        ax2.set_ylabel("Cost (USD)", color="darkorange")
        ax2.tick_params(axis="y", labelcolor="darkorange")
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha="right")
        ax1.set_title("Accuracy and cost by model")
        fig.legend(loc="upper right", bbox_to_anchor=(1.12, 1))
        fig.tight_layout()
        chart_path = os.path.join(report_dir, "accuracy_chart.png")
        plt.savefig(chart_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"[report] Wrote {chart_path}")
    except Exception as e:
        print(f"[report] Chart skip: {e}")
    return out_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="data/results")
    p.add_argument("--csv", default=None, help="Ground truth from CSV; if omitted, use Langfuse dataset")
    p.add_argument("--langfuse-dataset", default=None, dest="langfuse_dataset_name", nargs="?", const="", help="Ground truth from Langfuse (default when no --csv)")
    p.add_argument("--report-dir", default="reports")
    args = p.parse_args()
    use_langfuse = args.csv is None
    generate_report(
        args.results_dir,
        ground_truth_csv=args.csv,
        langfuse_dataset_name=(args.langfuse_dataset_name if args.langfuse_dataset_name is not None else "") if use_langfuse else None,
        report_dir=args.report_dir,
    )
