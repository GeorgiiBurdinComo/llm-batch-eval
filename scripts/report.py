"""
Simple accuracy report from local result JSONLs + ground truth CSV.
Optionally write a chart to reports/.
"""

import json
import os
import sys
from typing import Dict, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ingest import _extract_prediction, get_ground_truth


def accuracy_per_model(
    results_dir: str = "data/results",
    ground_truth_csv: Optional[str] = "input/dataset.csv",
    langfuse_dataset_name: Optional[str] = None,
) -> Dict[str, float]:
    """Compute accuracy (0–1) per model from JSONL files."""
    if not os.path.isdir(results_dir):
        return {}
    truth = get_ground_truth(ground_truth_csv=ground_truth_csv, langfuse_dataset_name=langfuse_dataset_name)
    by_model: Dict[str, list] = {}

    for name in os.listdir(results_dir):
        if not name.endswith(".jsonl"):
            continue
        stem = name[:-6]
        parts = stem.split("_", 1)
        model = parts[-1] if len(parts) > 1 else stem
        provider = "openai" if stem.startswith("openai") else "gemini"
        path = os.path.join(results_dir, name)
        correct, total = 0, 0
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cid = obj.get("custom_id") or obj.get("key")
                expected = truth.get(str(cid)) if cid is not None else None
                if expected is None:
                    continue
                pred = _extract_prediction(obj.get("response", {}), provider)
                total += 1
                if pred == expected:
                    correct += 1
        if total:
            by_model[model] = correct / total
    return by_model


def generate_report(
    results_dir: str = "data/results",
    ground_truth_csv: Optional[str] = "input/dataset.csv",
    langfuse_dataset_name: Optional[str] = None,
    report_dir: str = "reports",
) -> str:
    """Print accuracy table and optionally save a bar chart. Returns report path."""
    acc = accuracy_per_model(results_dir, ground_truth_csv, langfuse_dataset_name)
    if not acc:
        print("[report] No result JSONLs found")
        return ""

    lines = ["model,accuracy"]
    for model in sorted(acc):
        lines.append(f"{model},{acc[model]:.4f}")
    text = "\n".join(lines)
    print(text)

    os.makedirs(report_dir, exist_ok=True)
    out_path = os.path.join(report_dir, "accuracy_report.csv")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"[report] Wrote {out_path}")

    try:
        import matplotlib.pyplot as plt
        models = sorted(acc)
        plt.figure(figsize=(max(8, len(models) * 0.5), 4))
        plt.bar(range(len(models)), [acc[m] for m in models], color="steelblue", edgecolor="navy")
        plt.xticks(range(len(models)), models, rotation=45, ha="right")
        plt.ylabel("Accuracy")
        plt.title("Accuracy by model")
        plt.tight_layout()
        chart_path = os.path.join(report_dir, "accuracy_chart.png")
        plt.savefig(chart_path, dpi=100)
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
