"""Accuracy report from local result JSONLs + ground truth. Optionally writes chart to reports/."""

import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ingest import _compute_cost, _extract_prediction, _get_effective_response, _load_pricing, _normalize_usage, get_ground_truth


def accuracy_and_cost_per_model(
    results_dir: str = "data/results",
    ground_truth_csv: Optional[str] = "input/dataset.csv",
    langfuse_dataset_name: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(ROOT, results_dir)
    if not os.path.isdir(results_dir):
        return {}, {}
    truth = get_ground_truth(ground_truth_csv=ground_truth_csv, langfuse_dataset_name=langfuse_dataset_name)
    pricing = _load_pricing()
    acc_by_model: Dict[str, float] = {}
    cost_by_model: Dict[str, float] = {}

    for name in os.listdir(results_dir):
        if not name.endswith(".jsonl"):
            continue
        parts = name[:-6].split("_", 1)
        provider, model = (parts[0], parts[1]) if len(parts) == 2 else ("openai", parts[0])
        correct = total = 0
        cost_usd = 0.0
        with open(os.path.join(results_dir, name), "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cid = obj.get("custom_id") or obj.get("key")
                effective = _get_effective_response(obj.get("response", {}), provider)
                expected = truth.get(str(cid)) if cid is not None else None
                if expected is not None:
                    total += 1
                    if _extract_prediction(effective, provider) == expected:
                        correct += 1
                inp_tok, out_tok, _ = _normalize_usage(effective, provider)
                bd = _compute_cost(model, inp_tok, out_tok, pricing)
                if bd:
                    cost_usd += bd["total_cost"]
        if total:
            acc_by_model[model] = correct / total
        cost_by_model[model] = cost_usd

    return acc_by_model, cost_by_model


def generate_report(
    results_dir: str = "data/results",
    ground_truth_csv: Optional[str] = "input/dataset.csv",
    langfuse_dataset_name: Optional[str] = None,
    report_dir: str = "reports",
    run_id: Optional[str] = None,
    timestamped: bool = False,
    history_path: Optional[str] = None,
) -> str:
    acc, costs = accuracy_and_cost_per_model(results_dir, ground_truth_csv, langfuse_dataset_name)
    if not acc:
        print("[report] No result JSONLs found")
        return ""

    models = sorted(acc)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id = run_id or os.getenv("RUN_ID") or ""

    lines = ["model,accuracy,cost_usd,run_id,generated_at"]
    for m in models:
        lines.append(f"{m},{acc[m]:.4f},{costs.get(m, 0):.6f},{run_id},{generated_at}")
    text = "\n".join(lines)
    print(text)

    if not os.path.isabs(report_dir):
        report_dir = os.path.join(ROOT, report_dir)
    os.makedirs(report_dir, exist_ok=True)

    suffix = (_safe_suffix(run_id) if run_id else _safe_suffix(generated_at)) if timestamped else ""
    csv_name = f"accuracy_report_{suffix}.csv" if suffix else "accuracy_report.csv"
    csv_path = os.path.join(report_dir, csv_name)
    with open(csv_path, "w") as f:
        f.write(text)
    print(f"[report] Wrote {csv_path}")

    if history_path:
        os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)
        exists = os.path.isfile(history_path)
        with open(history_path, "a") as f:
            if not exists:
                f.write("model,accuracy,cost_usd,run_id,generated_at\n")
            for m in models:
                f.write(f"{m},{acc[m]:.4f},{costs.get(m, 0):.6f},{run_id},{generated_at}\n")
        print(f"[report] Appended {history_path}")

    _try_chart(models, acc, costs, report_dir, suffix)
    return csv_path


def _safe_suffix(value: str) -> str:
    return value.replace(":", "").replace(" ", "_").replace("/", "_").replace("\\", "_")


def _try_chart(models, acc, costs, report_dir, suffix):
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax1 = plt.subplots(figsize=(max(8, len(models) * 0.5), 5))
        x = np.arange(len(models))
        w = 0.35
        ax1.bar(x - w / 2, [acc[m] for m in models], w, color="steelblue", edgecolor="navy", label="Accuracy")
        ax1.set_ylabel("Accuracy", color="steelblue")
        ax1.set_ylim(0.8, 1.02)
        ax1.tick_params(axis="y", labelcolor="steelblue")

        ax2 = ax1.twinx()
        ax2.bar(x + w / 2, [costs.get(m, 0) for m in models], w, color="darkorange", alpha=0.8, edgecolor="chocolate", label="Cost (USD)")
        ax2.set_ylabel("Cost (USD)", color="darkorange")
        ax2.tick_params(axis="y", labelcolor="darkorange")

        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha="right")
        ax1.set_title("Accuracy and cost by model")
        fig.legend(loc="upper right", bbox_to_anchor=(1.12, 1))
        fig.tight_layout()

        chart_name = f"accuracy_chart_{suffix}.png" if suffix else "accuracy_chart.png"
        plt.savefig(os.path.join(report_dir, chart_name), dpi=100, bbox_inches="tight")
        plt.close()
        print(f"[report] Wrote {chart_name}")
    except Exception as e:
        print(f"[report] Chart skip: {e}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="data/results")
    p.add_argument("--csv", default=None)
    p.add_argument("--langfuse-dataset", default=None, dest="langfuse_dataset_name", nargs="?", const="")
    p.add_argument("--report-dir", default="reports")
    p.add_argument("--run-id", default=None)
    p.add_argument("--timestamped", action="store_true")
    p.add_argument("--history", default=None)
    args = p.parse_args()

    use_langfuse = args.csv is None
    generate_report(
        args.results_dir,
        ground_truth_csv=args.csv,
        langfuse_dataset_name=(args.langfuse_dataset_name if args.langfuse_dataset_name is not None else "") if use_langfuse else None,
        report_dir=args.report_dir, run_id=args.run_id,
        timestamped=args.timestamped, history_path=args.history,
    )
