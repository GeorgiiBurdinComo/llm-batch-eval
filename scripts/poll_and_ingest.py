"""
Poll batch IDs until completed, download results, ingest into Langfuse, optionally update baseline.
"""

import json
import os
import sys
import time
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import batch_openai
import batch_gemini
from ingest import ingest_results
from sample import aggregate_baseline


def poll_until_done(
    batch_ids_path: str = "data/batch_ids.json",
    results_dir: str = "data/results",
    ground_truth_csv: Optional[str] = None,
    langfuse_dataset_name: Optional[str] = None,
    poll_interval_sec: int = 300,
    max_wait_sec: int = 86400 * 2,
) -> None:
    with open(batch_ids_path) as f:
        data = json.load(f)
    batches = [b for b in data.get("batches", []) if b.get("batch_id")]

    os.makedirs(results_dir, exist_ok=True)
    start = time.time()
    while time.time() - start < max_wait_sec:
        pending = []
        for b in batches:
            bid = b["batch_id"]
            provider = b["provider"]
            model = b["model"]
            if provider == "openai":
                st = batch_openai.get_openai_batch_status(bid)
            else:
                st = batch_gemini.get_gemini_batch_status(bid)
            if st["status"] == "completed":
                out_path = os.path.join(results_dir, f"{provider}_{model}.jsonl")
                # Always re-download so we overwrite stale results from an older run (e.g. 5 vs 100)
                if provider == "openai":
                    batch_openai.download_openai_batch_results(bid, out_path)
                else:
                    batch_gemini.download_gemini_batch_results(bid, out_path)
            elif st["status"] in ("failed", "cancelled", "expired"):
                print(f"[poll] {provider}/{model} {st['status']}")
            else:
                pending.append(f"{provider}/{model}")
        if not pending:
            break
        print(f"[poll] Waiting for {len(pending)} batches: {pending[:5]}...")
        time.sleep(poll_interval_sec)
    else:
        print("[poll] Timeout; some batches may still be running")

    for b in batches:
        model, provider = b["model"], b["provider"]
        out_path = os.path.join(results_dir, f"{provider}_{model}.jsonl")
        if os.path.isfile(out_path):
            ingest_results(
                out_path,
                model,
                provider,
                ground_truth_csv=ground_truth_csv,
                langfuse_dataset_name=langfuse_dataset_name,
            )

    # Optionally refresh baseline from this run's results
    aggregate_baseline(results_dir, "data/baseline_predictions.json")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--batch-ids", default="data/batch_ids.json")
    p.add_argument("--results-dir", default="data/results")
    p.add_argument("--csv", default=None, help="Ground truth from CSV; if omitted, use Langfuse dataset")
    p.add_argument("--langfuse-dataset", default=None, dest="langfuse_dataset_name", nargs="?", const="", help="Ground truth from Langfuse (default when no --csv)")
    p.add_argument("--interval", type=int, default=300)
    p.add_argument("--max-wait", type=int, default=86400 * 2)
    args = p.parse_args()
    # When --csv set: use CSV; else use Langfuse (pass "" so ingest uses env/default name when --langfuse-dataset not given)
    use_langfuse = args.csv is None
    poll_until_done(
        batch_ids_path=args.batch_ids,
        results_dir=args.results_dir,
        ground_truth_csv=args.csv,
        langfuse_dataset_name=(args.langfuse_dataset_name if args.langfuse_dataset_name is not None else "") if use_langfuse else None,
        poll_interval_sec=args.interval,
        max_wait_sec=args.max_wait,
    )
