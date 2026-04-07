"""Poll batch IDs until completed, download results, ingest into Langfuse."""

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
import batch_claude
from ingest import ingest_results

_STATUS_FN = {
    "openai": batch_openai.get_openai_batch_status,
    "gemini": batch_gemini.get_gemini_batch_status,
    "claude": batch_claude.get_claude_batch_status,
}


def poll_until_done(
    batch_ids_path: str = "data/batch_ids.json",
    results_dir: str = "data/results",
    ground_truth_csv: Optional[str] = None,
    langfuse_dataset_name: Optional[str] = None,
    run_id: Optional[str] = None,
    poll_interval_sec: int = 300,
    max_wait_sec: int = 86400 * 2,
    wait_for_all: bool = False,
) -> None:
    with open(batch_ids_path) as f:
        data = json.load(f)
    batches = [b for b in data.get("batches", []) if b.get("batch_id")]
    run_id = run_id or data.get("run_id") or os.getenv("RUN_ID")
    os.makedirs(results_dir, exist_ok=True)

    unresolved = list(batches)
    start = time.time()
    while unresolved and (time.time() - start < max_wait_sec):
        pending = []
        for b in unresolved:
            bid, provider, model = b["batch_id"], b["provider"], b["model"]
            fn = _STATUS_FN.get(provider)
            if not fn:
                print(f"[poll] Unknown provider {provider} for {model}")
                continue

            st = fn(bid)
            if st["status"] == "completed":
                out_path = os.path.join(results_dir, f"{provider}_{model}.jsonl")
                if provider == "openai":
                    batch_openai.download_openai_batch_results(bid, out_path)
                elif provider == "gemini":
                    batch_gemini.download_gemini_batch_results(bid, out_path)
                elif provider == "claude":
                    inputs = os.path.abspath(os.path.join(
                        os.path.dirname(batch_ids_path), "..", "batches", f"claude_{model}_input.jsonl",
                    ))
                    claude_id_map = b.get("claude_id_map") if isinstance(b, dict) else None
                    batch_claude.download_claude_batch_results(
                        bid,
                        out_path,
                        input_jsonl_path=inputs if os.path.isfile(inputs) else None,
                        id_map=claude_id_map if isinstance(claude_id_map, dict) else None,
                    )
            elif st["status"] in ("failed", "cancelled", "expired"):
                # These states are non-retrievable; skip so ingestion can continue.
                print(f"[poll] Skipping {provider}/{model}: {st['status']} (cannot download results)")
            else:
                if wait_for_all:
                    pending.append(b)
                else:
                    # Some providers can remain in progress forever; skip unresolved jobs.
                    print(f"[poll] Skipping {provider}/{model}: {st['status']} (not ready yet)")

        if not wait_for_all:
            break
        if not pending:
            break
        unresolved = pending
        pending_labels = [f"{b['provider']}/{b['model']}" for b in unresolved]
        print(f"[poll] Waiting for {len(unresolved)} batches: {pending_labels[:5]}...")
        time.sleep(poll_interval_sec)
    else:
        if wait_for_all and unresolved:
            pending_labels = [f"{b['provider']}/{b['model']}" for b in unresolved]
            print(f"[poll] Timeout; skipping unresolved batches: {pending_labels[:5]}...")

    for b in batches:
        out_path = os.path.join(results_dir, f"{b['provider']}_{b['model']}.jsonl")
        if os.path.isfile(out_path):
            ingest_results(out_path, b["model"], b["provider"],
                           ground_truth_csv=ground_truth_csv,
                           langfuse_dataset_name=langfuse_dataset_name, run_id=run_id)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--batch-ids", default="data/batch_ids.json")
    p.add_argument("--results-dir", default="data/results")
    p.add_argument("--csv", default=None)
    p.add_argument("--langfuse-dataset", default=None, dest="langfuse_dataset_name", nargs="?", const="")
    p.add_argument("--run-id", default=None)
    p.add_argument("--interval", type=int, default=300)
    p.add_argument("--max-wait", type=int, default=86400 * 2)
    p.add_argument("--wait-for-all", action="store_true", help="Keep polling until all batches are terminal")
    args = p.parse_args()

    use_langfuse = args.csv is None
    poll_until_done(
        batch_ids_path=args.batch_ids,
        results_dir=args.results_dir,
        ground_truth_csv=args.csv,
        langfuse_dataset_name=(args.langfuse_dataset_name if args.langfuse_dataset_name is not None else "") if use_langfuse else None,
        run_id=args.run_id,
        poll_interval_sec=args.interval,
        max_wait_sec=args.max_wait,
        wait_for_all=args.wait_for_all,
    )
