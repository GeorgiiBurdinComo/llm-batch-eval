"""
OpenAI batch helpers – simple, flat functions.

Usage pattern:
    batch_id = create_openai_batch(model, examples)
    status = get_openai_batch_status(batch_id)
    if status["status"] == "completed":
        download_openai_batch_results(batch_id, "results/openai_<model>.jsonl")
"""

import json
import os
from typing import Dict, List

from openai import OpenAI


def _openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def create_openai_batch(model: str, examples: List[Dict], output_dir: str = "batches") -> str:
    """
    Create an OpenAI Batch job for the given model and examples.

    Each example must be a dict with:
      - custom_id: unique id (string)
      - body: dict matching the Responses API body (we override 'model')
    """
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, f"openai_{model}.jsonl")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ex in examples:
            body = dict(ex["body"])  # shallow copy
            body["model"] = model
            line = {
                "custom_id": ex["custom_id"],
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
            f.write(json.dumps(line) + "\n")

    client = _openai_client()

    with open(jsonl_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )

    print(f"[batch_openai] Created batch {batch.id} for model {model}")
    return batch.id


def get_openai_batch_status(batch_id: str) -> Dict:
    """Return basic status info for a batch."""
    client = _openai_client()
    batch = client.batches.retrieve(batch_id)
    return {
        "id": batch.id,
        "status": batch.status,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
    }


def download_openai_batch_results(batch_id: str, output_path: str) -> None:
    """
    Download the completed batch results JSONL to output_path.
    Raises if batch is not completed.
    """
    status = get_openai_batch_status(batch_id)
    if status["status"] != "completed":
        raise RuntimeError(f"Batch {batch_id} not completed (status={status['status']})")

    client = _openai_client()
    file_id = status["output_file_id"]
    if not file_id:
        raise RuntimeError(f"No output_file_id for batch {batch_id}")

    resp = client.files.content(file_id)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(resp.read())

    print(f"[batch_openai] Downloaded results for {batch_id} → {output_path}")


if __name__ == "__main__":
    # Tiny smoke test wrapper (not for prod; real use via run_eval.py)
    import argparse

    p = argparse.ArgumentParser(description="Inspect OpenAI batch status")
    p.add_argument("batch_id")
    p.add_argument("--download", help="Path to save results JSONL")
    args = p.parse_args()

    st = get_openai_batch_status(args.batch_id)
    print(st)
    if args.download and st["status"] == "completed":
        download_openai_batch_results(args.batch_id, args.download)

