"""OpenAI batch helpers – create, poll, download via Responses API."""

import json
import os
from typing import Dict, List

from openai import OpenAI

_REASONING_PREFIXES = ("gpt-5", "o3", "o4")


def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def create_openai_batch(
    model: str,
    examples: List[Dict],
    output_dir: str = "batches",
    reasoning_effort: str = None,
) -> str:
    """Create an OpenAI Batch job. Returns batch ID.

    reasoning_effort: "low"|"medium"|"high" for reasoning models. Defaults to "low".
    """
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, f"openai_{model}.jsonl")

    is_reasoning = any(model.startswith(p) for p in _REASONING_PREFIXES)
    effort = reasoning_effort or ("low" if is_reasoning else None)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ex in examples:
            body = dict(ex["body"])
            body["model"] = model
            if effort and "reasoning" not in body:
                body["reasoning"] = {"effort": effort}
            f.write(json.dumps({
                "custom_id": ex["custom_id"], "method": "POST",
                "url": "/v1/responses", "body": body,
            }) + "\n")

    client = _client()
    with open(jsonl_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=file_obj.id, endpoint="/v1/responses", completion_window="24h",
    )
    print(f"[batch_openai] Created batch {batch.id} for model {model}")
    return batch.id


def get_openai_batch_status(batch_id: str) -> Dict:
    batch = _client().batches.retrieve(batch_id)
    return {
        "id": batch.id, "status": batch.status,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
    }


def cancel_openai_batch(batch_id: str) -> Dict:
    batch = _client().batches.cancel(batch_id)
    return {
        "id": batch.id, "status": batch.status,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
    }


def download_openai_batch_results(batch_id: str, output_path: str) -> None:
    status = get_openai_batch_status(batch_id)
    if status["status"] != "completed":
        raise RuntimeError(f"Batch {batch_id} not completed (status={status['status']})")
    file_id = status["output_file_id"]
    if not file_id:
        raise RuntimeError(f"No output_file_id for batch {batch_id}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(_client().files.content(file_id).read())
    print(f"[batch_openai] Downloaded results for {batch_id} → {output_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Inspect OpenAI batch status")
    p.add_argument("batch_id")
    p.add_argument("--download", help="Path to save results JSONL")
    args = p.parse_args()

    st = get_openai_batch_status(args.batch_id)
    print(st)
    if args.download and st["status"] == "completed":
        download_openai_batch_results(args.batch_id, args.download)
