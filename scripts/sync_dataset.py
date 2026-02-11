"""
Sync dataset CSV to Langfuse for versioning and evaluation.
Run once to upload the 1200 examples. Uses dataset hash for version name.

Requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY (env or .env).
Optional: LANGFUSE_HOST or LANGFUSE_BASE_URL for self-hosted.
"""

import ast
import csv
import hashlib
import os
import sys
from typing import Optional
from dotenv import load_dotenv
load_dotenv()


def sync_dataset_to_langfuse(
    csv_path: str,
    dataset_name: str = "campaign_relevance",
    langfuse_public_key: Optional[str] = None,
    langfuse_secret_key: Optional[str] = None,
    langfuse_host: Optional[str] = None,
) -> str:
    """Sync CSV to Langfuse dataset. Returns versioned dataset name (Langfuse SDK v3)."""
    from langfuse import Langfuse

    public_key = (langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip().strip('"')
    secret_key = (langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY") or "").strip().strip('"')
    if not public_key or not secret_key:
        raise RuntimeError(
            "Missing Langfuse credentials. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY "
            "(env vars or in .env at project root). See README → API keys and secrets."
        )

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with open(csv_path, "rb") as f:
        dataset_hash = hashlib.sha256(f.read()).hexdigest()[:12]
    versioned_name = f"{dataset_name}_{dataset_hash}"

    base_url = langfuse_host or os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL")
    lf = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
    )
    try:
        lf.get_dataset(versioned_name)
        print(f"Dataset {versioned_name} already exists, skipping upload")
        return versioned_name
    except Exception:
        pass

    lf.create_dataset(name=versioned_name)
    count = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            try:
                body = ast.literal_eval(row["body"])
            except (ValueError, SyntaxError):
                continue
            expected = str(row.get("campaign_relevant", "")).strip().lower() == "true"
            lf.create_dataset_item(
                dataset_name=versioned_name,
                input=body.get("input", []),
                expected_output={"campaign_relevant": expected},
                metadata={
                    "custom_id": row.get("custom_id"),
                    "row_index": idx,
                    "relevancy_reasoning": row.get("relevancy_reasoning", ""),
                },
            )
            count += 1
    print(f"Uploaded {count} items to Langfuse dataset: {versioned_name}")
    return versioned_name


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="input/dataset.csv")
    p.add_argument("--name", default="campaign_relevance")
    args = p.parse_args()
    if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sync_dataset_to_langfuse(args.csv, args.name)
