"""Rewrite system prompt in Langfuse campaign dataset and submit gpt-5-nano batch.

Reads Langfuse dataset `campaign_relevance_02e1a68ccb0f`, rewrites ONLY the system
prompt in each example to use `new_prompt.txt`, keeps the JSON schema/fields
from the existing request template, and submits a Batch API job via batch_openai.
Also writes a debug JSONL with original vs rewritten bodies.
"""

import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from langfuse import Langfuse  # type: ignore

import batch_openai  # type: ignore
from scripts.lib.load_dataset import _load_body_template  # type: ignore


DATASET_NAME = "campaign_relevance_02e1a68ccb0f"
NEW_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "new_prompt.txt")


def _extract_messages(raw_input: Any) -> Tuple[List[Dict[str, Any]], bool]:
    """Return (messages, is_wrapped_body).

    - If raw_input is a full body dict with an `input` field, return that list and
      is_wrapped_body=True.
    - If raw_input is already a messages list, return it and is_wrapped_body=False.
    - Otherwise, fall back to a single user message with the JSON payload.
    """
    if isinstance(raw_input, dict) and "input" in raw_input:
        inp = raw_input["input"]
        if isinstance(inp, list):
            return deepcopy(inp), True
        return [{"role": "user", "content": json.dumps(inp, ensure_ascii=False)}], True

    if isinstance(raw_input, list):
        return deepcopy(raw_input), False

    # Fallback: arbitrary payload -> user message
    return [
        {
            "role": "user",
            "content": json.dumps(raw_input, ensure_ascii=False),
        }
    ], False


def _rewrite_system_prompt(messages: List[Dict[str, Any]], new_prompt: str) -> List[Dict[str, Any]]:
    """Replace or inject the system prompt in a messages list."""
    replaced = False
    new_messages: List[Dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system" and not replaced:
            updated = dict(msg)
            updated["content"] = new_prompt
            new_messages.append(updated)
            replaced = True
        else:
            new_messages.append(msg)

    if not replaced:
        new_messages.insert(0, {"role": "system", "content": new_prompt})

    return new_messages


def main() -> None:
    dotenv.load_dotenv()

    with open(NEW_PROMPT_PATH, "r", encoding="utf-8") as f:
        new_system_prompt = f.read().strip()

    template = _load_body_template(None)

    client = Langfuse()
    dataset = client.get_dataset(DATASET_NAME)

    data_dir = Path(ROOT) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    debug_path = data_dir / f"{DATASET_NAME}_rewritten.jsonl"

    examples: List[Dict[str, Any]] = []

    with debug_path.open("w", encoding="utf-8") as dbg_f:
        for item in dataset.items:
            cid = str((item.metadata or {}).get("custom_id") or item.id)
            if not cid:
                continue

            raw_input = item.input

            # Determine original body shape and messages.
            if isinstance(raw_input, dict) and ("text" in raw_input or "metadata" in raw_input or "input" in raw_input):
                # Assume this is already a full request body.
                original_body = deepcopy(raw_input)
                messages, _ = _extract_messages(original_body)
            else:
                # Historical pattern: dataset stored just the input messages;
                # reconstruct body from the shared template.
                messages, _ = _extract_messages(raw_input)
                original_body = {**template, "input": deepcopy(messages)}

            new_messages = _rewrite_system_prompt(messages, new_system_prompt)

            new_body = deepcopy(original_body)
            new_body["input"] = new_messages

            # Ensure the body matches the Responses API / batch payload shape.
            # This mirrors what batch_openai.create_openai_batch produces.
            if "model" not in new_body:
                new_body["model"] = "gpt-5-nano"
            if "reasoning" not in new_body:
                new_body["reasoning"] = {"effort": "low"}

            # Debug line: original vs rewritten body for sanity checks.
            dbg_f.write(
                json.dumps(
                    {
                        "custom_id": cid,
                        "original_body": original_body,
                        "new_body": new_body,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            examples.append({"custom_id": cid, "body": new_body})

    print(
        f"[rewrite_campaign_template_batch] Prepared {len(examples)} examples "
        f"from Langfuse dataset {DATASET_NAME}"
    )
    print(f"[rewrite_campaign_template_batch] Debug JSONL: {debug_path}")

    batches_dir = Path(ROOT) / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)

    batch_id = batch_openai.create_openai_batch(
        model="gpt-5-nano",
        examples=examples,
        output_dir=str(batches_dir),
    )
    print(f"[rewrite_campaign_template_batch] Submitted OpenAI batch: {batch_id}")


if __name__ == "__main__":
    main()

