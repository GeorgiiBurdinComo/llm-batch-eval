"""Claude batch helpers using Anthropic Message Batches API."""

import json
import os
import re
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import requests

_EFFORT_PREFIXES = ("claude-sonnet-4-6", "claude-opus-4-6", "claude-opus-4-5")


def _client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise RuntimeError("Claude batch requires anthropic SDK: pip install anthropic") from e
    return Anthropic(api_key=api_key)


def _convert_input_to_anthropic(body: Dict) -> Tuple[Optional[str], List[Dict]]:
    """Convert Responses API body.input to Anthropic system + messages."""
    system_chunks: List[str] = []
    messages: List[Dict[str, Any]] = []

    for item in body.get("input") or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role", "user")
        content = item.get("content")
        blocks: List[Dict[str, Any]] = []

        if isinstance(content, str):
            text = content.strip()
            if text:
                blocks.append({"type": "text", "text": text})
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "input_text" and "text" in part:
                    blocks.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "input_image" and isinstance(part.get("image_url"), str):
                    blocks.append({"type": "image", "source": {"type": "url", "url": part["image_url"]}})

        if not blocks:
            continue
        if role == "system":
            system_chunks.extend(b["text"] for b in blocks if b.get("type") == "text")
        else:
            messages.append({"role": "user" if role == "user" else "assistant", "content": blocks})

    return ("\n\n".join(system_chunks) if system_chunks else None), messages


def _sanitize_custom_id(raw: str) -> str:
    """Sanitize to match Anthropic's ^[a-zA-Z0-9_-]{1,64}$."""
    s = re.sub(r"[^a-zA-Z0-9_-]", "-", str(raw)) or "id"
    if len(s) <= 64:
        return s
    return f"{s[:55]}-{hashlib.sha1(s.encode()).hexdigest()[:8]}"


def _build_output_config(body: Dict, effort: Optional[str] = None) -> Optional[Dict]:
    cfg: Dict[str, Any] = {}
    fmt = body.get("text", {}).get("format", {})
    if fmt.get("type") == "json_schema" and "schema" in fmt:
        cfg["format"] = {"type": "json_schema", "schema": fmt["schema"]}
    if effort:
        cfg["effort"] = effort
    return cfg or None


def create_claude_batch(
    model: str,
    examples: List[Dict],
    output_dir: str = "batches",
    effort: Optional[str] = None,
    max_tokens: int = 1024,
) -> str:
    """Create a Claude Message Batch. Returns batch ID.

    effort: "low"|"medium"|"high"|"max". Defaults to "low" for supported models.
    """
    os.makedirs(output_dir, exist_ok=True)
    supports_effort = any(model.startswith(p) for p in _EFFORT_PREFIXES)
    eff_effort = effort or ("low" if supports_effort else None)

    input_jsonl_path = os.path.join(output_dir, f"claude_{model}_input.jsonl")
    records: List[Tuple[str, str, Dict[str, Any]]] = []
    with open(input_jsonl_path, "w", encoding="utf-8") as f:
        for ex in examples:
            orig_id = str(ex["custom_id"])
            san_id = _sanitize_custom_id(orig_id)
            records.append((orig_id, san_id, ex["body"]))
            f.write(json.dumps({"custom_id": orig_id, "sanitized_id": san_id, "body": ex["body"]}) + "\n")

    client = _client()
    payload: List[Dict[str, Any]] = []

    for orig_id, san_id, body in records:
        system, messages = _convert_input_to_anthropic(body)
        if not messages:
            continue
        params: Dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if system:
            params["system"] = system
        out_cfg = _build_output_config(body, effort=eff_effort)
        if out_cfg:
            params["output_config"] = out_cfg
        payload.append({"custom_id": san_id, "params": params})

    if not payload:
        raise RuntimeError("No valid Claude requests could be built from examples")

    batch = client.messages.batches.create(requests=payload)
    print(f"[batch_claude] Created batch {batch.id} for model {model} ({len(payload)} requests)")
    return batch.id


def get_claude_batch_status(batch_id: str) -> Dict:
    batch = _client().messages.batches.retrieve(message_batch_id=batch_id)
    ps = getattr(batch, "processing_status", None) or getattr(batch, "processingStatus", None)
    return {
        "id": batch_id,
        "status": "completed" if ps == "ended" else "in_progress",
        "processing_status": ps,
        "request_counts": getattr(batch, "request_counts", None),
    }


def download_claude_batch_results(
    batch_id: str,
    output_path: str,
    input_jsonl_path: Optional[str] = None,
    id_map: Optional[Dict[str, Any]] = None,
) -> None:
    """Download batch results as normalized JSONL."""
    status = get_claude_batch_status(batch_id)
    if status["status"] != "completed":
        raise RuntimeError(f"Batch {batch_id} not completed (status={status['status']})")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    # id_map can come from:
    # - caller-provided mapping (e.g. manifest): {sanitized_id: original_custom_id}
    # - input_jsonl_path records (preferred because it also contains body): {sanitized_id: {"original_custom_id": ..., "body": ...}}
    resolved_map: Dict[str, Any] = dict(id_map or {})
    if input_jsonl_path and os.path.isfile(input_jsonl_path):
        with open(input_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                san = obj.get("sanitized_id") or obj.get("custom_id")
                if san:
                    resolved_map[str(san)] = {"original_custom_id": obj.get("custom_id"), "body": obj.get("body")}

    resp = requests.get(
        f"https://api.anthropic.com/v1/messages/batches/{batch_id}/results",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        stream=True, timeout=60,
    )
    resp.raise_for_status()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f_out:
        for raw in resp.iter_lines():
            if not raw:
                continue
            item = json.loads(raw)
            result = item.get("result") or {}
            message = result.get("message") or {} if result.get("type") == "succeeded" else {"result_type": result.get("type"), "error": result}

            mapping = resolved_map.get(str(item.get("custom_id"))) or {}
            if isinstance(mapping, str):
                mapping = {"original_custom_id": mapping}
            out: Dict[str, Any] = {
                "custom_id": mapping.get("original_custom_id", item.get("custom_id")),
                "response": message,
            }
            if mapping.get("body") is not None:
                out["body"] = mapping["body"]
            f_out.write(json.dumps(out) + "\n")
            count += 1

    print(f"[batch_claude] Downloaded results for {batch_id} → {output_path} ({count} lines)")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Inspect Claude batch status")
    p.add_argument("batch_id")
    p.add_argument("--download", help="Path to save normalized results JSONL")
    p.add_argument("--inputs", dest="input_jsonl_path", default=None)
    args = p.parse_args()

    st = get_claude_batch_status(args.batch_id)
    print(st)
    if args.download and st["status"] == "completed":
        download_claude_batch_results(args.batch_id, args.download, input_jsonl_path=args.input_jsonl_path)
