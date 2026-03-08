"""Gemini batch helpers using native google-genai SDK."""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IMAGE_CACHE_PATH = os.path.join(ROOT, "data", "gemini_image_cache.json")

_THINKING_PREFIXES = ("gemini-2.5", "gemini-3")

_TYPE_MAP = {
    "string": "STRING", "boolean": "BOOLEAN", "integer": "INTEGER",
    "number": "NUMBER", "array": "ARRAY", "object": "OBJECT",
}


def _genai_client():
    try:
        from google import genai
    except ImportError:
        raise RuntimeError("Gemini batch requires google-genai. Install with: pip install google-genai")
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY, GOOGLE_GENAI_API_KEY, or GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


def _mime_from_url(url: str) -> str:
    low = url.lower()
    if ".png" in low:
        return "image/png"
    if ".gif" in low:
        return "image/gif"
    if ".webp" in low:
        return "image/webp"
    return "image/jpeg"


def _collect_image_urls(examples: List[Dict]) -> Set[str]:
    urls: Set[str] = set()
    for ex in examples:
        for msg in ex.get("body", {}).get("input", []):
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "input_image" and part.get("image_url"):
                        url = part["image_url"]
                        if isinstance(url, str) and url.startswith("http"):
                            urls.add(url)
    return urls


def _example_uses_urls(example: Dict, urls: Set[str]) -> bool:
    for msg in example.get("body", {}).get("input", []):
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "input_image" and part.get("image_url") in urls:
                    return True
    return False


def _extract_system_and_parts(body: Dict, url_to_uri: Dict[str, str]) -> Tuple[str, List[Dict], int]:
    """Extract system instruction and user parts from Responses API body.input."""
    system = ""
    user_parts: List[Dict] = []
    total_images = 0

    for item in body.get("input") or []:
        if not isinstance(item, dict) or "content" not in item:
            continue
        content = item["content"]
        role = item.get("role", "user")

        if isinstance(content, str):
            text = content.strip()
            if role == "system":
                system = text
            elif text:
                user_parts.append({"text": text})
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "input_text" and "text" in part:
                    user_parts.append({"text": part["text"]})
                elif part.get("type") == "input_image" and "image_url" in part:
                    total_images += 1
                    url = part["image_url"]
                    uri = url_to_uri.get(url)
                    if not uri:
                        raise RuntimeError(f"Image URL not uploaded: {url[:120]}")
                    user_parts.append({"fileData": {"fileUri": uri, "mimeType": _mime_from_url(url)}})

    return system, user_parts, total_images


def _convert_schema_node(node: Dict) -> Dict:
    if not isinstance(node, dict):
        return node
    result: Dict[str, Any] = {}
    if node.get("type") in _TYPE_MAP:
        result["type"] = _TYPE_MAP[node["type"]]
    if "properties" in node:
        result["properties"] = {k: _convert_schema_node(v) for k, v in node["properties"].items()}
    if "items" in node:
        result["items"] = _convert_schema_node(node["items"])
    if "required" in node:
        result["required"] = node["required"]
    if "enum" in node:
        result["enum"] = node["enum"]
    return result


def _build_generation_config(body: Dict, *, disable_thinking: bool = False, max_output_tokens: int = 4096) -> Dict:
    config: Dict[str, Any] = {
        "temperature": 0,
        "maxOutputTokens": max_output_tokens,
        "responseMimeType": "application/json",
    }
    if disable_thinking:
        config["thinkingConfig"] = {"thinkingBudget": 0}

    fmt = body.get("text", {}).get("format", {})
    if fmt.get("type") == "json_schema" and "schema" in fmt:
        config["responseSchema"] = _convert_schema_node(fmt["schema"])
    return config


def create_gemini_batch(
    model: str,
    examples: List[Dict],
    output_dir: str = "batches",
    image_cache_path: str = None,
    disable_thinking: bool = None,
    max_output_tokens: int = 4096,
) -> str:
    """Create a Gemini batch job. Returns batch name/ID."""
    from image_cache import load_image_cache

    cache_path = image_cache_path or DEFAULT_IMAGE_CACHE_PATH
    image_urls = _collect_image_urls(examples)
    url_to_uri: Dict[str, str] = load_image_cache(cache_path) if image_urls else {}

    missing = image_urls - set(url_to_uri.keys())
    if missing:
        n_before = len(examples)
        examples = [ex for ex in examples if not _example_uses_urls(ex, missing)]
        image_urls = _collect_image_urls(examples)
        still_missing = image_urls - set(url_to_uri.keys())
        if still_missing:
            raise RuntimeError(f"{len(still_missing)} image(s) still missing. Cache: {cache_path}")
        if not examples:
            raise RuntimeError("All examples use missing image URLs; cannot create batch")
        print(f"[batch_gemini] Skipped {n_before - len(examples)} example(s) with inaccessible images")

    if image_urls:
        print(f"[batch_gemini] Using {len(url_to_uri)} images from cache {cache_path}")

    client = _genai_client()
    is_thinking = any(model.startswith(p) for p in _THINKING_PREFIXES)
    eff_disable = disable_thinking if disable_thinking is not None else is_thinking

    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, f"gemini_{model}.jsonl")
    total_images = 0

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ex in examples:
            body = ex["body"]
            sys_content, user_parts, n_img = _extract_system_and_parts(body, url_to_uri)
            gen_config = _build_generation_config(body, disable_thinking=eff_disable, max_output_tokens=max_output_tokens)
            total_images += n_img

            request: Dict[str, Any] = {
                "contents": [{"role": "user", "parts": user_parts}],
                "generationConfig": gen_config,
            }
            if sys_content:
                request["systemInstruction"] = {"parts": [{"text": sys_content}]}

            f.write(json.dumps({"key": ex["custom_id"], "request": request}, ensure_ascii=False) + "\n")

    print(f"[batch_gemini] JSONL: {len(examples)} examples, {total_images} images")

    from google.genai import types
    uploaded = client.files.upload(
        file=jsonl_path,
        config=types.UploadFileConfig(display_name=f"batch-{model}", mime_type="application/jsonl"),
    )
    print(f"[batch_gemini] Uploaded {jsonl_path} → {uploaded.name}")

    job = client.batches.create(
        model=model, src=uploaded.name,
        config={"display_name": f"benchmark-{model}-{datetime.now().strftime('%Y%m%d')}"},
    )
    print(f"[batch_gemini] Created batch {job.name} for model {model}")
    return job.name


_STATE_MAP = {
    "JOB_STATE_SUCCEEDED": "completed",
    "JOB_STATE_FAILED": "failed",
    "JOB_STATE_CANCELLED": "cancelled",
    "JOB_STATE_EXPIRED": "expired",
}


def get_gemini_batch_status(batch_id: str) -> Dict:
    client = _genai_client()
    job = client.batches.get(name=batch_id)
    state = getattr(job.state, "name", str(job.state))
    dest = getattr(job, "dest", None)
    return {
        "id": batch_id,
        "status": _STATE_MAP.get(state, "in_progress"),
        "output_file_id": getattr(dest, "file_name", None) if dest else None,
        "error_file_id": None,
        "raw_state": state,
    }


def cancel_gemini_batch(batch_id: str) -> Dict:
    client = _genai_client()
    try:
        client.batches.cancel(name=batch_id)
    except Exception:
        pass
    try:
        job = client.batches.get(name=batch_id)
        state = getattr(job.state, "name", str(job.state))
        status = _STATE_MAP.get(state, "in_progress")
    except Exception:
        state, status = None, "unknown"
    return {"id": batch_id, "status": status, "raw_state": state}


def download_gemini_batch_results(batch_id: str, output_path: str) -> None:
    status = get_gemini_batch_status(batch_id)
    if status["status"] != "completed":
        raise RuntimeError(f"Batch {batch_id} not completed (status={status['status']}, raw={status.get('raw_state')})")

    file_id = status["output_file_id"]
    if not file_id:
        raise RuntimeError(f"No output file for batch {batch_id}")

    content = _genai_client().files.download(file=file_id)
    if hasattr(content, "read"):
        data = content.read()
    elif isinstance(content, bytes):
        data = content
    elif isinstance(content, str):
        data = content.encode("utf-8")
    else:
        data = bytes(content) if content else b""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(data)
    print(f"[batch_gemini] Downloaded results for {batch_id} → {output_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Inspect Gemini batch status")
    p.add_argument("batch_id")
    p.add_argument("--download", help="Path to save results JSONL")
    args = p.parse_args()

    st = get_gemini_batch_status(args.batch_id)
    print(st)
    if args.download and st["status"] == "completed":
        download_gemini_batch_results(args.batch_id, args.download)
