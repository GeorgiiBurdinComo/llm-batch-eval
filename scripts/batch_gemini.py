"""
Gemini batch helpers using native google-genai SDK.

Uses Gemini's native batch API format (NOT OpenAI-compatible endpoint):
- JSONL: {"key": "custom_id", "request": {"contents": [...], "generationConfig": {...}}}
- Submission: client.batches.create(model=model, src=uploaded.name)
- Images: loaded from persistent cache (run upload_gemini_images.py first).

Based on working reference: research_costs/scripts/create_batch_inputs.py + run_gemini_batch.py
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IMAGE_CACHE_PATH = os.path.join(ROOT, "data", "gemini_image_cache.json")


def _genai_client():
    """Create google-genai client with API key."""
    try:
        from google import genai
    except ImportError:
        raise RuntimeError(
            "Gemini batch requires google-genai. Install with: pip install google-genai"
        )
    api_key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_GENAI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY, GOOGLE_GENAI_API_KEY, or GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# Image upload helpers
# ---------------------------------------------------------------------------

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
    """Collect all image URLs from examples' Responses-API message bodies."""
    urls: Set[str] = set()
    for ex in examples:
        body = ex.get("body", {})
        for msg in body.get("input", []):
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "input_image" and part.get("image_url"):
                        url = part["image_url"]
                        if isinstance(url, str) and url.startswith("http"):
                            urls.add(url)
    return urls


# ---------------------------------------------------------------------------
# Message conversion (Responses API → native Gemini format)
# ---------------------------------------------------------------------------

def _extract_system_and_parts(
    body: Dict,
    url_to_uri: Dict[str, str],
) -> Tuple[str, List[Dict], int]:
    """
    Extract system instruction and user parts from OpenAI Responses API body.input.
    Maps input_text → {"text": ...}, input_image → {"fileData": ...} using pre-uploaded URIs.

    Raises RuntimeError if any image URL is missing from url_to_uri.
    Returns: (system_content, user_parts, total_images)
    """
    inp = body.get("input") or []
    system_content = ""
    user_parts: List[Dict] = []
    total_images = 0

    for item in inp:
        if not isinstance(item, dict) or "content" not in item:
            continue
        content = item["content"]
        role = item.get("role", "user")

        if isinstance(content, str):
            text = content.strip()
            if role == "system":
                system_content = text
            else:
                if text:
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
                        raise RuntimeError(
                            f"Image URL not uploaded to Gemini Files API: {url[:120]}. "
                            f"All images must be uploaded before building batch JSONL."
                        )
                    # REST API uses camelCase (fileData, fileUri, mimeType)
                    user_parts.append({
                        "fileData": {
                            "fileUri": uri,
                            "mimeType": _mime_from_url(url),
                        }
                    })

    return system_content, user_parts, total_images


# ---------------------------------------------------------------------------
# Schema conversion (OpenAI json_schema → Gemini responseSchema)
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "string": "STRING",
    "boolean": "BOOLEAN",
    "integer": "INTEGER",
    "number": "NUMBER",
    "array": "ARRAY",
    "object": "OBJECT",
}


def _convert_schema_node(node: Dict) -> Dict:
    """Recursively convert an OpenAI JSON Schema node to Gemini responseSchema format."""
    if not isinstance(node, dict):
        return node
    result: Dict[str, Any] = {}
    t = node.get("type", "")
    if t in _TYPE_MAP:
        result["type"] = _TYPE_MAP[t]
    if "properties" in node:
        result["properties"] = {
            k: _convert_schema_node(v) for k, v in node["properties"].items()
        }
    if "items" in node:
        result["items"] = _convert_schema_node(node["items"])
    if "required" in node:
        result["required"] = node["required"]
    if "enum" in node:
        result["enum"] = node["enum"]
    return result


def _build_generation_config(body: Dict) -> Dict:
    """Build Gemini generationConfig from the Responses API body."""
    config: Dict[str, Any] = {
        "temperature": 0,
        "maxOutputTokens": 8192,
        "responseMimeType": "application/json",
    }

    # Extract response schema from body.text.format (Responses API structured output)
    text_cfg = body.get("text", {})
    fmt = text_cfg.get("format", {})
    if fmt.get("type") == "json_schema" and "schema" in fmt:
        config["responseSchema"] = _convert_schema_node(fmt["schema"])

    return config


# ---------------------------------------------------------------------------
# Batch creation / status / download
# ---------------------------------------------------------------------------

def create_gemini_batch(
    model: str,
    examples: List[Dict],
    output_dir: str = "batches",
    image_cache_path: str = None,
) -> str:
    """
    Create a Gemini batch job using native genai SDK.

    1. Loads image url->uri from persistent cache (run upload_gemini_images.py first)
    2. Builds native Gemini batch JSONL (key/request format)
    3. Submits via client.batches.create(model=..., src=...)
    """
    from image_cache import load_image_cache

    cache_path = image_cache_path or DEFAULT_IMAGE_CACHE_PATH
    image_urls = _collect_image_urls(examples)
    url_to_uri: Dict[str, str] = load_image_cache(cache_path) if image_urls else {}

    missing = image_urls - set(url_to_uri.keys())
    if missing:
        raise RuntimeError(
            f"{len(missing)} image(s) missing from cache (expired or not uploaded). "
            f"Run: python scripts/upload_gemini_images.py\n"
            f"Cache path: {cache_path}\n"
            f"First missing: {sorted(missing)[0][:120]}"
        )

    if image_urls:
        print(f"[batch_gemini] Using {len(url_to_uri)} images from cache {cache_path}")

    client = _genai_client()

    # Build JSONL in native Gemini batch format
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, f"gemini_{model}.jsonl")

    total_images_all = 0

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ex in examples:
            body = ex["body"]
            # Raises RuntimeError if any image URL is missing from url_to_uri
            system_content, user_parts, total_img = _extract_system_and_parts(body, url_to_uri)
            generation_config = _build_generation_config(body)
            total_images_all += total_img

            request: Dict[str, Any] = {
                "contents": [{"role": "user", "parts": user_parts}],
                "generationConfig": generation_config,
            }
            if system_content:
                request["systemInstruction"] = {"parts": [{"text": system_content}]}

            # Native Gemini batch format: {"key": "...", "request": {...}}
            line = {"key": ex["custom_id"], "request": request}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"[batch_gemini] JSONL: {len(examples)} examples, {total_images_all} images — all included")

    # Upload JSONL via genai Files API
    from google.genai import types

    uploaded = client.files.upload(
        file=jsonl_path,
        config=types.UploadFileConfig(
            display_name=f"batch-{model}",
            mime_type="application/jsonl",
        ),
    )
    print(f"[batch_gemini] Uploaded {jsonl_path} → {uploaded.name}")

    # Submit batch via native genai SDK
    job = client.batches.create(
        model=model,
        src=uploaded.name,
        config={"display_name": f"benchmark-{model}-{datetime.now().strftime('%Y%m%d')}"},
    )

    print(f"[batch_gemini] Created batch {job.name} for model {model}")
    return job.name


# Map Gemini job states to simple status strings (compatible with poll_and_ingest.py)
_STATE_MAP = {
    "JOB_STATE_SUCCEEDED": "completed",
    "JOB_STATE_FAILED": "failed",
    "JOB_STATE_CANCELLED": "cancelled",
    "JOB_STATE_EXPIRED": "expired",
}


def get_gemini_batch_status(batch_id: str) -> Dict:
    """Return basic status info for a Gemini batch (interface-compatible with batch_openai)."""
    client = _genai_client()
    job = client.batches.get(name=batch_id)
    state = getattr(job.state, "name", str(job.state))
    status = _STATE_MAP.get(state, "in_progress")

    # Get output file from dest
    dest = getattr(job, "dest", None)
    output_file_id = getattr(dest, "file_name", None) if dest else None

    return {
        "id": batch_id,
        "status": status,
        "output_file_id": output_file_id,
        "error_file_id": None,
        "raw_state": state,
    }


def download_gemini_batch_results(batch_id: str, output_path: str) -> None:
    """Download completed batch results via native genai SDK."""
    status = get_gemini_batch_status(batch_id)
    if status["status"] != "completed":
        raise RuntimeError(
            f"Batch {batch_id} not completed (status={status['status']}, raw={status.get('raw_state')})"
        )

    file_id = status["output_file_id"]
    if not file_id:
        raise RuntimeError(f"No output file for batch {batch_id}")

    client = _genai_client()
    content = client.files.download(file=file_id)

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
