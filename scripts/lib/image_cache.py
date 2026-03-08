"""Gemini image cache: url -> (uri, name, uploaded_at). Files expire after 48h."""

import json
import os
from datetime import datetime, timezone
from typing import Dict

CACHE_VERSION = 1
TTL_HOURS = 47


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ts(entry: Dict) -> datetime | None:
    raw = entry.get("uploaded_at")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00") if raw.endswith("Z") else raw)
    except (ValueError, TypeError):
        return None


def is_expired(entry: Dict) -> bool:
    dt = _parse_ts(entry)
    if dt is None:
        return True
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt).total_seconds() / 3600 >= TTL_HOURS


def load_image_cache_raw(cache_path: str) -> Dict[str, Dict]:
    if not os.path.isfile(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return {k: v for k, v in (data.get("entries") or {}).items() if isinstance(v, dict)}


def load_image_cache(cache_path: str) -> Dict[str, str]:
    """Load cache, filtering out expired entries. Returns url -> uri."""
    result: Dict[str, str] = {}
    for url, entry in load_image_cache_raw(cache_path).items():
        uri = entry.get("uri") or entry.get("file_uri")
        if uri and not is_expired(entry):
            result[url] = uri
    return result


def save_image_cache(cache_path: str, entries: Dict[str, Dict]) -> None:
    dirpath = os.path.dirname(cache_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    tmp = cache_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"version": CACHE_VERSION, "entries": entries}, f, indent=2, ensure_ascii=False)
    os.replace(tmp, cache_path)
