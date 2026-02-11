"""
Shared Gemini image cache: load/save url -> (uri, name, uploaded_at).
Gemini Files API files expire after 48h; we treat entries older than 47h as expired.
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict

CACHE_VERSION = 1
TTL_HOURS = 47  # 1h safety margin before 48h Gemini expiry


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_uploaded_at(entry: Dict) -> datetime | None:
    raw = entry.get("uploaded_at")
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return datetime.fromisoformat(raw)
    except (ValueError, TypeError):
        return None


def is_expired(entry: Dict) -> bool:
    """True if entry is older than TTL_HOURS (47h)."""
    dt = _parse_uploaded_at(entry)
    if dt is None:
        return True
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    return age_hours >= TTL_HOURS


def now_iso_utc() -> str:
    """Current time as ISO UTC string for uploaded_at."""
    return _now_utc()


def load_image_cache_raw(cache_path: str) -> Dict[str, Dict]:
    """
    Load full cache from JSON (no expiry filtering).
    Returns url -> { "uri", "name", "uploaded_at" } for use by upload script.
    """
    if not os.path.isfile(cache_path):
        return {}

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    entries = data.get("entries") or {}
    return {k: v for k, v in entries.items() if isinstance(v, dict)}


def load_image_cache(cache_path: str) -> Dict[str, str]:
    """
    Load cache from JSON; filter out expired entries.
    Returns url -> uri only for entries that are still valid.
    """
    raw = load_image_cache_raw(cache_path)
    result: Dict[str, str] = {}
    for url, entry in raw.items():
        uri = entry.get("uri") or entry.get("file_uri")
        if not uri:
            continue
        if is_expired(entry):
            continue
        result[url] = uri
    return result


def save_image_cache(cache_path: str, entries: Dict[str, Dict]) -> None:
    """
    Save cache to JSON atomically (write to temp then rename).
    entries: url -> { "uri": str, "name": str, "uploaded_at": str }
    """
    data = {"version": CACHE_VERSION, "entries": entries}
    dirpath = os.path.dirname(cache_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    tmp = cache_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, cache_path)
