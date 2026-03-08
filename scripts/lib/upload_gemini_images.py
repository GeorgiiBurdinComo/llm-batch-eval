"""Upload dataset images to Gemini Files API and persist url->uri mapping in a cache."""

import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

LIB_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(LIB_DIR)
ROOT = os.path.dirname(SCRIPTS_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import batch_gemini
from image_cache import is_expired, load_image_cache, load_image_cache_raw, now_iso_utc, save_image_cache
from load_dataset import load_dataset_rows

DEFAULT_CACHE = os.path.join(ROOT, "data", "gemini_image_cache.json")
SAVE_EVERY_N = 50


def _upload_one(url: str, client) -> Tuple[str, str, str]:
    import requests

    r = requests.get(url, timeout=30, headers={"User-Agent": "Benchmark/1.0"})
    r.raise_for_status()

    ext = ".png" if ".png" in url.lower() else ".gif" if ".gif" in url.lower() else ".jpg"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(r.content)
        tmp = f.name
    try:
        up = client.files.upload(file=tmp)
        uri = getattr(up, "uri", None)
        name = getattr(up, "name", str(up))
        if not uri:
            uri = f"https://generativelanguage.googleapis.com/v1beta/{name}"
        return url, uri, name
    except Exception as e:
        raise RuntimeError(f"Failed to upload: {url[:120]}\n{e}") from e
    finally:
        os.unlink(tmp)


def ensure_images_uploaded(
    cache_path: str = DEFAULT_CACHE,
    force: bool = False,
    dry_run: bool = False,
    workers: int = 4,
    examples: Optional[List[Dict]] = None,
    csv_path: Optional[str] = None,
    langfuse_dataset_name: Optional[str] = None,
    body_template_path: Optional[str] = None,
) -> Dict[str, str]:
    """Ensure all image URLs are cached in Gemini Files API. Returns url->uri mapping."""
    if examples is not None:
        rows = [{"custom_id": e["custom_id"], "body": e["body"]} for e in examples]
    else:
        rows = [{"custom_id": r["custom_id"], "body": r["body"]}
                for r in load_dataset_rows(csv_path=csv_path, langfuse_dataset_name=langfuse_dataset_name,
                                           body_template_path=body_template_path)]

    all_urls: Set[str] = batch_gemini._collect_image_urls(rows)
    if not all_urls:
        print("[upload_gemini_images] No image URLs in dataset")
        return {}

    cache_raw = load_image_cache_raw(cache_path)
    entries: Dict[str, Dict] = dict(cache_raw)

    to_upload = [url for url in sorted(all_urls)
                 if force or not entries.get(url) or is_expired(entries[url])]

    if dry_run:
        print(f"[upload_gemini_images] Dry run: {len(to_upload)}/{len(all_urls)} would be uploaded")
        return load_image_cache(cache_path)
    if not to_upload:
        print(f"[upload_gemini_images] Cache valid: all {len(all_urls)} images cached")
        return load_image_cache(cache_path)

    client = batch_gemini._genai_client()
    print(f"[upload_gemini_images] Uploading {len(to_upload)} images (workers={workers})...")

    completed = skipped = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_upload_one, url, client): url for url in to_upload}
        for fut in as_completed(futures):
            try:
                url, uri, name = fut.result()
                entries[url] = {"uri": uri, "name": name, "uploaded_at": now_iso_utc()}
                completed += 1
                if completed % SAVE_EVERY_N == 0:
                    save_image_cache(cache_path, entries)
                    print(f"[upload_gemini_images] Progress: {completed}/{len(to_upload)}")
            except Exception as e:
                print(f"[upload_gemini_images] Skip: {str(futures[fut])[:80]}... {str(e)[:120]}")
                skipped += 1

    save_image_cache(cache_path, entries)
    msg = f"Done: {completed} uploaded"
    if skipped:
        msg += f", {skipped} skipped"
    print(f"[upload_gemini_images] {msg}")
    return load_image_cache(cache_path)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Upload dataset images to Gemini Files API")
    p.add_argument("--csv", default=None)
    p.add_argument("--langfuse-dataset", default=None, dest="langfuse_dataset_name")
    p.add_argument("--cache", default=DEFAULT_CACHE)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    ensure_images_uploaded(
        cache_path=args.cache, force=args.force, dry_run=args.dry_run,
        workers=args.workers, csv_path=args.csv, langfuse_dataset_name=args.langfuse_dataset_name,
    )
