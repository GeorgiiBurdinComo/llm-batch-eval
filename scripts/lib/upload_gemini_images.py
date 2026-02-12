"""
Upload dataset images to Gemini Files API and persist url->uri mapping in a cache.

Used by run_eval; can also be run directly:
  python scripts/lib/upload_gemini_images.py
  python scripts/lib/upload_gemini_images.py --dry-run
  python scripts/lib/upload_gemini_images.py --force --workers 8
"""

import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

# scripts/lib -> scripts -> project root
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

DEFAULT_CSV = os.path.join(ROOT, "input", "dataset.csv")
DEFAULT_CACHE = os.path.join(ROOT, "data", "gemini_image_cache.json")
SAVE_EVERY_N = 50


def _upload_one(url: str, client) -> Tuple[str, str, str]:
    """
    Download image from URL and upload to Gemini Files API.
    Returns (url, uri, name). Raises on failure.
    """
    import requests

    r = requests.get(url, timeout=30, headers={"User-Agent": "Benchmark/1.0"})
    r.raise_for_status()

    ext = ".jpg"
    if ".png" in url.lower():
        ext = ".png"
    elif ".gif" in url.lower():
        ext = ".gif"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(r.content)
        tmp = f.name
    try:
        up = client.files.upload(file=tmp)
        uri = getattr(up, "uri", None)
        name = getattr(up, "name", str(up))
        if not uri:
            uri = f"https://generativelanguage.googleapis.com/v1beta/{name}"
        return (url, uri, name)
    except Exception as e:
        raise RuntimeError(f"Failed to upload image: {url[:120]}\n{e}") from e
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
    """
    Ensure all image URLs from the dataset are in the Gemini Files API cache.
    Uploads missing or expired entries; saves cache periodically and at end.

    Examples: if provided, use these rows (e.g. from run_eval subset). Else load from
    csv_path (if set) or from Langfuse (langfuse_dataset_name or env/default).

    Returns url -> uri mapping (valid entries only). Raises on any upload failure.
    """
    if examples is not None:
        # Use provided rows (e.g. from run_eval); need {custom_id, body}
        examples = [{"custom_id": e["custom_id"], "body": e["body"]} for e in examples]
    else:
        rows = load_dataset_rows(
            csv_path=csv_path,
            langfuse_dataset_name=langfuse_dataset_name,
            body_template_path=body_template_path,
        )
        examples = [{"custom_id": r["custom_id"], "body": r["body"]} for r in rows]
    all_urls: Set[str] = batch_gemini._collect_image_urls(examples)
    if not all_urls:
        print("[upload_gemini_images] No image URLs in dataset")
        return {}

    cache_raw = load_image_cache_raw(cache_path)
    # Working copy we will update
    entries: Dict[str, Dict] = dict(cache_raw)

    urls_to_upload: List[str] = []
    for url in sorted(all_urls):
        if force:
            urls_to_upload.append(url)
            continue
        entry = entries.get(url)
        if not entry:
            urls_to_upload.append(url)
            continue
        if is_expired(entry):
            urls_to_upload.append(url)
            continue

    if dry_run:
        print(f"[upload_gemini_images] Dry run: {len(urls_to_upload)}/{len(all_urls)} images would be uploaded")
        return load_image_cache(cache_path)
    if not urls_to_upload:
        print(f"[upload_gemini_images] Cache valid: all {len(all_urls)} images already cached")
        return load_image_cache(cache_path)

    client = batch_gemini._genai_client()
    total = len(urls_to_upload)
    print(f"[upload_gemini_images] Uploading {total} images to Gemini Files API (workers={workers})...")

    import requests
    completed = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_upload_one, url, client): url for url in urls_to_upload}
        for fut in as_completed(futures):
            try:
                url, uri, name = fut.result()
            except requests.exceptions.HTTPError as e:
                url = futures.get(fut, "?")
                print(f"[upload_gemini_images] Skip (403/error): {str(url)[:80]}...")
                skipped += 1
                continue
            entries[url] = {
                "uri": uri,
                "name": name,
                "uploaded_at": now_iso_utc(),
            }
            completed += 1
            if completed % 50 == 0:
                save_image_cache(cache_path, entries)
                print(f"[upload_gemini_images] Progress: {completed}/{total} (cache saved)")

    save_image_cache(cache_path, entries)
    print(f"[upload_gemini_images] Done: {completed} uploaded" + (f", {skipped} skipped (403/error)" if skipped else "") + f", cache saved to {cache_path}")

    return load_image_cache(cache_path)


def main():
    import argparse

    p = argparse.ArgumentParser(description="Upload dataset images to Gemini Files API and cache mappings")
    p.add_argument("--csv", default=None, help="Use CSV; if omitted, use Langfuse dataset")
    p.add_argument("--langfuse-dataset", default=None, dest="langfuse_dataset_name", help="Langfuse dataset name (default from env)")
    p.add_argument("--cache", default=DEFAULT_CACHE, help="Cache JSON path")
    p.add_argument("--force", action="store_true", help="Re-upload all images")
    p.add_argument("--dry-run", action="store_true", help="Only report what would be uploaded")
    p.add_argument("--workers", type=int, default=4, help="Parallel upload workers")
    args = p.parse_args()

    ensure_images_uploaded(
        cache_path=args.cache,
        force=args.force,
        dry_run=args.dry_run,
        workers=args.workers,
        csv_path=args.csv,
        langfuse_dataset_name=args.langfuse_dataset_name,
    )


if __name__ == "__main__":
    main()
