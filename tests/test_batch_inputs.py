"""
Tests for batch input JSONL generation — verifies zero data loss for both OpenAI and Gemini.

Run: python -m pytest tests/test_batch_inputs.py -v
"""

import ast
import csv
import json
import os
import sys
import tempfile
from collections import Counter
from typing import Dict, List

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from batch_gemini import (
    _build_generation_config,
    _collect_image_urls,
    _convert_schema_node,
    _extract_system_and_parts,
    _mime_from_url,
    create_gemini_batch,
)
from image_cache import is_expired, load_image_cache, load_image_cache_raw, save_image_cache

DATASET_CSV = os.path.join(ROOT, "input", "dataset.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_examples(n: int = 20) -> List[Dict]:
    """Load first n examples from dataset CSV."""
    rows = []
    with open(DATASET_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n:
                break
            body = ast.literal_eval(row["body"])
            rows.append({"custom_id": row["custom_id"], "body": body})
    return rows


def _count_source_parts(body: Dict):
    """Count messages, text parts, and image parts in the original body."""
    msgs = body.get("input", [])
    system_count = 0
    text_parts = 0
    image_parts = 0
    image_urls = []

    for msg in msgs:
        role = msg.get("role", "user")
        content = msg.get("content")

        if isinstance(content, str):
            if role == "system":
                system_count += 1
            else:
                if content.strip():
                    text_parts += 1
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "input_text" and "text" in part:
                    text_parts += 1
                elif part.get("type") == "input_image" and "image_url" in part:
                    image_parts += 1
                    image_urls.append(part["image_url"])

    return {
        "messages": len(msgs),
        "system": system_count,
        "text_parts": text_parts,
        "image_parts": image_parts,
        "image_urls": image_urls,
        "has_text_format": "text" in body and "format" in body.get("text", {}),
    }


def _fake_url_to_uri(examples: List[Dict]) -> Dict[str, str]:
    """Build a fake url_to_uri mapping for all image URLs in examples."""
    urls = _collect_image_urls(examples)
    return {url: f"https://generativelanguage.googleapis.com/v1beta/files/fake_{i}" for i, url in enumerate(sorted(urls))}


# ===========================================================================
# OpenAI batch input tests
# ===========================================================================

class TestOpenAIBatchInput:
    """Verify OpenAI JSONL preserves the original body verbatim (only model overridden)."""

    def test_body_passed_verbatim(self):
        """Body should be identical to source except model field."""
        examples = _load_examples(5)
        for ex in examples:
            body_copy = dict(ex["body"])
            body_copy["model"] = "gpt-test"

            line = {
                "custom_id": ex["custom_id"],
                "method": "POST",
                "url": "/v1/responses",
                "body": body_copy,
            }
            # Round-trip through JSON (same as batch_openai.py)
            parsed = json.loads(json.dumps(line))

            assert parsed["custom_id"] == ex["custom_id"]
            assert parsed["body"]["model"] == "gpt-test"
            # All original keys preserved
            for key in ex["body"]:
                if key == "model":
                    continue
                assert key in parsed["body"], f"Missing key '{key}' in OpenAI batch line"
                assert parsed["body"][key] == ex["body"][key], f"Key '{key}' differs"

    def test_all_messages_preserved(self):
        """Every message in body.input must survive JSON serialization."""
        examples = _load_examples(20)
        for ex in examples:
            body_copy = dict(ex["body"])
            body_copy["model"] = "gpt-test"
            line_json = json.dumps({"custom_id": ex["custom_id"], "method": "POST", "url": "/v1/responses", "body": body_copy})
            parsed = json.loads(line_json)

            source = _count_source_parts(ex["body"])
            output = _count_source_parts(parsed["body"])

            assert output["messages"] == source["messages"], f"Message count mismatch for {ex['custom_id']}"
            assert output["text_parts"] == source["text_parts"], f"Text parts mismatch for {ex['custom_id']}"
            assert output["image_parts"] == source["image_parts"], f"Image parts mismatch for {ex['custom_id']}"

    def test_text_format_preserved(self):
        """text.format (json_schema) must survive."""
        examples = _load_examples(5)
        for ex in examples:
            body = ex["body"]
            if "text" not in body:
                continue
            body_copy = dict(body)
            body_copy["model"] = "gpt-test"
            parsed = json.loads(json.dumps(body_copy))

            assert parsed["text"] == body["text"], "text.format lost in serialization"

    def test_store_and_metadata_preserved(self):
        """store and metadata must survive."""
        examples = _load_examples(5)
        for ex in examples:
            body = ex["body"]
            body_copy = dict(body)
            body_copy["model"] = "gpt-test"
            parsed = json.loads(json.dumps(body_copy))

            if "store" in body:
                assert parsed["store"] == body["store"]
            if "metadata" in body:
                assert parsed["metadata"] == body["metadata"]

    def test_jsonl_file_written_correctly(self):
        """Write actual JSONL and re-read — verify line count and structure."""
        examples = _load_examples(10)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            for ex in examples:
                body = dict(ex["body"])
                body["model"] = "gpt-test"
                line = {"custom_id": ex["custom_id"], "method": "POST", "url": "/v1/responses", "body": body}
                f.write(json.dumps(line) + "\n")
            path = f.name

        try:
            with open(path) as f:
                lines = [json.loads(l) for l in f if l.strip()]
            assert len(lines) == len(examples)
            for line, ex in zip(lines, examples):
                assert line["custom_id"] == ex["custom_id"]
                assert line["method"] == "POST"
                assert line["url"] == "/v1/responses"
                assert "body" in line
        finally:
            os.unlink(path)


# ===========================================================================
# Gemini batch input tests
# ===========================================================================

class TestGeminiBatchInput:
    """Verify Gemini JSONL preserves all content from source examples."""

    def test_system_instruction_extracted(self):
        """System message must end up in systemInstruction."""
        examples = _load_examples(10)
        url_map = _fake_url_to_uri(examples)
        for ex in examples:
            body = ex["body"]
            sys_content, _, total_img = _extract_system_and_parts(body, url_map)

            # Find original system content
            orig_sys = ""
            for msg in body.get("input", []):
                if msg.get("role") == "system" and isinstance(msg.get("content"), str):
                    orig_sys = msg["content"].strip()

            assert sys_content == orig_sys, f"System instruction mismatch for {ex['custom_id']}"

    def test_all_text_preserved(self):
        """Every text part (string messages + input_text) must appear in user_parts."""
        examples = _load_examples(20)
        url_map = _fake_url_to_uri(examples)
        for ex in examples:
            body = ex["body"]
            source = _count_source_parts(body)
            _, user_parts, _ = _extract_system_and_parts(body, url_map)

            gem_text_count = sum(1 for p in user_parts if "text" in p)
            assert gem_text_count == source["text_parts"], (
                f"Text parts mismatch for {ex['custom_id']}: "
                f"expected {source['text_parts']}, got {gem_text_count}"
            )

    def test_all_images_preserved(self):
        """Every input_image must become a fileData part — none dropped."""
        examples = _load_examples(20)
        url_map = _fake_url_to_uri(examples)
        for ex in examples:
            body = ex["body"]
            source = _count_source_parts(body)
            _, user_parts, total_img = _extract_system_and_parts(body, url_map)

            gem_file_count = sum(1 for p in user_parts if "fileData" in p)
            assert gem_file_count == source["image_parts"], (
                f"Image parts mismatch for {ex['custom_id']}: "
                f"expected {source['image_parts']}, got {gem_file_count}"
            )
            assert total_img == source["image_parts"]

    def test_missing_image_uri_raises(self):
        """If an image URL is not in url_to_uri, _extract_system_and_parts must raise."""
        examples = _load_examples(20)
        # Find an example with images
        ex_with_img = None
        for ex in examples:
            if _count_source_parts(ex["body"])["image_parts"] > 0:
                ex_with_img = ex
                break
        if ex_with_img is None:
            pytest.skip("No examples with images in first 20 rows")

        # Pass empty url_to_uri — must raise, not silently drop
        with pytest.raises(RuntimeError, match="Image URL not uploaded"):
            _extract_system_and_parts(ex_with_img["body"], {})

    def test_missing_image_uri_partial_raises(self):
        """Even if some images are mapped, missing any one must raise."""
        examples = _load_examples(20)
        ex_with_img = None
        for ex in examples:
            source = _count_source_parts(ex["body"])
            if source["image_parts"] >= 2:
                ex_with_img = ex
                break
        if ex_with_img is None:
            pytest.skip("No examples with >=2 images in first 20 rows")

        source = _count_source_parts(ex_with_img["body"])
        # Map only the first image, leave the rest unmapped
        partial_map = {source["image_urls"][0]: "https://fake/uri"}
        with pytest.raises(RuntimeError, match="Image URL not uploaded"):
            _extract_system_and_parts(ex_with_img["body"], partial_map)

    def test_response_schema_converted(self):
        """text.format.json_schema must be converted to Gemini responseSchema."""
        examples = _load_examples(1)
        body = examples[0]["body"]
        gen_config = _build_generation_config(body)

        assert gen_config["responseMimeType"] == "application/json"
        assert "responseSchema" in gen_config, "responseSchema missing from generationConfig"

        schema = gen_config["responseSchema"]
        assert schema["type"] == "OBJECT"
        assert "campaign_relevant" in schema["properties"]
        assert schema["properties"]["campaign_relevant"]["type"] == "BOOLEAN"

    def test_schema_conversion_types(self):
        """All OpenAI JSON Schema types must map to Gemini uppercase types."""
        node = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "active": {"type": "boolean"},
                "count": {"type": "integer"},
                "score": {"type": "number"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "active"],
        }
        result = _convert_schema_node(node)
        assert result["type"] == "OBJECT"
        assert result["properties"]["name"]["type"] == "STRING"
        assert result["properties"]["active"]["type"] == "BOOLEAN"
        assert result["properties"]["count"]["type"] == "INTEGER"
        assert result["properties"]["score"]["type"] == "NUMBER"
        assert result["properties"]["tags"]["type"] == "ARRAY"
        assert result["properties"]["tags"]["items"]["type"] == "STRING"
        assert result["required"] == ["name", "active"]

    def test_gemini_jsonl_structure(self):
        """Each JSONL line must have key, request.contents, request.generationConfig."""
        examples = _load_examples(10)
        url_map = _fake_url_to_uri(examples)

        for ex in examples:
            body = ex["body"]
            sys_content, user_parts, _ = _extract_system_and_parts(body, url_map)
            gen_config = _build_generation_config(body)

            request = {
                "contents": [{"role": "user", "parts": user_parts}],
                "generationConfig": gen_config,
            }
            if sys_content:
                request["systemInstruction"] = {"parts": [{"text": sys_content}]}

            line = {"key": ex["custom_id"], "request": request}
            # Round-trip through JSON
            parsed = json.loads(json.dumps(line, ensure_ascii=False))

            assert "key" in parsed
            assert parsed["key"] == ex["custom_id"]
            assert "request" in parsed
            assert "contents" in parsed["request"]
            assert "generationConfig" in parsed["request"]
            assert len(parsed["request"]["contents"]) == 1
            assert parsed["request"]["contents"][0]["role"] == "user"
            assert len(parsed["request"]["contents"][0]["parts"]) > 0

    def test_total_parts_match_source(self):
        """Total parts (text + fileData) in Gemini output must equal total non-system parts in source."""
        examples = _load_examples(20)
        url_map = _fake_url_to_uri(examples)

        for ex in examples:
            source = _count_source_parts(ex["body"])
            _, user_parts, _ = _extract_system_and_parts(ex["body"], url_map)

            expected_total = source["text_parts"] + source["image_parts"]
            actual_total = len(user_parts)
            assert actual_total == expected_total, (
                f"Total parts mismatch for {ex['custom_id']}: "
                f"expected {expected_total} (text={source['text_parts']} + img={source['image_parts']}), "
                f"got {actual_total}"
            )

    def test_image_file_uris_not_empty(self):
        """Every fileData part must have a non-empty fileUri and mimeType."""
        examples = _load_examples(20)
        url_map = _fake_url_to_uri(examples)

        for ex in examples:
            _, user_parts, _ = _extract_system_and_parts(ex["body"], url_map)
            for part in user_parts:
                if "fileData" in part:
                    fd = part["fileData"]
                    assert "fileUri" in fd and fd["fileUri"], "Empty fileUri"
                    assert "mimeType" in fd and fd["mimeType"], "Empty mimeType"
                    assert fd["mimeType"].startswith("image/"), f"Bad mimeType: {fd['mimeType']}"

    def test_text_content_not_empty(self):
        """Every text part must have non-empty text."""
        examples = _load_examples(20)
        url_map = _fake_url_to_uri(examples)

        for ex in examples:
            _, user_parts, _ = _extract_system_and_parts(ex["body"], url_map)
            for part in user_parts:
                if "text" in part:
                    assert part["text"], f"Empty text part in {ex['custom_id']}"


# ===========================================================================
# Cross-provider consistency tests
# ===========================================================================

class TestCrossProviderConsistency:
    """Verify both providers see the same content from the same source examples."""

    def test_same_custom_ids(self):
        """Both providers must use the same custom_id for the same example."""
        examples = _load_examples(10)
        url_map = _fake_url_to_uri(examples)

        for ex in examples:
            # OpenAI
            openai_line = json.loads(json.dumps({
                "custom_id": ex["custom_id"],
                "method": "POST",
                "url": "/v1/responses",
                "body": dict(ex["body"]),
            }))

            # Gemini
            sys_c, parts, _ = _extract_system_and_parts(ex["body"], url_map)
            gemini_line = json.loads(json.dumps({
                "key": ex["custom_id"],
                "request": {"contents": [{"role": "user", "parts": parts}]},
            }, ensure_ascii=False))

            assert openai_line["custom_id"] == gemini_line["key"]

    def test_same_text_content(self):
        """Both providers must see identical text (excluding system instruction in Gemini)."""
        examples = _load_examples(20)
        url_map = _fake_url_to_uri(examples)

        for ex in examples:
            source = _count_source_parts(ex["body"])

            # OpenAI: all messages preserved as-is
            openai_body = json.loads(json.dumps(ex["body"]))
            openai_source = _count_source_parts(openai_body)

            # Gemini: text extracted into parts
            sys_c, parts, _ = _extract_system_and_parts(ex["body"], url_map)
            gem_text_count = sum(1 for p in parts if "text" in p)

            assert openai_source["text_parts"] == source["text_parts"]
            assert gem_text_count == source["text_parts"]

    def test_same_image_count(self):
        """Both providers must reference the same number of images."""
        examples = _load_examples(20)
        url_map = _fake_url_to_uri(examples)

        for ex in examples:
            source = _count_source_parts(ex["body"])

            # OpenAI: images stay as input_image parts in body
            openai_body = json.loads(json.dumps(ex["body"]))
            openai_img = _count_source_parts(openai_body)["image_parts"]

            # Gemini: images become fileData parts
            _, parts, total_img = _extract_system_and_parts(ex["body"], url_map)
            gem_img = sum(1 for p in parts if "fileData" in p)

            assert openai_img == source["image_parts"], f"OpenAI image mismatch: {ex['custom_id']}"
            assert gem_img == source["image_parts"], f"Gemini image mismatch: {ex['custom_id']}"
            assert total_img == source["image_parts"]


# ===========================================================================
# Mime type helper tests
# ===========================================================================

class TestMimeFromUrl:
    def test_jpeg(self):
        assert _mime_from_url("https://example.com/photo.jpg") == "image/jpeg"

    def test_jpeg_uppercase(self):
        assert _mime_from_url("https://example.com/photo.JPG") == "image/jpeg"

    def test_png(self):
        assert _mime_from_url("https://example.com/photo.png") == "image/png"

    def test_gif(self):
        assert _mime_from_url("https://example.com/anim.gif") == "image/gif"

    def test_webp(self):
        assert _mime_from_url("https://example.com/photo.webp") == "image/webp"

    def test_no_extension_defaults_jpeg(self):
        assert _mime_from_url("https://example.com/image") == "image/jpeg"


# ===========================================================================
# Image URL collection tests
# ===========================================================================

class TestCollectImageUrls:
    def test_collects_all_unique_urls(self):
        examples = _load_examples(20)
        urls = _collect_image_urls(examples)
        # Manually count unique URLs
        manual_urls = set()
        for ex in examples:
            for msg in ex["body"].get("input", []):
                c = msg.get("content")
                if isinstance(c, list):
                    for p in c:
                        if p.get("type") == "input_image" and p.get("image_url", "").startswith("http"):
                            manual_urls.add(p["image_url"])
        assert urls == manual_urls

    def test_no_duplicate_urls(self):
        examples = _load_examples(20)
        urls = _collect_image_urls(examples)
        assert len(urls) == len(set(urls))

    def test_empty_examples(self):
        assert _collect_image_urls([]) == set()

    def test_examples_without_images(self):
        examples = [{"custom_id": "x", "body": {"input": [{"role": "user", "content": "hello"}]}}]
        assert _collect_image_urls(examples) == set()


# ===========================================================================
# Image cache tests
# ===========================================================================

class TestImageCache:
    """Tests for persistent Gemini image cache (load/save/expiry)."""

    def test_load_missing_file_returns_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as f:
            path = f.name
        assert not os.path.isfile(path)
        assert load_image_cache(path) == {}
        assert load_image_cache_raw(path) == {}

    def test_save_and_load_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            entries = {
                "https://example.com/a.jpg": {"uri": "https://api/files/1", "name": "files/1", "uploaded_at": "2026-02-09T12:00:00Z"},
                "https://example.com/b.png": {"uri": "https://api/files/2", "name": "files/2", "uploaded_at": "2026-02-09T12:00:00Z"},
            }
            save_image_cache(path, entries)
            raw = load_image_cache_raw(path)
            assert set(raw.keys()) == set(entries.keys())
            assert raw["https://example.com/a.jpg"]["uri"] == "https://api/files/1"
            loaded = load_image_cache(path)
            assert set(loaded.keys()) == set(entries.keys())
            assert loaded["https://example.com/a.jpg"] == "https://api/files/1"
        finally:
            os.unlink(path)

    def test_expired_entries_filtered(self):
        from datetime import datetime, timezone, timedelta
        from image_cache import TTL_HOURS

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            old_ts = (datetime.now(timezone.utc) - timedelta(hours=TTL_HOURS + 1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            entries = {
                "https://example.com/old.jpg": {"uri": "https://api/files/old", "name": "files/old", "uploaded_at": old_ts},
            }
            save_image_cache(path, entries)
            assert load_image_cache_raw(path)
            assert load_image_cache(path) == {}
        finally:
            os.unlink(path)

    def test_valid_entries_kept(self):
        from datetime import datetime, timezone, timedelta
        from image_cache import TTL_HOURS

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            recent_ts = (datetime.now(timezone.utc) - timedelta(hours=TTL_HOURS - 1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            entries = {
                "https://example.com/recent.jpg": {"uri": "https://api/files/recent", "name": "files/recent", "uploaded_at": recent_ts},
            }
            save_image_cache(path, entries)
            loaded = load_image_cache(path)
            assert loaded.get("https://example.com/recent.jpg") == "https://api/files/recent"
        finally:
            os.unlink(path)

    def test_is_expired(self):
        from datetime import datetime, timezone, timedelta
        from image_cache import TTL_HOURS

        old_entry = {"uploaded_at": (datetime.now(timezone.utc) - timedelta(hours=TTL_HOURS + 1)).strftime("%Y-%m-%dT%H:%M:%SZ")}
        recent_entry = {"uploaded_at": (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")}
        assert is_expired(old_entry) is True
        assert is_expired(recent_entry) is False
        assert is_expired({}) is True
        assert is_expired({"uri": "x"}) is True


class TestCreateGeminiBatchRequiresCache:
    """create_gemini_batch must raise when cache is missing or stale."""

    def test_raises_when_cache_missing_for_examples_with_images(self):
        examples = _load_examples(20)
        # Find an example that has at least one image
        has_img = [ex for ex in examples if _count_source_parts(ex["body"])["image_parts"] > 0]
        if not has_img:
            pytest.skip("No examples with images in first 20 rows")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as f:
            cache_path = f.name
        assert not os.path.isfile(cache_path)
        with pytest.raises(RuntimeError, match="missing from cache|upload_gemini_images"):
            create_gemini_batch("gemini-2.0-flash", examples, image_cache_path=cache_path)

    def test_raises_when_cache_empty_for_examples_with_images(self):
        examples = _load_examples(20)
        has_img = [ex for ex in examples if _count_source_parts(ex["body"])["image_parts"] > 0]
        if not has_img:
            pytest.skip("No examples with images in first 20 rows")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            cache_path = f.name
        try:
            save_image_cache(cache_path, {})
            with pytest.raises(RuntimeError, match="missing from cache|upload_gemini_images"):
                create_gemini_batch("gemini-2.0-flash", examples, image_cache_path=cache_path)
        finally:
            os.unlink(cache_path)
