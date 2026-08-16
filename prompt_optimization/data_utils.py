"""Load Langfuse dataset, convert to GEPA examples, and create train/val/test splits."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "campaign_relevance_02e1a68ccb0f"
SPLIT_SEED = 42
# Larger val/test slices reduce noisy val overfit (strategy 4).
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20
VAL_MARGIN_FOR_SELECTION = 0.02  # pick best test among candidates within this val gap


def _to_multimodal_input(raw_input: Any) -> Dict[str, Any]:
    text_parts: List[str] = []
    image_urls: List[str] = []
    raw_user_messages: List[Dict[str, Any]] = []

    if isinstance(raw_input, list):
        for msg in raw_input:
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            raw_user_messages.append(msg)
            content = msg.get("content")
            if isinstance(content, str):
                txt = content.strip()
                if txt:
                    text_parts.append(txt)
                continue
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")
                    if block_type == "input_text":
                        txt = str(block.get("text") or "").strip()
                        if txt:
                            text_parts.append(txt)
                    elif block_type == "input_image":
                        image_url = block.get("image_url")
                        if image_url:
                            image_urls.append(str(image_url))
        return {
            "text_parts": text_parts,
            "image_urls": image_urls,
            "raw_user_messages": raw_user_messages,
        }

    return {
        "text_parts": [json.dumps(raw_input, ensure_ascii=False)],
        "image_urls": [],
        "raw_user_messages": [],
    }


def _label_to_gold(label: bool) -> str:
    return "TRUE" if label else "FALSE"


def load_gepa_examples(dataset_name: str = DEFAULT_DATASET) -> List[Dict[str, Any]]:
    load_dotenv(ROOT / ".env")
    from langfuse import Langfuse

    dataset = Langfuse().get_dataset(dataset_name)
    examples: List[Dict[str, Any]] = []
    for item in dataset.items:
        cid = str((item.metadata or {}).get("custom_id") or item.id)
        if not cid:
            continue
        if item.expected_output and isinstance(item.expected_output, dict):
            label = bool(item.expected_output.get("campaign_relevant", False))
        else:
            label = False
        if item.input is None:
            continue
        mm_input = _to_multimodal_input(item.input)
        examples.append(
            {
                "input": mm_input,
                "answer": _label_to_gold(label),
                "additional_context": {
                    "custom_id": cid,
                    "num_text_parts": str(len(mm_input["text_parts"])),
                    "num_images": str(len(mm_input["image_urls"])),
                },
            }
        )
    return examples


def extract_seed_system_prompt(dataset_name: str = DEFAULT_DATASET) -> str:
    load_dotenv(ROOT / ".env")
    from langfuse import Langfuse

    dataset = Langfuse().get_dataset(dataset_name)
    for item in dataset.items:
        inp = item.input
        if not isinstance(inp, list):
            continue
        for msg in inp:
            if isinstance(msg, dict) and msg.get("role") == "system":
                return str(msg.get("content") or "").strip()
    return ""


def stratified_split(
    examples: List[Dict[str, Any]],
    seed: int = SPLIT_SEED,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    import random

    if abs(train_ratio + val_ratio + TEST_RATIO - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    rng = random.Random(seed)
    by_label: Dict[str, List[Dict[str, Any]]] = {"TRUE": [], "FALSE": []}
    for ex in examples:
        by_label[ex["answer"]].append(ex)

    train, val, test = [], [], []
    for label, rows in by_label.items():
        rng.shuffle(rows)
        n = len(rows)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        train.extend(rows[:n_train])
        val.extend(rows[n_train : n_train + n_val])
        test.extend(rows[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def save_splits(
    train: List[Dict[str, Any]],
    val: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    out_dir: Path,
    dataset_name: str,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_name": dataset_name,
        "seed": SPLIT_SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "counts": {"train": len(train), "val": len(val), "test": len(test), "total": len(train) + len(val) + len(test)},
    }
    for name, rows in [("train", train), ("val", val), ("test", test)]:
        with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    with open(out_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_splits(split_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    train = json.loads((split_dir / "train.json").read_text(encoding="utf-8"))
    val = json.loads((split_dir / "val.json").read_text(encoding="utf-8"))
    test = json.loads((split_dir / "test.json").read_text(encoding="utf-8"))
    return train, val, test


def split_manifest_matches(manifest: dict) -> bool:
    return (
        manifest.get("train_ratio") == TRAIN_RATIO
        and manifest.get("val_ratio") == VAL_RATIO
        and manifest.get("test_ratio") == TEST_RATIO
        and manifest.get("seed") == SPLIT_SEED
    )


def prepare_splits(split_dir: Path, dataset_name: str, force: bool = False) -> dict:
    manifest_path = split_dir / "split_manifest.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if split_manifest_matches(manifest):
            return manifest
    examples = load_gepa_examples(dataset_name)
    train, val, test = stratified_split(examples)
    return save_splits(train, val, test, split_dir, dataset_name)
