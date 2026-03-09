"""Orchestrator: load eval subset from Langfuse, submit batches for selected models, write batch_ids."""

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_lib_dir = os.path.join(_scripts_dir, "lib")
for _d in (_scripts_dir, _lib_dir):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import yaml
from load_dataset import load_dataset_rows
import batch_openai
import batch_gemini
import batch_claude
from upload_gemini_images import ensure_images_uploaded


def load_models_config(config_path: str = None) -> dict:
    with open(config_path or os.path.join(ROOT, "config", "models.yaml"), "r") as f:
        return yaml.safe_load(f)


def run_submit(
    models: list = None,
    langfuse_dataset_name: str = None,
    csv_path: str = None,
    body_template_path: str = None,
    batches_dir: str = "batches",
    batch_ids_path: str = "data/batch_ids.json",
    run_id: str = None,
) -> str:
    results_dir = os.path.join(ROOT, "data", "results")
    if os.path.isdir(results_dir):
        shutil.rmtree(results_dir)
        print(f"[run_eval] Cleaned {results_dir}")

    cfg = load_models_config()
    run_id = run_id or os.getenv("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")

    if models is None:
        models = list(cfg.get("openai", [])) + list(cfg.get("gemini", [])) + list(cfg.get("claude", []))

    provider_sets = {
        "openai": set(cfg.get("openai", [])),
        "gemini": set(cfg.get("gemini", [])),
        "claude": set(cfg.get("claude", [])),
    }
    create_fns = {
        "openai": lambda m, s: batch_openai.create_openai_batch(m, s, output_dir=batches_dir),
        "gemini": lambda m, s: batch_gemini.create_gemini_batch(m, s, output_dir=batches_dir),
        "claude": lambda m, s: batch_claude.create_claude_batch(m, s, output_dir=batches_dir),
    }

    subset = load_dataset_rows(
        csv_path=csv_path,
        langfuse_dataset_name=langfuse_dataset_name,
        body_template_path=body_template_path,
    )
    if not subset:
        raise RuntimeError("Empty dataset; check dataset source")

    print(f"[run_eval] Loaded {len(subset)} examples from dataset")

    os.makedirs(os.path.dirname(batch_ids_path) or ".", exist_ok=True)

    if any(m in provider_sets["gemini"] for m in models):
        ensure_images_uploaded(examples=subset)

    batch_ids = []
    for model in models:
        provider = next((p for p, s in provider_sets.items() if model in s), None)
        if not provider:
            print(f"[run_eval] Skip unknown model: {model}")
            continue
        try:
            bid = create_fns[provider](model, subset)
            batch_ids.append({"model": model, "provider": provider, "batch_id": bid})
        except Exception as e:
            print(f"[run_eval] Failed {model}: {e}")
            batch_ids.append({"model": model, "provider": provider, "batch_id": None, "error": str(e)})

    with open(batch_ids_path, "w") as f:
        json.dump({"run_id": run_id, "subset_size": len(subset), "batches": batch_ids}, f, indent=2)
    print(f"[run_eval] Wrote {batch_ids_path} (run_id={run_id})")
    return batch_ids_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--models", type=str)
    p.add_argument("--csv", default=None)
    p.add_argument("--langfuse-dataset", default=None, dest="langfuse_dataset_name",
                    help="Langfuse dataset name containing the eval subset")
    p.add_argument("--body-template", default=None, dest="body_template_path")
    p.add_argument("--batch-ids", default="data/batch_ids.json")
    p.add_argument("--run-id", default=None)
    args = p.parse_args()

    run_submit(
        models=[m.strip() for m in args.models.split(",")] if args.models else None,
        csv_path=args.csv, langfuse_dataset_name=args.langfuse_dataset_name,
        body_template_path=args.body_template_path,
        batch_ids_path=args.batch_ids, run_id=args.run_id,
    )
