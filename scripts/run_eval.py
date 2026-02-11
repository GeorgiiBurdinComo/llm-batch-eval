"""
Orchestrator: sample subset, submit batches for selected models, write batch_ids for later poll/ingest.
"""

import json
import os
import sys
from dotenv import load_dotenv
load_dotenv()

# Allow running from project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml

# Scripts and lib (helpers not meant as entrypoints)
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_lib_dir = os.path.join(_scripts_dir, "lib")
for _d in (_scripts_dir, _lib_dir):
    if _d not in sys.path:
        sys.path.insert(0, _d)
from sample import sample_subset
import batch_openai
import batch_gemini
from upload_gemini_images import ensure_images_uploaded


def load_models_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.path.join(ROOT, "config", "models.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_submit(
    subset_size: int = 300,
    models: list = None,
    csv_path: str = None,
    langfuse_dataset_name: str = None,
    body_template_path: str = None,
    baseline_path: str = "data/baseline_predictions.json",
    batches_dir: str = "batches",
    batch_ids_path: str = "data/batch_ids.json",
) -> str:
    """
    Sample subset, create batches for each model, write batch_ids to JSON.
    models: list of model ids (e.g. ["gpt-5-mini", "gemini-2.5-flash"]) or None = all from config.
    Data source: csv_path if set, else Langfuse dataset (langfuse_dataset_name or env/default).
    Returns path to batch_ids.json.
    """
    cfg = load_models_config()
    if models is None:
        models = list(cfg.get("openai", [])) + list(cfg.get("gemini", []))
    openai_models = set(cfg.get("openai", []))
    gemini_models = set(cfg.get("gemini", []))

    subset = sample_subset(
        subset_size,
        csv_path=csv_path,
        langfuse_dataset_name=langfuse_dataset_name,
        body_template_path=body_template_path,
        baseline_path=baseline_path if os.path.isfile(baseline_path) else None,
    )
    if len(subset) == 0:
        raise RuntimeError("Empty subset; check dataset source (Langfuse or --csv) and optional baseline")

    os.makedirs(os.path.dirname(batch_ids_path) or ".", exist_ok=True)
    batch_ids = []

    # Ensure Gemini image cache is warm if we will create any Gemini batch (use sampled subset)
    models_to_run = [m for m in models if m in openai_models or m in gemini_models]
    if any(m in gemini_models for m in models_to_run):
        ensure_images_uploaded(examples=subset)

    for model in models:
        if model not in openai_models and model not in gemini_models:
            print(f"[run_eval] Skip unknown model: {model}")
            continue
        try:
            if model in openai_models:
                bid = batch_openai.create_openai_batch(model, subset, output_dir=batches_dir)
                batch_ids.append({"model": model, "provider": "openai", "batch_id": bid})
            else:
                bid = batch_gemini.create_gemini_batch(model, subset, output_dir=batches_dir)
                batch_ids.append({"model": model, "provider": "gemini", "batch_id": bid})
        except Exception as e:
            print(f"[run_eval] Failed {model}: {e}")
            batch_ids.append({"model": model, "provider": "openai" if model in openai_models else "gemini", "batch_id": None, "error": str(e)})

    with open(batch_ids_path, "w") as f:
        json.dump({"subset_size": len(subset), "batches": batch_ids}, f, indent=2)
    print(f"[run_eval] Wrote {batch_ids_path}")
    return batch_ids_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--subset-size", type=int, default=300)
    p.add_argument("--models", type=str, help="Comma-separated model ids; default all")
    p.add_argument("--csv", default=None, help="Use CSV; if omitted, use Langfuse dataset")
    p.add_argument("--langfuse-dataset", default=None, dest="langfuse_dataset_name", help="Langfuse dataset name (default from env)")
    p.add_argument("--body-template", default=None, dest="body_template_path", help="Request body template path (json/csv) for Langfuse source")
    p.add_argument("--baseline", default="data/baseline_predictions.json")
    p.add_argument("--batch-ids", default="data/batch_ids.json")
    args = p.parse_args()
    models = [m.strip() for m in args.models.split(",")] if args.models else None
    run_submit(
        subset_size=args.subset_size,
        models=models,
        csv_path=args.csv,
        langfuse_dataset_name=args.langfuse_dataset_name,
        body_template_path=args.body_template_path,
        baseline_path=args.baseline,
        batch_ids_path=args.batch_ids,
    )
