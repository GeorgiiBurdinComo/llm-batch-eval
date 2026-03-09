"""Ingest batch results JSONL into Langfuse as traces with accuracy scores and costs."""

import csv
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

import yaml
from langfuse import get_client

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LANGFUSE_DATASET = "campaign_relevance_02e1a68ccb0f"


# ── Ground truth loading ─────────────────────────────────────────────────────

def _load_ground_truth_csv(csv_path: str) -> Dict[str, bool]:
    truth: Dict[str, bool] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row.get("custom_id")
            if cid:
                truth[cid] = str(row.get("campaign_relevant", "")).strip().lower() == "true"
    return truth


def _load_from_langfuse(dataset_name: str) -> Tuple[Dict[str, bool], Dict[str, Any]]:
    """Load ground truth and inputs from a Langfuse dataset in one API call."""
    dataset = get_client().get_dataset(dataset_name)
    truth: Dict[str, bool] = {}
    inputs: Dict[str, Any] = {}
    for item in dataset.items:
        cid = str((item.metadata or {}).get("custom_id") or item.id)
        if not cid:
            continue
        if item.expected_output and isinstance(item.expected_output, dict):
            truth[cid] = bool(item.expected_output.get("campaign_relevant", False))
        else:
            truth[cid] = False
        if item.input is not None:
            inputs[cid] = item.input
    return truth, inputs


def get_ground_truth(
    ground_truth_csv: Optional[str] = "input/dataset.csv",
    langfuse_dataset_name: Optional[str] = None,
) -> Dict[str, bool]:
    if langfuse_dataset_name is not None:
        name = langfuse_dataset_name or os.getenv("LANGFUSE_DATASET_NAME") or DEFAULT_LANGFUSE_DATASET
        truth, _ = _load_from_langfuse(name)
        return truth
    csv_path = ground_truth_csv or "input/dataset.csv"
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(ROOT, csv_path)
    return _load_ground_truth_csv(csv_path)


# ── Response parsing ─────────────────────────────────────────────────────────

def _get_effective_response(response: Dict, provider: str) -> Dict:
    if provider == "openai":
        return response.get("body") or response
    return response


def _normalize_usage(effective: Dict, provider: str) -> Tuple[int, int, int]:
    """Return (input_tokens, output_tokens, total_tokens). Includes reasoning/thinking tokens."""
    if provider == "openai":
        u = effective.get("usage") or {}
        inp = u.get("input_tokens") or u.get("prompt_tokens") or 0
        out = u.get("output_tokens") or u.get("completion_tokens") or 0
        return inp, out, u.get("total_tokens") or (inp + out)

    if provider == "claude":
        u = effective.get("usage") or {}
        inp = u.get("input_tokens") or 0
        out = u.get("output_tokens") or 0
        return inp, out, u.get("total_tokens") or (inp + out)

    # Gemini — thinking tokens are tracked separately, billed as output
    m = effective.get("usageMetadata") or effective.get("usage_metadata") or {}
    inp = m.get("promptTokenCount") or 0
    thoughts = m.get("thoughtsTokenCount") or m.get("thoughts_token_count") or 0
    out = (m.get("candidatesTokenCount") or 0) + thoughts
    return inp, out, m.get("totalTokenCount") or (inp + out)


def _extract_prediction(response: Dict, provider: str) -> bool:
    """Extract boolean campaign_relevant from the provider-specific response."""
    try:
        if provider == "openai":
            return bool(json.loads(_extract_openai_text(response) or "{}").get("campaign_relevant", False))

        if provider == "claude":
            for block in response.get("content") or []:
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    return bool(json.loads(block["text"]).get("campaign_relevant", False))
            return False

        # Gemini
        parts = ((response.get("candidates") or [{}])[0].get("content", {}).get("parts") or [])
        text = parts[0].get("text", "") if parts else ""
        parsed = json.loads(text) if isinstance(text, str) else text
        return bool(parsed.get("campaign_relevant", False))
    except Exception:
        return False


def _extract_openai_text(body: Dict) -> Optional[str]:
    for out in body.get("output") or []:
        if out.get("type") != "message":
            continue
        for block in out.get("content") or []:
            if block.get("type") == "output_text" and "text" in block:
                return block["text"]
    text_obj = body.get("text")
    if isinstance(text_obj, str):
        return text_obj
    if isinstance(text_obj, dict):
        return json.dumps(text_obj) if text_obj else None
    return None


# ── Pricing ──────────────────────────────────────────────────────────────────

def _load_pricing() -> Dict[str, Dict[str, float]]:
    path = os.path.join(ROOT, "config", "models.yaml")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("pricing") or {}


def _compute_cost(model: str, input_tokens: int, output_tokens: int, pricing: Dict) -> Optional[Dict[str, float]]:
    """Cost breakdown in USD. Returns None if model not in pricing."""
    p = pricing.get(model)
    if not p and model:
        p = pricing.get(re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model))
    if not p:
        return None
    input_cost = (input_tokens * float(p.get("input") or 0)) / 1e6
    output_cost = (output_tokens * float(p.get("output") or 0)) / 1e6
    return {"input_cost": input_cost, "output_cost": output_cost, "total_cost": input_cost + output_cost}


# ── Main ingestion ───────────────────────────────────────────────────────────

def ingest_results(
    jsonl_path: str,
    model: str,
    provider: str,
    ground_truth_csv: Optional[str] = "input/dataset.csv",
    langfuse_dataset_name: Optional[str] = None,
    run_id: Optional[str] = None,
) -> None:
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"Results file not found: {jsonl_path}")

    lf = get_client()

    if langfuse_dataset_name is not None:
        name = langfuse_dataset_name or os.getenv("LANGFUSE_DATASET_NAME") or DEFAULT_LANGFUSE_DATASET
        ground_truth, dataset_inputs = _load_from_langfuse(name)
    else:
        ground_truth = get_ground_truth(ground_truth_csv=ground_truth_csv)
        dataset_inputs = {}

    pricing = _load_pricing()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    print(f"[ingest] Ingesting {len(lines)} results for {provider}/{model}")

    for item in lines:
        custom_id = item.get("custom_id") or item.get("key")
        cid = str(custom_id) if custom_id is not None else ""
        response = item.get("response", {})
        effective = _get_effective_response(response, provider)

        if provider == "openai":
            ts = effective.get("completed_at") or effective.get("created_at") or response.get("created")
            created_at = datetime.fromtimestamp(ts) if ts is not None else datetime.utcnow()
        else:
            created_at = datetime.utcnow()

        predicted = _extract_prediction(effective, provider)
        expected = ground_truth.get(cid)
        correct = expected is not None and (predicted == expected)

        inp_tok, out_tok, total_tok = _normalize_usage(effective, provider)
        cost = _compute_cost(model, inp_tok, out_tok, pricing)

        trace_input = dataset_inputs.get(cid) or item.get("body", {}).get("input")
        trace_meta = {"model": model, "provider": provider, "custom_id": custom_id, "batch_eval": True}
        if run_id:
            trace_meta["run_id"] = run_id

        gen_kw: Dict[str, Any] = {
            "model": model, "input": trace_input, "output": response,
            "usage_details": {"prompt_tokens": inp_tok, "completion_tokens": out_tok, "total_tokens": total_tok},
            "completion_start_time": created_at,
        }
        if cost:
            gen_kw["cost_details"] = cost

        tags = ["batch_evaluation", f"model:{model}"] + ([f"run:{run_id}"] if run_id else [])

        with lf.start_as_current_observation(
            as_type="span", name=f"batch_eval_{model}",
            input=trace_input, output={"campaign_relevant": predicted}, metadata=trace_meta,
        ) as span:
            span.update_trace(
                input=trace_input, output={"campaign_relevant": predicted},
                metadata=trace_meta, tags=tags,
            )
            with lf.start_as_current_observation(as_type="generation", name=f"{model}_generation", **gen_kw):
                pass

            if expected is not None:
                span.score_trace(
                    name="accuracy",
                    value=1.0 if correct else 0.0,
                    data_type="NUMERIC",
                    comment=f"expected={expected}, predicted={predicted}",
                )

                # Confusion matrix label: TP / FP / TN / FN
                if expected is True and predicted is True:
                    confusion_label = "true_positive"
                elif expected is True and predicted is False:
                    confusion_label = "false_negative"
                elif expected is False and predicted is False:
                    confusion_label = "true_negative"
                else:
                    confusion_label = "false_positive"

                span.score_trace(
                    name="error_type",
                    value=confusion_label,
                    data_type="CATEGORICAL",
                    comment=f"expected={expected}, predicted={predicted}",
                )

            if cost:
                span.score_trace(
                    name="cost_usd",
                    value=cost["total_cost"],
                    data_type="NUMERIC",
                    comment=f"input_cost={cost['input_cost']:.8f}, output_cost={cost['output_cost']:.8f}",
                )

    lf.flush()
    print(f"[ingest] Done for {provider}/{model}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Ingest batch results JSONL into Langfuse")
    p.add_argument("jsonl_path")
    p.add_argument("--model", required=True)
    p.add_argument("--provider", required=True, choices=["openai", "gemini", "claude"])
    p.add_argument("--csv", default=None)
    p.add_argument("--langfuse-dataset", default=None, dest="langfuse_dataset_name")
    args = p.parse_args()

    ingest_results(args.jsonl_path, args.model, args.provider,
                   ground_truth_csv=args.csv, langfuse_dataset_name=args.langfuse_dataset_name)
