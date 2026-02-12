"""
Ingest batch results JSONL into Langfuse as traces with accuracy scores and costs.

Uses Langfuse SDK v3 (OpenTelemetry-based): root span as trace, generation as child, create_score for accuracy.
"""

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


def _load_ground_truth_csv(csv_path: str) -> Dict[str, bool]:
    """Load custom_id → ground_truth boolean from the dataset CSV."""
    truth: Dict[str, bool] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("custom_id")
            if not cid:
                continue
            val = str(row.get("campaign_relevant", "")).strip().lower()
            truth[cid] = val == "true"
    return truth


def _load_ground_truth_langfuse(dataset_name: str) -> Dict[str, bool]:
    """Load custom_id → campaign_relevant from Langfuse dataset expected_output."""
    truth, _ = _load_from_langfuse(dataset_name)
    return truth


def _load_from_langfuse(dataset_name: str) -> Tuple[Dict[str, bool], Dict[str, Any]]:
    """Load custom_id → campaign_relevant and custom_id → input from Langfuse dataset. Single API call."""
    lf = get_client()
    dataset = lf.get_dataset(dataset_name)
    truth: Dict[str, bool] = {}
    inputs: Dict[str, Any] = {}
    for item in dataset.items:
        cid = (item.metadata or {}).get("custom_id") or item.id
        if not cid:
            continue
        cid_str = str(cid)
        val = False
        if item.expected_output and isinstance(item.expected_output, dict):
            val = bool(item.expected_output.get("campaign_relevant", False))
        truth[cid_str] = val
        if item.input is not None:
            inputs[cid_str] = item.input
    return truth, inputs


def get_ground_truth(
    ground_truth_csv: Optional[str] = "input/dataset.csv",
    langfuse_dataset_name: Optional[str] = None,
) -> Dict[str, bool]:
    """Load custom_id → campaign_relevant. Use Langfuse if langfuse_dataset_name is not None, else CSV."""
    if langfuse_dataset_name is not None:
        name = langfuse_dataset_name or os.getenv("LANGFUSE_DATASET_NAME") or DEFAULT_LANGFUSE_DATASET
        return _load_ground_truth_langfuse(name)
    return _load_ground_truth_csv(ground_truth_csv or "input/dataset.csv")


def _get_effective_response(response: Dict, provider: str) -> Dict:
    """Response body for extraction/usage: OpenAI batch puts payload in response.body."""
    if provider == "openai":
        return response.get("body") or response
    return response


def _normalize_usage(effective_response: Dict, provider: str) -> Tuple[int, int, int]:
    """Return (prompt_tokens, completion_tokens, total_tokens) for usage_details."""
    if provider == "openai":
        usage = effective_response.get("usage") or {}
        inp = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        out = usage.get("output_tokens") or usage.get("completion_tokens") or 0
        total = usage.get("total_tokens") or (inp + out)
        return inp, out, total
    # Gemini
    meta = effective_response.get("usageMetadata") or effective_response.get("usage_metadata") or {}
    inp = meta.get("promptTokenCount") or 0
    out = meta.get("candidatesTokenCount") or 0
    total = meta.get("totalTokenCount") or (inp + out)
    return inp, out, total


def _load_pricing() -> Dict[str, Dict[str, float]]:
    """Load config/models.yaml pricing section: model -> {input: $/1M, output: $/1M}."""
    path = os.path.join(ROOT, "config", "models.yaml")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("pricing") or {}


def _compute_cost_breakdown(
    model: str,
    input_tokens: int,
    output_tokens: int,
    pricing: Dict[str, Dict[str, float]],
) -> Optional[Dict[str, float]]:
    """Cost breakdown in USD from pricing per 1M tokens. Returns None if model not in pricing."""
    p = pricing.get(model)
    if not p and model:
        # e.g. gpt-5-mini-2025-08-07 -> gpt-5-mini
        base = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)
        p = pricing.get(base)
    if not p:
        return None
    inp_price = float(p.get("input") or 0)
    out_price = float(p.get("output") or 0)
    input_cost = (input_tokens * inp_price) / 1e6
    output_cost = (output_tokens * out_price) / 1e6
    total_cost = input_cost + output_cost
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def ingest_results(
    jsonl_path: str,
    model: str,
    provider: str,
    ground_truth_csv: Optional[str] = "input/dataset.csv",
    langfuse_dataset_name: Optional[str] = None,
) -> None:
    """
    Ingest a single batch's results into Langfuse (SDK v3).

    Creates a root span (trace) per result, a generation child, and an accuracy score on the trace.
    """
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"Results file not found: {jsonl_path}")

    lf = get_client()
    if langfuse_dataset_name is not None:
        name = langfuse_dataset_name or os.getenv("LANGFUSE_DATASET_NAME") or DEFAULT_LANGFUSE_DATASET
        ground_truth, dataset_inputs = _load_from_langfuse(name)
    else:
        ground_truth = get_ground_truth(ground_truth_csv=ground_truth_csv, langfuse_dataset_name=None)
        dataset_inputs = {}

    pricing = _load_pricing()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    print(f"[ingest] Ingesting {len(lines)} results for {provider}/{model} from {jsonl_path}")

    for item in lines:
        custom_id = item.get("custom_id") or item.get("key")
        cid_str = str(custom_id) if custom_id is not None else ""
        response = item.get("response", {})
        body = item.get("body", {})
        effective = _get_effective_response(response, provider)

        # created_at: OpenAI batch has body.completed_at or body.created_at
        if provider == "openai":
            ts = effective.get("completed_at") or effective.get("created_at") or response.get("created")
            created_at = datetime.fromtimestamp(ts) if ts is not None else datetime.utcnow()
        else:
            created_at = datetime.utcnow()

        predicted = _extract_prediction(effective, provider)
        expected = ground_truth.get(cid_str)
        correct = expected is not None and (predicted == expected)

        prompt_tokens, completion_tokens, total_tokens = _normalize_usage(effective, provider)
        cost_breakdown = _compute_cost_breakdown(model, prompt_tokens, completion_tokens, pricing)

        trace_input = dataset_inputs.get(cid_str) or body.get("input")
        trace_meta = {
            "model": model,
            "provider": provider,
            "custom_id": custom_id,
            "batch_eval": True,
        }

        usage_details = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        gen_kw: Dict[str, Any] = {
            "model": model,
            "input": trace_input,
            "output": response,
            "usage_details": usage_details,
            "completion_start_time": created_at,
        }
        if cost_breakdown is not None:
            gen_kw["cost_details"] = cost_breakdown

        with lf.start_as_current_observation(
            as_type="span",
            name=f"batch_eval_{model}",
            input=trace_input,
            output={"campaign_relevant": predicted},
            metadata=trace_meta,
        ) as root_span:
            root_span.update_trace(
                input=trace_input,
                output={"campaign_relevant": predicted},
                metadata=trace_meta,
                tags=["batch_evaluation", f"model:{model}"],
            )

            with lf.start_as_current_observation(
                as_type="generation",
                name=f"{model}_generation",
                **gen_kw,
            ):
                pass

            if expected is not None:
                root_span.score_trace(
                    name="accuracy",
                    value=1.0 if correct else 0.0,
                    data_type="NUMERIC",
                    comment=f"expected={expected}, predicted={predicted}",
                )

            # Make cost easy to chart in Langfuse Scores (in addition to generation.cost_details).
            if cost_breakdown is not None:
                root_span.score_trace(
                    name="cost_usd",
                    value=cost_breakdown["total_cost"],
                    data_type="NUMERIC",
                    comment=(
                        f"input_cost={cost_breakdown['input_cost']:.8f}, "
                        f"output_cost={cost_breakdown['output_cost']:.8f}"
                    ),
                )

    lf.flush()
    print(f"[ingest] Done for {provider}/{model}")


def _extract_openai_text(body: Dict) -> Optional[str]:
    """Get assistant text from OpenAI Responses API body (output array or legacy .text)."""
    # Prefer output[] (actual response); body.text is often the request schema in batch responses
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


def _extract_prediction(response: Dict, provider: str) -> bool:
    """Extract boolean campaign_relevant from provider response."""
    if provider == "openai":
        text = _extract_openai_text(response)
        if not text:
            return False
        try:
            return bool(json.loads(text).get("campaign_relevant", False))
        except Exception:
            return False

    try:
        candidates = response.get("candidates") or []
        if not candidates:
            return False
        content = candidates[0].get("content", {})
        parts = content.get("parts") or []
        if not parts:
            return False
        text = parts[0].get("text", "")
        if isinstance(text, str):
            parsed = json.loads(text)
        else:
            parsed = text
        return bool(parsed.get("campaign_relevant", False))
    except Exception:
        return False


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Ingest batch results JSONL into Langfuse")
    p.add_argument("jsonl_path")
    p.add_argument("--model", required=True)
    p.add_argument("--provider", required=True, choices=["openai", "gemini"])
    p.add_argument("--csv", default=None, help="Ground truth CSV; if omitted and no --langfuse-dataset, use input/dataset.csv")
    p.add_argument("--langfuse-dataset", default=None, dest="langfuse_dataset_name", help="Ground truth from Langfuse dataset (default from env)")
    args = p.parse_args()

    ingest_results(
        args.jsonl_path,
        args.model,
        args.provider,
        ground_truth_csv=args.csv,
        langfuse_dataset_name=args.langfuse_dataset_name,
    )
