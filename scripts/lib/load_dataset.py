"""
Load dataset rows from Langfuse (default) or from local CSV.
Returns same shape: list of {custom_id, body, campaign_relevant} for sample/batch/ingest.
"""

import ast
import csv
import json
import os
import sys
from typing import Dict, List, Optional

# Project root (scripts/lib -> scripts -> project root)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_LANGFUSE_DATASET = "campaign_relevance_02e1a68ccb0f"
DEFAULT_TEMPLATE_JSON = os.path.join(ROOT, "config", "request_template.json")
DEFAULT_TEMPLATE_CSV = os.path.join(ROOT, "input", "dataset.csv")


def _default_body_template() -> Dict:
    """
    Fallback template when no CSV is available.
    Keeps structured output so downstream parsers can reliably read campaign_relevant.
    """
    return {
        "store": True,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "CampaignRelevanceMinimal",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "campaign_relevant": {"type": "boolean"},
                    },
                    "required": ["campaign_relevant"],
                    "additionalProperties": False,
                },
            }
        },
        "metadata": {"endpoint": "campaign-relevance"},
    }


def _body_template_from_csv(csv_path: str) -> Dict:
    """Load first row's body from CSV as template (text, store, metadata, model; no input)."""
    if not os.path.isfile(csv_path):
        print(f"[load_dataset] Template CSV not found ({csv_path}); using built-in fallback template")
        return _default_body_template()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
    if not row:
        print(f"[load_dataset] Template CSV is empty ({csv_path}); using built-in fallback template")
        return _default_body_template()
    try:
        body = ast.literal_eval(row["body"])
    except (ValueError, SyntaxError, KeyError) as e:
        raise ValueError(f"Invalid body in template CSV first row: {e}") from e
    # Template = full body; we will overwrite "input" per item
    return dict(body)


def _body_template_from_json(json_path: str) -> Dict:
    """Load request body template from JSON config file."""
    with open(json_path, "r", encoding="utf-8") as f:
        body = json.load(f)
    if not isinstance(body, dict):
        raise ValueError(f"Template JSON must contain an object at root: {json_path}")
    return dict(body)


def _load_body_template(body_template_path: Optional[str]) -> Dict:
    """
    Resolve and load body template.
    Priority:
      1) explicit body_template_path (JSON or CSV),
      2) config/request_template.json,
      3) input/dataset.csv first row body,
      4) built-in fallback.
    """
    if body_template_path:
        path = os.path.join(ROOT, body_template_path) if not os.path.isabs(body_template_path) else body_template_path
        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Template JSON not found: {path}")
            return _body_template_from_json(path)
        return _body_template_from_csv(path)

    if os.path.isfile(DEFAULT_TEMPLATE_JSON):
        return _body_template_from_json(DEFAULT_TEMPLATE_JSON)
    return _body_template_from_csv(DEFAULT_TEMPLATE_CSV)


def _load_rows_from_langfuse(
    dataset_name: str,
    body_template_path: Optional[str],
) -> List[Dict]:
    """Fetch dataset from Langfuse and build rows using body template."""
    from langfuse import get_client

    template = _load_body_template(body_template_path)
    lf = get_client()
    dataset = lf.get_dataset(dataset_name)
    rows = []
    for idx, item in enumerate(dataset.items):
        custom_id = (item.metadata or {}).get("custom_id") or item.id
        campaign_relevant = False
        if item.expected_output and isinstance(item.expected_output, dict):
            campaign_relevant = bool(item.expected_output.get("campaign_relevant", False))
        body = {**template, "input": item.input}
        rows.append({
            "custom_id": str(custom_id),
            "body": body,
            "campaign_relevant": campaign_relevant,
            "row_index": idx,
        })
    return rows


def load_csv_rows(csv_path: str) -> List[Dict]:
    """Load dataset CSV into list of dicts with custom_id, body (parsed), campaign_relevant."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            try:
                body = ast.literal_eval(row["body"])
            except (ValueError, SyntaxError):
                continue
            relevant = str(row.get("campaign_relevant", "")).strip().lower() == "true"
            rows.append({
                "custom_id": row["custom_id"],
                "body": body,
                "campaign_relevant": relevant,
                "row_index": idx,
            })
    return rows


def load_dataset_rows(
    csv_path: Optional[str] = None,
    langfuse_dataset_name: Optional[str] = None,
    body_template_path: Optional[str] = None,
) -> List[Dict]:
    """
    Load dataset rows from Langfuse (default) or from local CSV.
    Returns list of {custom_id, body, campaign_relevant} (and row_index when from Langfuse).

    - If csv_path is set: load from CSV (ignores langfuse_dataset_name).
    - Else: load from Langfuse dataset. Dataset name = langfuse_dataset_name or
      env LANGFUSE_DATASET_NAME or DEFAULT_LANGFUSE_DATASET.
      Body template is taken from:
      explicit --body-template path (json/csv) -> config/request_template.json -> input/dataset.csv -> built-in fallback.
    """
    if csv_path:
        csv_abs = os.path.join(ROOT, csv_path) if not os.path.isabs(csv_path) else csv_path
        return load_csv_rows(csv_abs)

    name = (
        langfuse_dataset_name
        or os.getenv("LANGFUSE_DATASET_NAME")
        or DEFAULT_LANGFUSE_DATASET
    )
    return _load_rows_from_langfuse(name, body_template_path)
