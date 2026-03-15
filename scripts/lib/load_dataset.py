"""Load dataset rows from Langfuse (default) or local CSV.

Returns: list of {custom_id, body, campaign_relevant} for sample/batch/ingest.
"""

import ast
import csv
import json
import os
import sys
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_LANGFUSE_DATASET = "campaign_relevance_disagree_subset_9d488308aa46"
DEFAULT_TEMPLATE_JSON = os.path.join(ROOT, "config", "request_template.json")
DEFAULT_TEMPLATE_CSV = os.path.join(ROOT, "input", "dataset.csv")


def _default_body_template() -> Dict:
    return {
        "store": True,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "CampaignRelevanceMinimal",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"campaign_relevant": {"type": "boolean"}},
                    "required": ["campaign_relevant"],
                    "additionalProperties": False,
                },
            }
        },
        "metadata": {"endpoint": "campaign-relevance"},
    }


def _body_template_from_csv(csv_path: str) -> Dict:
    if not os.path.isfile(csv_path):
        print(f"[load_dataset] Template CSV not found ({csv_path}); using fallback")
        return _default_body_template()
    with open(csv_path, "r", encoding="utf-8") as f:
        row = next(csv.DictReader(f), None)
    if not row:
        print(f"[load_dataset] Template CSV empty ({csv_path}); using fallback")
        return _default_body_template()
    try:
        return dict(ast.literal_eval(row["body"]))
    except (ValueError, SyntaxError, KeyError) as e:
        raise ValueError(f"Invalid body in template CSV: {e}") from e


def _body_template_from_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        body = json.load(f)
    if not isinstance(body, dict):
        raise ValueError(f"Template JSON must be an object: {path}")
    return dict(body)


def _load_body_template(path: Optional[str]) -> Dict:
    """Resolve template: explicit path > config/request_template.json > CSV first row > fallback."""
    if path:
        abs_path = os.path.join(ROOT, path) if not os.path.isabs(path) else path
        if os.path.splitext(abs_path)[1].lower() == ".json":
            if not os.path.isfile(abs_path):
                raise FileNotFoundError(f"Template JSON not found: {abs_path}")
            return _body_template_from_json(abs_path)
        return _body_template_from_csv(abs_path)

    if os.path.isfile(DEFAULT_TEMPLATE_JSON):
        return _body_template_from_json(DEFAULT_TEMPLATE_JSON)
    return _body_template_from_csv(DEFAULT_TEMPLATE_CSV)


def load_csv_rows(csv_path: str) -> List[Dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            try:
                body = ast.literal_eval(row["body"])
            except (ValueError, SyntaxError):
                continue
            rows.append({
                "custom_id": row["custom_id"],
                "body": body,
                "campaign_relevant": str(row.get("campaign_relevant", "")).strip().lower() == "true",
                "row_index": idx,
            })
    return rows


def load_dataset_rows(
    csv_path: Optional[str] = None,
    langfuse_dataset_name: Optional[str] = None,
    body_template_path: Optional[str] = None,
) -> List[Dict]:
    """Load from CSV (if csv_path set) or Langfuse dataset."""
    if csv_path:
        return load_csv_rows(os.path.join(ROOT, csv_path) if not os.path.isabs(csv_path) else csv_path)

    from langfuse import Langfuse

    name = langfuse_dataset_name or os.getenv("LANGFUSE_DATASET_NAME") or DEFAULT_LANGFUSE_DATASET
    template = _load_body_template(body_template_path)
    dataset = Langfuse().get_dataset(name)

    rows = []
    for idx, item in enumerate(dataset.items):
        cid = (item.metadata or {}).get("custom_id") or item.id
        relevant = bool((item.expected_output or {}).get("campaign_relevant", False)) if isinstance(item.expected_output, dict) else False
        rows.append({
            "custom_id": str(cid),
            "body": {**template, "input": item.input},
            "campaign_relevant": relevant,
            "row_index": idx,
        })
    return rows
