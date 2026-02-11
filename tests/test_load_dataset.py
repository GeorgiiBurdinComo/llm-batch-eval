"""
Tests for load_dataset: load_dataset_rows from CSV and (mocked) Langfuse.
Run: python -m pytest tests/test_load_dataset.py -v
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from load_dataset import (
    DEFAULT_LANGFUSE_DATASET,
    _body_template_from_csv,
    load_csv_rows,
    load_dataset_rows,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fixture_csv(tmp_path):
    """Minimal CSV with 2 rows: custom_id, body (repr), campaign_relevant. Body is quoted so commas inside don't split columns."""
    body1 = "{'input': [{'role': 'user', 'content': 'hello'}], 'text': {'format': 'json'}, 'model': 'gpt-4', 'store': True}"
    body2 = "{'input': [{'role': 'user', 'content': 'world'}], 'text': {'format': 'json'}, 'model': 'gpt-4', 'store': True}"
    path = tmp_path / "dataset.csv"
    # Quote body so CSV reader keeps it as one field
    path.write_text(
        "custom_id,body,campaign_relevant\n"
        f'id1,"{body1}",true\n'
        f'id2,"{body2}",false\n',
        encoding="utf-8",
    )
    return str(path)


# ---------------------------------------------------------------------------
# load_csv_rows / load_dataset_rows from CSV
# ---------------------------------------------------------------------------

def test_load_csv_rows_shape_and_content(fixture_csv):
    rows = load_csv_rows(fixture_csv)
    assert len(rows) == 2
    for r in rows:
        assert "custom_id" in r
        assert "body" in r
        assert "campaign_relevant" in r
        assert "row_index" in r
        assert isinstance(r["body"], dict)
        assert "input" in r["body"]
    assert rows[0]["custom_id"] == "id1"
    assert rows[0]["campaign_relevant"] is True
    assert rows[1]["custom_id"] == "id2"
    assert rows[1]["campaign_relevant"] is False


def test_load_dataset_rows_with_csv_path(fixture_csv):
    rows = load_dataset_rows(csv_path=fixture_csv)
    assert len(rows) == 2
    assert rows[0]["custom_id"] == "id1"
    assert rows[0]["body"]["input"][0]["content"] == "hello"
    assert rows[0]["campaign_relevant"] is True
    assert rows[1]["campaign_relevant"] is False


def test_load_dataset_rows_csv_takes_precedence(fixture_csv):
    """When csv_path is set, langfuse_dataset_name is ignored."""
    rows = load_dataset_rows(csv_path=fixture_csv, langfuse_dataset_name="other_dataset")
    assert len(rows) == 2
    assert rows[0]["custom_id"] == "id1"


# ---------------------------------------------------------------------------
# _body_template_from_csv
# ---------------------------------------------------------------------------

def test_body_template_from_csv(fixture_csv):
    template = _body_template_from_csv(fixture_csv)
    assert isinstance(template, dict)
    assert "input" in template
    assert "text" in template
    assert template.get("model") == "gpt-4"
    assert template["input"][0]["content"] == "hello"


def test_body_template_from_csv_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError) as exc_info:
        _body_template_from_csv(str(tmp_path / "nonexistent.csv"))
    assert "Body template CSV not found" in str(exc_info.value)


def test_body_template_from_csv_empty_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("custom_id,body,campaign_relevant\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        _body_template_from_csv(str(path))
    assert "empty" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# load_dataset_rows from Langfuse (mocked)
# ---------------------------------------------------------------------------

def test_load_dataset_rows_from_langfuse_mock(fixture_csv):
    """With Langfuse mocked, rows have body = template + item.input and campaign_relevant from expected_output."""
    item1 = MagicMock()
    item1.id = "lf-id-1"
    item1.metadata = {"custom_id": "cid1"}
    item1.expected_output = {"campaign_relevant": True}
    item1.input = [{"role": "user", "content": "from_langfuse_1"}]

    item2 = MagicMock()
    item2.id = "lf-id-2"
    item2.metadata = {"custom_id": "cid2"}
    item2.expected_output = {"campaign_relevant": False}
    item2.input = [{"role": "user", "content": "from_langfuse_2"}]

    mock_dataset = MagicMock()
    mock_dataset.items = [item1, item2]

    mock_lf = MagicMock()
    mock_lf.get_dataset.return_value = mock_dataset

    # Langfuse is imported inside _load_rows_from_langfuse; patch at source
    with patch("langfuse.get_client", return_value=mock_lf):
        rows = load_dataset_rows(
            langfuse_dataset_name="test_dataset",
            body_template_path=fixture_csv,
        )

    assert len(rows) == 2
    assert rows[0]["custom_id"] == "cid1"
    assert rows[0]["campaign_relevant"] is True
    assert rows[0]["body"]["input"] == [{"role": "user", "content": "from_langfuse_1"}]
    assert rows[0]["body"].get("text") == {"format": "json"}
    assert rows[0]["body"].get("model") == "gpt-4"

    assert rows[1]["custom_id"] == "cid2"
    assert rows[1]["campaign_relevant"] is False
    assert rows[1]["body"]["input"] == [{"role": "user", "content": "from_langfuse_2"}]


def test_load_dataset_rows_langfuse_fallback_custom_id_to_id(fixture_csv):
    """When metadata has no custom_id, use item.id."""
    item = MagicMock()
    item.id = "fallback-id"
    item.metadata = None
    item.expected_output = {"campaign_relevant": True}
    item.input = []

    mock_dataset = MagicMock()
    mock_dataset.items = [item]
    mock_lf = MagicMock()
    mock_lf.get_dataset.return_value = mock_dataset

    with patch("langfuse.get_client", return_value=mock_lf):
        rows = load_dataset_rows(
            langfuse_dataset_name="test",
            body_template_path=fixture_csv,
        )
    assert len(rows) == 1
    assert rows[0]["custom_id"] == "fallback-id"


def test_default_langfuse_dataset_constant():
    assert DEFAULT_LANGFUSE_DATASET == "campaign_relevance_02e1a68ccb0f"
