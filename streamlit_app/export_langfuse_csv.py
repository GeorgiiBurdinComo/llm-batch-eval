from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from requests.exceptions import (
    ChunkedEncodingError,
    ConnectionError,
    HTTPError,
    RetryError,
    Timeout,
)
from tqdm.auto import tqdm
from urllib3.exceptions import ProtocolError
from urllib3.util.retry import Retry

# Config lives next to the Streamlit app; keep a fallback so running from inside
# `streamlit_app/` also works (e.g. `python export_langfuse_csv.py`).
try:
    from streamlit_app.config import (
        LANGFUSE_EXPORT_END_BUFFER,
        LANGFUSE_EXPORT_START_DT,
    )
except ModuleNotFoundError:  # pragma: no cover
    from config import (  # type: ignore[no-redef]
        LANGFUSE_EXPORT_END_BUFFER,
        LANGFUSE_EXPORT_START_DT,
    )

# --- SETUP ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

PUBLIC_KEY = os.environ["LANGFUSE_PUBLIC_KEY"]
SECRET_KEY = os.environ["LANGFUSE_SECRET_KEY"]
HOST = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")

# How aggressively to split dense trace windows.
MAX_PAGES_PER_WINDOW = 10
MIN_WINDOW = timedelta(seconds=1)

# Optional extra logging for trace window splits.
VERBOSE_SPLITS = True

# Retries for transient transport errors (e.g. truncated chunked responses,
# urllib3 giving up after repeated 503s from an overloaded Langfuse backend).
_GET_JSON_MAX_ATTEMPTS = 8
_GET_JSON_TIMEOUT_SEC = 120
_GET_JSON_MAX_BACKOFF_SEC = 120
_TRANSIENT_GET_ERRORS = (
    ChunkedEncodingError,
    ConnectionError,
    Timeout,
    ProtocolError,
    RetryError,
)
# HTTP status codes worth retrying when the server returns them directly.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

session = requests.Session()
session.auth = HTTPBasicAuth(PUBLIC_KEY, SECRET_KEY)

retry = Retry(
    total=6,
    connect=6,
    read=6,
    backoff_factor=1.0,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    respect_retry_after_header=True,
)
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)
session.mount("http://", adapter)


@dataclass
class FetchStats:
    windows_probed: int = 0
    windows_split: int = 0
    leaf_windows: int = 0
    pages_fetched: int = 0
    raw_rows: int = 0
    unique_ids: int = 0


def _trace_params(
    start_dt: datetime,
    end_dt: datetime,
    *,
    page: int,
    limit: int = 100,
) -> dict[str, Any]:
    return {
        "fromTimestamp": start_dt.isoformat(timespec="milliseconds"),
        "toTimestamp": end_dt.isoformat(timespec="milliseconds"),
        "limit": limit,
        "page": page,
    }


def _get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(1, _GET_JSON_MAX_ATTEMPTS + 1):
        try:
            resp = session.get(url, params=params, timeout=_GET_JSON_TIMEOUT_SEC)
            resp.raise_for_status()
            return resp.json()
        except HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status not in _RETRYABLE_STATUS or attempt >= _GET_JSON_MAX_ATTEMPTS:
                raise
            last_exc = exc
            wait = min(2 ** attempt, _GET_JSON_MAX_BACKOFF_SEC)
            print(
                f"Retryable Langfuse API status {status}, "
                f"retry {attempt}/{_GET_JSON_MAX_ATTEMPTS} in {wait}s..."
            )
            time.sleep(wait)
        except _TRANSIENT_GET_ERRORS as exc:
            last_exc = exc
            if attempt >= _GET_JSON_MAX_ATTEMPTS:
                raise
            wait = min(2 ** attempt, _GET_JSON_MAX_BACKOFF_SEC)
            print(
                f"Transient Langfuse API error ({type(exc).__name__}), "
                f"retry {attempt}/{_GET_JSON_MAX_ATTEMPTS} in {wait}s..."
            )
            time.sleep(wait)
    raise last_exc  # pragma: no cover


def _probe_window(url: str, start_dt: datetime, end_dt: datetime) -> tuple[int, int]:
    payload = _get_json(url, _trace_params(start_dt, end_dt, page=1))
    meta = payload.get("meta", {})
    return int(meta.get("totalItems", 0)), int(meta.get("totalPages", 0))


def _pick_col(df: pd.DataFrame, *candidates: str) -> str:
    for cand in candidates:
        if cand in df.columns:
            return cand
    raise KeyError(f"None of these columns exist: {candidates}")


def fetch_traces_windowed(
    start_dt: datetime,
    end_dt: datetime,
    max_pages_per_window: int = MAX_PAGES_PER_WINDOW,
    min_window: timedelta = MIN_WINDOW,
) -> list[dict[str, Any]]:
    """
    Fetch traces robustly by recursively splitting the time range until each
    window is small enough to paginate safely.

    Notes:
    - Langfuse defaults to timestamp desc; dedup by trace id handles overlaps.
    - Langfuse meta.totalItems is raw rows, not deduped unique IDs.
    - Progress is tracked by fetched pages, discovered leaf windows, raw rows,
      and deduped unique IDs.
    """
    url = f"{HOST}/api/public/traces"
    items_by_id: dict[str, dict[str, Any]] = {}
    stats = FetchStats()

    root_items, root_pages = _probe_window(url, start_dt, end_dt)
    print(
        f"Traces root window: {root_items:,} raw rows reported across "
        f"{root_pages:,} pages before splitting"
    )

    def _set_bar_postfix(pbar: tqdm) -> None:
        pbar.set_postfix(
            windows=stats.windows_probed,
            splits=stats.windows_split,
            leafs=stats.leaf_windows,
            raw_rows=f"{stats.raw_rows:,}",
            unique_ids=f"{stats.unique_ids:,}",
        )

    with tqdm(
        total=0,
        desc="Traces pages",
        unit="page",
        dynamic_ncols=True,
    ) as pbar:

        def walk(lo: datetime, hi: datetime) -> None:
            if lo >= hi:
                return

            stats.windows_probed += 1
            total_items, total_pages = _probe_window(url, lo, hi)
            _set_bar_postfix(pbar)

            if total_items == 0:
                return

            if total_pages > max_pages_per_window and (hi - lo) > min_window:
                stats.windows_split += 1
                _set_bar_postfix(pbar)

                if VERBOSE_SPLITS:
                    print(
                        f"Splitting dense trace window: "
                        f"{lo.isoformat()} -> {hi.isoformat()} "
                        f"({total_items:,} rows, {total_pages} pages)"
                    )

                mid = lo + (hi - lo) / 2
                walk(lo, mid)   # [lo, mid)
                walk(mid, hi)   # [mid, hi)
                return

            stats.leaf_windows += 1
            pbar.total = (pbar.total or 0) + max(total_pages, 1)
            pbar.refresh()
            _set_bar_postfix(pbar)

            for page in range(1, max(total_pages, 1) + 1):
                payload = _get_json(url, _trace_params(lo, hi, page=page))
                data = payload.get("data", [])

                stats.pages_fetched += 1
                stats.raw_rows += len(data)

                for row in data:
                    items_by_id[row["id"]] = row

                stats.unique_ids = len(items_by_id)

                pbar.update(1)
                _set_bar_postfix(pbar)

        walk(start_dt, end_dt)

    print(
        f"Traces done: {stats.pages_fetched:,} pages fetched, "
        f"{stats.raw_rows:,} raw rows seen, "
        f"{len(items_by_id):,} unique IDs kept"
    )
    return list(items_by_id.values())


def fetch_scores_simple(start_dt: datetime, end_dt: datetime) -> list[dict[str, Any]]:
    """
    Fetch scores with regular page-based pagination.
    Progress is tracked across pages, raw rows, and deduped score IDs.
    """
    url = f"{HOST}/api/public/v2/scores"
    items_by_id: dict[str, dict[str, Any]] = {}

    first_payload = _get_json(
        url,
        {
            "fromTimestamp": start_dt.isoformat(timespec="milliseconds"),
            "toTimestamp": end_dt.isoformat(timespec="milliseconds"),
            "limit": 100,
            "page": 1,
        },
    )

    first_data = first_payload.get("data", [])
    first_meta = first_payload.get("meta", {})
    total_pages = int(first_meta.get("totalPages", 1))
    total_items = int(first_meta.get("totalItems", len(first_data)))

    print(
        f"Scores root window: {total_items:,} raw rows reported across "
        f"{total_pages:,} pages"
    )

    raw_rows = 0

    with tqdm(
        total=max(total_pages, 1),
        desc="Scores pages",
        unit="page",
        dynamic_ncols=True,
    ) as pbar:
        for row in first_data:
            items_by_id[row["id"]] = row
        raw_rows += len(first_data)

        pbar.update(1)
        pbar.set_postfix(raw_rows=f"{raw_rows:,}", unique_ids=f"{len(items_by_id):,}")

        for page in range(2, total_pages + 1):
            payload = _get_json(
                url,
                {
                    "fromTimestamp": start_dt.isoformat(timespec="milliseconds"),
                    "toTimestamp": end_dt.isoformat(timespec="milliseconds"),
                    "limit": 100,
                    "page": page,
                },
            )
            data = payload.get("data", [])

            raw_rows += len(data)
            for row in data:
                items_by_id[row["id"]] = row

            pbar.update(1)
            pbar.set_postfix(raw_rows=f"{raw_rows:,}", unique_ids=f"{len(items_by_id):,}")

    print(
        f"Scores done: {total_pages:,} pages fetched, "
        f"{raw_rows:,} raw rows seen, "
        f"{len(items_by_id):,} unique IDs kept"
    )
    return list(items_by_id.values())


def print_final_counts(df_traces: pd.DataFrame, df_scores: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("FINAL TRACE & SCORE COUNTS")
    print("=" * 70)

    if df_traces.empty:
        print("No traces exported.")
        return

    try:
        t_id = _pick_col(df_traces, "id", "trace_id")
        s_tid = _pick_col(df_scores, "traceId", "trace_id")
        model_col = _pick_col(df_traces, "metadata.model", "model")
    except KeyError as e:
        print(f"Could not build final report: {e}")
        print(f"Trace columns: {list(df_traces.columns)[:50]}")
        print(f"Score columns: {list(df_scores.columns)[:50]}")
        return

    trace_ids = df_traces[t_id].astype(str).str.strip()
    trace_models = df_traces[model_col].fillna("unknown").astype(str)
    mapping = pd.Series(trace_models.values, index=trace_ids).to_dict()

    df_scores = df_scores.copy()
    df_scores[s_tid] = df_scores[s_tid].astype(str).str.strip()
    df_scores["_model"] = df_scores[s_tid].map(mapping)

    tr_counts = (
        df_traces.assign(_trace_id=trace_ids, _model=trace_models)
        .groupby("_model")["_trace_id"]
        .nunique()
        .rename("traces")
    )

    if df_scores.empty or "name" not in df_scores.columns:
        print(tr_counts.to_frame().sort_values("traces", ascending=False).to_string())
        return

    sc_counts = (
        df_scores.groupby(["_model", "name"])[s_tid]
        .nunique()
        .unstack(fill_value=0)
    )

    report = (
        tr_counts.to_frame()
        .join(sc_counts, how="left")
        .fillna(0)
        .sort_values("traces", ascending=False)
    )

    int_cols = [c for c in report.columns]
    report[int_cols] = report[int_cols].astype(int)
    print(report.to_string())


def _select_existing_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing = [c for c in columns if c in df.columns]
    return df.reindex(columns=existing)


def main() -> None:
    end_dt = datetime.now(timezone.utc) - LANGFUSE_EXPORT_END_BUFFER
    start_dt = LANGFUSE_EXPORT_START_DT

    print(f"Trace export: {start_dt.isoformat()} -> {end_dt.isoformat()}")

    traces_raw = fetch_traces_windowed(start_dt, end_dt)
    scores_raw = fetch_scores_simple(start_dt, end_dt)

    traces_full = pd.json_normalize(traces_raw) if traces_raw else pd.DataFrame()
    scores_full = pd.json_normalize(scores_raw) if scores_raw else pd.DataFrame()

    TRACE_EXPORT_COLUMNS = [
        "id",
        "timestamp",
        "name",
        "userId",
        "sessionId",
        "metadata.model",
        "metadata.custom_id",
        "metadata.run_id",
    ]

    SCORE_EXPORT_COLUMNS = [
        "id",
        "traceId",
        "name",
        "timestamp",
        "value",
        "stringValue",
        "comment",
    ]

    df_traces = _select_existing_columns(traces_full, TRACE_EXPORT_COLUMNS)
    df_scores = _select_existing_columns(scores_full, SCORE_EXPORT_COLUMNS)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    traces_csv = os.path.join(out_dir, "langfuse_traces.csv")
    scores_csv = os.path.join(out_dir, "langfuse_scores.csv")

    print("\nWriting CSVs...")
    df_traces.to_csv(traces_csv, index=False)
    df_scores.to_csv(scores_csv, index=False)

    print(f"Unique traces exported: {len(df_traces):,}")
    print(f"Unique scores exported: {len(df_scores):,}")
    print(f"Saved traces CSV: {traces_csv}")
    print(f"Saved scores CSV: {scores_csv}")

    print("Trace columns:", df_traces.columns.tolist())
    print("Score columns:", df_scores.columns.tolist())


if __name__ == "__main__":
    main()