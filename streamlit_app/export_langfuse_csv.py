import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from urllib3.util.retry import Retry


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))


def _secret(key: str, default: str = "") -> str:
    val = os.environ.get(key)
    if val is not None:
        return val.strip().strip('"').strip("'")
    return default


LANGFUSE_PUBLIC_KEY = _secret("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = _secret("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = _secret("LANGFUSE_HOST") or "https://cloud.langfuse.com"

TRACES_URL = f"{LANGFUSE_HOST}/api/public/traces"
SCORES_V2_URL = f"{LANGFUSE_HOST}/api/public/v2/scores"


def _build_session() -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


SESSION = _build_session()


def _fetch_paginated(
    url: str,
    auth: HTTPBasicAuth,
    base_params: dict,
    *,
    page_limit: int = 100,
    timeout: tuple[int, int] = (10, 120),  # (connect timeout, read timeout)
    sleep_s: float = 0.0,
) -> list[dict]:
    rows: list[dict] = []
    page = 1

    while True:
        params = {**base_params, "limit": page_limit, "page": page}
        print(f"Fetching {url} page={page} limit={page_limit} ...")

        try:
            res = SESSION.get(url, auth=auth, params=params, timeout=timeout)
        except requests.exceptions.ReadTimeout as e:
            raise RuntimeError(
                f"Read timeout on {url} page={page} limit={page_limit}. "
                f"Try lowering page_limit further or increasing read timeout."
            ) from e

        if res.status_code != 200:
            raise RuntimeError(f"Request failed ({url}): {res.status_code} {res.text}")

        payload = res.json()
        data = payload.get("data") or []
        meta = payload.get("meta") or {}

        print(
            f"  got {len(data)} rows "
            f"(page {meta.get('page', page)} / {meta.get('totalPages', '?')})"
        )

        if not data:
            break

        rows.extend(data)

        cur_page = meta.get("page", page)
        total_pages = meta.get("totalPages", page)

        if cur_page >= total_pages:
            break

        page += 1

        if sleep_s:
            time.sleep(sleep_s)

    return rows


def main() -> None:
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        raise RuntimeError("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set.")

    auth = HTTPBasicAuth(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY)

    start = datetime(2026, 3, 8, tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)

    from_ts = start.isoformat()
    to_ts = end.isoformat()

    print(f"Downloading traces from {from_ts} to {to_ts}...")
    traces_raw = _fetch_paginated(
        TRACES_URL,
        auth,
        {
            "fromTimestamp": from_ts,
            "toTimestamp": to_ts,
        },
        page_limit=100,
        timeout=(10, 120),
        sleep_s=0.1,
    )
    print(f"Fetched {len(traces_raw)} traces.")

    print(f"Downloading scores from {from_ts} to {to_ts}...")
    scores_raw = _fetch_paginated(
        SCORES_V2_URL,
        auth,
        {
            "fromTimestamp": from_ts,
            "toTimestamp": to_ts,
        },
        page_limit=50,      # key change
        timeout=(10, 180),  # key change
        sleep_s=0.1,
    )
    print(f"Fetched {len(scores_raw)} scores.")

    out_dir = os.path.dirname(__file__)
    traces_path = os.path.join(out_dir, "langfuse_traces.csv")
    scores_path = os.path.join(out_dir, "langfuse_scores.csv")

    df_traces = pd.json_normalize(traces_raw) if traces_raw else pd.DataFrame()
    df_scores = pd.json_normalize(scores_raw) if scores_raw else pd.DataFrame()

    # Keep only the columns that the Streamlit dashboard actually uses,
    # to shrink the CSVs without breaking metrics.
    if not df_traces.empty:
        trace_cols = [
            # core identity / timestamp (the app will pick one timestamp column)
            "id",
            "timestamp",
            "createdAt",
            "created_at",
            # tags and model-level metadata
            "tags",
            "metadata.model",
            "metadata.custom_id",
            "metadata.batch_eval",
            "metadata.run_id",
        ]
        existing_trace_cols = [c for c in trace_cols if c in df_traces.columns]
        df_traces = df_traces[existing_trace_cols]

    if not df_scores.empty:
        score_cols = [
            # id + metric identity
            "id",
            "score_id",
            "name",
            "score_name",
            "value",
            "timestamp",
            # join to traces
            "traceId",
            "trace_id",
            # tag-like columns (batch_evaluation etc.)
            "tags",
            "traceTags",
            "scoreTags",
            "trace.tags",
        ]
        existing_score_cols = [c for c in score_cols if c in df_scores.columns]
        df_scores = df_scores[existing_score_cols]

    df_traces.to_csv(traces_path, index=False)
    df_scores.to_csv(scores_path, index=False)

    print(f"Wrote traces to {traces_path}")
    print(f"Wrote scores to {scores_path}")


if __name__ == "__main__":
    main()