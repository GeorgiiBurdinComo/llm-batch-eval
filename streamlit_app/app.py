from __future__ import annotations

import os
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from math import comb
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import yaml
except ImportError:
    yaml = None

try:
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token as google_id_token

    HAS_GOOGLE_AUTH = True
except ImportError:
    HAS_GOOGLE_AUTH = False


# =============================================================================
# App setup
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent
CONFIG_MODELS_PATH = BASE_DIR / "config" / "models.yaml"

st.set_page_config(page_title="Model Accuracy Dashboard", layout="wide")
load_dotenv(BASE_DIR / ".env")


# =============================================================================
# Configuration
# =============================================================================

def get_secret(key: str, default: str = "") -> Any:
    env_val = os.environ.get(key)
    if env_val is not None:
        return env_val.strip().strip('"').strip("'")

    try:
        secret_val = st.secrets.get(key, default)
        if isinstance(secret_val, str):
            return secret_val.strip().strip('"').strip("'")
        return secret_val
    except Exception:
        return default


@dataclass(frozen=True)
class AppConfig:
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str
    google_client_id: str
    google_client_secret: str
    google_redirect_uri: str
    allowed_domain: str = "gocomo.io"

    @property
    def scores_url(self) -> str:
        return f"{self.langfuse_host}/api/public/scores"

    @property
    def scores_v2_url(self) -> str:
        return f"{self.langfuse_host}/api/public/v2/scores"

    @property
    def traces_url(self) -> str:
        return f"{self.langfuse_host}/api/public/traces"

    @property
    def has_langfuse_auth(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def has_google_oauth(self) -> bool:
        return bool(
            self.google_client_id
            and self.google_client_secret
            and self.google_redirect_uri
        )


def load_config() -> AppConfig:
    return AppConfig(
        langfuse_public_key=get_secret("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=get_secret("LANGFUSE_SECRET_KEY"),
        langfuse_host=get_secret("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        google_client_id=get_secret("GOOGLE_CLIENT_ID", ""),
        google_client_secret=get_secret("GOOGLE_CLIENT_SECRET", ""),
        google_redirect_uri=get_secret("GOOGLE_REDIRECT_URI", "http://localhost:8501/"),
    )


CONFIG = load_config()


def build_http_session() -> Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


HTTP = build_http_session()


# =============================================================================
# Errors
# =============================================================================

class AppError(Exception):
    """User-facing application error."""


# =============================================================================
# Auth
# =============================================================================

def require_google_login(config: AppConfig) -> None:
    if st.session_state.get("user_email"):
        return

    if not config.has_google_oauth:
        raise AppError("Google OAuth is not configured.")

    code = st.query_params.get("code")
    if code:
        email = exchange_google_code_for_email(code, config)
        if not email.endswith(f"@{config.allowed_domain}"):
            raise AppError(f"Access restricted to @{config.allowed_domain} accounts.")

        st.session_state["user_email"] = email
        st.query_params.clear()
        return

    auth_url = build_google_auth_url(config)
    st.markdown("### Model Accuracy Dashboard")
    st.link_button("Login with Google", auth_url)
    st.stop()


def build_google_auth_url(config: AppConfig) -> str:
    params = urllib.parse.urlencode(
        {
            "client_id": config.google_client_id,
            "redirect_uri": config.google_redirect_uri,
            "response_type": "code",
            "scope": "openid email",
            "access_type": "online",
            "prompt": "select_account",
            "hd": config.allowed_domain,
        }
    )
    return f"https://accounts.google.com/o/oauth2/v2/auth?{params}"


def exchange_google_code_for_email(code: str, config: AppConfig) -> str:
    token_res = HTTP.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code": code,
            "client_id": config.google_client_id,
            "client_secret": config.google_client_secret,
            "redirect_uri": config.google_redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=15,
    )

    if token_res.status_code != 200:
        try:
            payload = token_res.json()
            err = (
                payload.get("error_description")
                or payload.get("error")
                or token_res.text
            )
        except Exception:
            err = token_res.text
        raise AppError(f"Google login failed: {err}")

    token_payload = token_res.json()
    raw_id_token = token_payload.get("id_token", "")
    access_token = token_payload.get("access_token", "")

    if HAS_GOOGLE_AUTH and raw_id_token:
        try:
            info = google_id_token.verify_oauth2_token(
                raw_id_token,
                google_requests.Request(),
                config.google_client_id,
            )
        except Exception as exc:
            raise AppError("Invalid Google token.") from exc
    else:
        userinfo_res = HTTP.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        if userinfo_res.status_code != 200:
            raise AppError("Failed to fetch Google user info.")
        info = userinfo_res.json()

    return str(info.get("email", "")).strip().lower()


# =============================================================================
# Pricing
# =============================================================================

@st.cache_data(ttl=900)
def load_input_prices(config_path: Path) -> dict[str, str]:
    if yaml is None:
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return {}

    prices: dict[str, str] = {}
    for model, entry in (cfg.get("pricing") or {}).items():
        if isinstance(entry, dict) and "input" in entry:
            try:
                price = float(entry["input"])
                prices[model] = f"${price:.3f}".rstrip("0").rstrip(".")
            except (TypeError, ValueError):
                continue

    return prices


INPUT_PRICES = load_input_prices(CONFIG_MODELS_PATH)


# =============================================================================
# Local CSV loaders (offline Langfuse data)
# =============================================================================

SCORES_CSV = DATA_DIR / "langfuse_scores.csv"
TRACES_CSV = DATA_DIR / "langfuse_traces.csv"


@st.cache_data(ttl=900)
def load_scores_csv() -> pd.DataFrame:
    if not SCORES_CSV.exists():
        raise AppError(
            f"Local scores CSV not found at {SCORES_CSV}. "
            "Run export_langfuse_csv.py next to app.py first."
        )

    df = pd.read_csv(SCORES_CSV)
    if df.empty:
        raise AppError("Local scores CSV is empty.")

    # Normalise timestamp column
    ts_col = None
    for cand in ("timestamp", "createdAt", "created_at"):
        if cand in df.columns:
            ts_col = cand
            break
    if ts_col is None:
        raise AppError("Scores CSV does not contain a timestamp column.")

    df["timestamp"] = ensure_utc_timestamp(df[ts_col])
    return df


@st.cache_data(ttl=900)
def load_traces_csv() -> pd.DataFrame:
    if not TRACES_CSV.exists():
        raise AppError(
            f"Local traces CSV not found at {TRACES_CSV}. "
            "Run export_langfuse_csv.py next to app.py first."
        )

    df = pd.read_csv(TRACES_CSV)
    if df.empty:
        raise AppError("Local traces CSV is empty.")

    # Choose timestamp column and normalise
    ts_col = None
    for cand in ("timestamp", "createdAt", "created_at"):
        if cand in df.columns:
            ts_col = cand
            break
    if ts_col is None:
        raise AppError("Traces CSV does not contain a timestamp column.")

    trace_timestamp = ensure_utc_timestamp(df[ts_col])

    def _col(name: str) -> pd.Series:
        return df[name] if name in df.columns else pd.Series([None] * len(df))

    traces_df = pd.DataFrame(
        {
            "trace_id": df.get("id"),
            "trace_timestamp": trace_timestamp,
            "model": _col("metadata.model").fillna("unknown"),
            "custom_id": _col("metadata.custom_id"),
            "batch_eval": _col("metadata.batch_eval").fillna(False).astype(bool),
            "run_id": _col("metadata.run_id"),
            "tags": df.get("tags") if "tags" in df.columns else None,
        }
    )

    return traces_df


def _filter_interval(
    df: pd.DataFrame,
    ts_col: str,
    from_ts: pd.Timestamp,
    to_ts: pd.Timestamp,
) -> pd.DataFrame:
    if df.empty:
        return df
    ts = ensure_utc_timestamp(df[ts_col])
    mask = (ts >= from_ts) & (ts < to_ts)
    out = df.loc[mask].copy()
    out[ts_col] = ensure_utc_timestamp(out[ts_col])
    return out


def _has_batch_tag(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series([True] * 0)
    return series.astype(str).str.contains("batch_evaluation", na=False)


@st.cache_data(ttl=900)
def fetch_scores(from_ts: pd.Timestamp, to_ts: pd.Timestamp) -> pd.DataFrame:
    try:
        scores_df = load_scores_csv()
        traces_df = load_traces_csv()
    except AppError as exc:
        st.error(str(exc))
        st.stop()

    if scores_df.empty:
        return pd.DataFrame()

    scores_df = _filter_interval(scores_df, "timestamp", from_ts, to_ts)
    if scores_df.empty:
        return pd.DataFrame()

    # keep only relevant metrics
    keep_metrics = {"accuracy", "cost_usd", "error_type"}
    if "name" in scores_df.columns:
        scores_df = scores_df[scores_df["name"].isin(keep_metrics)]
    else:
        scores_df = scores_df[scores_df["score_name"].isin(keep_metrics)]

    if scores_df.empty:
        return pd.DataFrame()

    # filter by batch_evaluation tag if available
    tag_col = None
    for cand in ("tags", "traceTags", "scoreTags"):
        if cand in scores_df.columns:
            tag_col = cand
            break
    if tag_col is not None:
        scores_df = scores_df[_has_batch_tag(scores_df[tag_col])]

    if scores_df.empty:
        return pd.DataFrame()

    # join with traces to get model/custom_id/run_id
    traces_df = traces_df.drop_duplicates(subset=["trace_id"])

    if "traceId" in scores_df.columns:
        left_on = "traceId"
    elif "trace_id" in scores_df.columns:
        left_on = "trace_id"
    else:
        left_on = None

    if left_on is not None:
        merged = scores_df.merge(
            traces_df[["trace_id", "model", "custom_id", "run_id"]],
            left_on=left_on,
            right_on="trace_id",
            how="left",
        )
    else:
        merged = scores_df.copy()
        merged["model"] = "unknown"
        merged["custom_id"] = None
        merged["run_id"] = None

    if "name" in merged.columns:
        score_name_col = "name"
    else:
        score_name_col = "score_name"

    # Resolve categorical string value (error_type labels)
    str_val_col = None
    for cand in ("stringValue", "string_value", "valueString", "value_string"):
        if cand in merged.columns:
            str_val_col = cand
            break

    df = pd.DataFrame(
        {
            "timestamp": ensure_utc_timestamp(merged["timestamp"]),
            "score_name": merged[score_name_col],
            "value": pd.to_numeric(merged["value"], errors="coerce"),
            "string_value": merged[str_val_col] if str_val_col else pd.Series([None] * len(merged)),
            "model": merged.get("model", "unknown").fillna("unknown"),
            "custom_id": merged.get("custom_id"),
            "run_id": merged.get("run_id"),
        }
    )

    df = df.dropna(subset=["timestamp"])

    # Drop scores where we could not reliably attribute a model
    if "model" in df.columns:
        df = df[df["model"].notna() & (df["model"] != "unknown")]

    return df


@st.cache_data(ttl=900)
def fetch_traces_and_scores_for_window(from_ts: str, to_ts: str) -> pd.DataFrame:
    try:
        scores_df = load_scores_csv()
        traces_df = load_traces_csv()
    except AppError as exc:
        st.error(str(exc))
        st.stop()

    if scores_df.empty:
        return pd.DataFrame()

    start = pd.to_datetime(from_ts, utc=True)
    end = pd.to_datetime(to_ts, utc=True)

    scores_df = _filter_interval(scores_df, "timestamp", start, end)
    if scores_df.empty:
        return pd.DataFrame()

    # Only accuracy scores
    if "name" in scores_df.columns:
        scores_df = scores_df[scores_df["name"] == "accuracy"]
    else:
        scores_df = scores_df[scores_df["score_name"] == "accuracy"]

    if scores_df.empty:
        return pd.DataFrame()

    tag_col = None
    for cand in ("tags", "traceTags", "scoreTags"):
        if cand in scores_df.columns:
            tag_col = cand
            break
    if tag_col is not None:
        scores_df = scores_df[_has_batch_tag(scores_df[tag_col])]

    if scores_df.empty:
        return pd.DataFrame()

    traces_df = _filter_interval(traces_df, "trace_timestamp", start, end)

    traces_df = traces_df.drop_duplicates(subset=["trace_id"])

    if "traceId" in scores_df.columns:
        left_on = "traceId"
    elif "trace_id" in scores_df.columns:
        left_on = "trace_id"
    else:
        left_on = None

    if left_on is not None:
        merged = scores_df.merge(
            traces_df[
                ["trace_id", "trace_timestamp", "model", "custom_id", "batch_eval", "run_id"]
            ],
            left_on=left_on,
            right_on="trace_id",
            how="left",
        )
    else:
        merged = scores_df.copy()
        merged["trace_id"] = None
        merged["trace_timestamp"] = ensure_utc_timestamp(merged["timestamp"])
        merged["model"] = "unknown"
        merged["custom_id"] = None
        merged["batch_eval"] = False
        merged["run_id"] = None

    score_id_col = "id" if "id" in merged.columns else "score_id"

    df = pd.DataFrame(
        {
            "score_id": merged[score_id_col],
            "trace_id": merged.get("trace_id"),
            "timestamp": ensure_utc_timestamp(
                merged.get("timestamp", merged.get("trace_timestamp"))
            ),
            "accuracy": merged["value"],
            "model": merged.get("model", "unknown").fillna("unknown"),
            "custom_id": merged.get("custom_id"),
            "batch_eval": merged.get("batch_eval", False).fillna(False).astype(bool),
            "run_id": merged.get("run_id"),
        }
    )

    df = df.dropna(subset=["timestamp"])
    return df


# =============================================================================
# Analytics
# =============================================================================

def ensure_utc_timestamp(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    return dt


METRIC_OPTIONS: list[tuple[str, str]] = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("specificity", "Specificity"),
    ("balanced_accuracy", "Balanced Accuracy"),
]

_METRIC_LABELS: dict[str, str] = dict(METRIC_OPTIONS)


def metric_from_counts(tp: int, fp: int, tn: int, fn: int, key: str) -> float | None:
    n = tp + fp + tn + fn
    if n == 0:
        return None
    if key == "accuracy":
        return (tp + tn) / n
    if key == "precision":
        d = tp + fp
        return tp / d if d > 0 else None
    if key == "recall":
        d = tp + fn
        return tp / d if d > 0 else None
    if key == "f1":
        pd_ = tp + fp
        rd = tp + fn
        if pd_ == 0 or rd == 0:
            return None
        prec, rec = tp / pd_, tp / rd
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else None
    if key == "specificity":
        d = tn + fp
        return tn / d if d > 0 else None
    if key == "balanced_accuracy":
        rd, sd = tp + fn, tn + fp
        if rd == 0 or sd == 0:
            return None
        return 0.5 * (tp / rd + tn / sd)
    return None


def _confusion_counts(
    df_error_type: pd.DataFrame,
    groupby_cols: list[str],
) -> pd.DataFrame:
    """Pivot error_type string_value into TP/FP/TN/FN columns per group."""
    label_col = "string_value"
    if label_col not in df_error_type.columns:
        df_error_type = df_error_type.copy()
        df_error_type[label_col] = df_error_type.apply(_resolve_string_value, axis=1)

    valid_labels = ["true_positive", "false_positive", "true_negative", "false_negative"]
    df_et = df_error_type[df_error_type[label_col].isin(valid_labels)].copy()
    if df_et.empty:
        return pd.DataFrame()

    counts = (
        df_et.groupby(groupby_cols + [label_col])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=valid_labels, fill_value=0)
        .reset_index()
    )
    counts.columns.name = None
    return counts.rename(columns={
        "true_positive": "TP",
        "false_positive": "FP",
        "true_negative": "TN",
        "false_negative": "FN",
    })


def build_daily_metric(df_error_type: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    df = df_error_type.copy()
    df["date"] = df["timestamp"].dt.date
    counts = _confusion_counts(df, ["date", "model"])
    if counts.empty:
        return pd.DataFrame(columns=["date", "model", "metric_value"])

    counts["metric_value"] = counts.apply(
        lambda r: metric_from_counts(int(r["TP"]), int(r["FP"]), int(r["TN"]), int(r["FN"]), metric_key),
        axis=1,
    )
    return counts[["date", "model", "metric_value"]].dropna(subset=["metric_value"])


def build_metric_summary(
    df_error_type: pd.DataFrame,
    df_cost: pd.DataFrame,
    metric_key: str,
    *,
    cost_agg: str = "sum",
) -> tuple[pd.DataFrame, str, str]:
    counts = _confusion_counts(df_error_type, ["model"])
    if counts.empty:
        return pd.DataFrame(), "", ""

    counts["n"] = counts["TP"] + counts["FP"] + counts["TN"] + counts["FN"]
    metric_label = _METRIC_LABELS.get(metric_key, metric_key)
    counts[metric_label] = counts.apply(
        lambda r: metric_from_counts(int(r["TP"]), int(r["FP"]), int(r["TN"]), int(r["FN"]), metric_key),
        axis=1,
    )
    counts["Evaluations"] = counts["n"]

    if not df_cost.empty:
        if cost_agg == "sum":
            cost = df_cost.groupby("model")["value"].sum().reset_index()
            cost_col = "Total Cost (USD)"
        else:
            cost = df_cost.groupby("model")["value"].mean().reset_index()
            cost_col = "Avg Cost per Eval (USD)"
        cost.columns = ["model", cost_col]
    else:
        cost_col = "Total Cost (USD)" if cost_agg == "sum" else "Avg Cost per Eval (USD)"
        cost = pd.DataFrame(columns=["model", cost_col])

    summary = counts.merge(cost, on="model", how="left")
    summary[cost_col] = summary[cost_col].fillna(0.0)
    summary["Input Cost (1M Tokens)"] = summary["model"].map(INPUT_PRICES).fillna("N/A")
    return summary, cost_col, metric_label


def mcnemar_exact_pvalue(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return float("nan")

    k = min(b, c)
    p = 0.0
    for i in range(0, k + 1):
        p += comb(n, i)

    p = 2 * p / (2**n)
    return min(p, 1.0)


def day_bounds(day: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(day, tz="UTC")
    end = start + pd.DateOffset(days=1)
    return start, end


def compute_mcnemar_by_model(
    df: pd.DataFrame,
    ref_date: date,
    cmp_date: date,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}

    for model in sorted(df["model"].dropna().unique()):
        df_model = df[df["model"] == model].copy()
        if df_model.empty:
            continue

        df_model["date"] = df_model["timestamp"].dt.date
        df_two = df_model[
            df_model["custom_id"].notna() & df_model["date"].isin([ref_date, cmp_date])
        ].copy()

        if df_two.empty:
            continue

        pivot = df_two.pivot_table(
            index="custom_id",
            columns="date",
            values="accuracy",
        )

        if ref_date not in pivot.columns or cmp_date not in pivot.columns:
            continue

        pivot = pivot.dropna(subset=[ref_date, cmp_date])
        if pivot.empty:
            continue

        pivot = pivot.astype(int)
        y_ref = pivot[ref_date]
        y_cmp = pivot[cmp_date]

        a = int(((y_ref == 1) & (y_cmp == 1)).sum())
        b = int(((y_ref == 1) & (y_cmp == 0)).sum())
        c = int(((y_ref == 0) & (y_cmp == 1)).sum())
        d = int(((y_ref == 0) & (y_cmp == 0)).sum())

        n = b + c
        chi2 = ((abs(b - c) - 1) ** 2 / n) if n > 0 else float("nan")
        p_value = mcnemar_exact_pvalue(b, c)

        results[model] = {
            "contingency": pd.DataFrame(
                [[a, b], [c, d]],
                index=[f"{ref_date} correct", f"{ref_date} wrong"],
                columns=[f"{cmp_date} correct", f"{cmp_date} wrong"],
            ),
            "b": b,
            "c": c,
            "chi2": chi2,
            "p_value": p_value,
        }

    return results


def _resolve_string_value(row: pd.Series) -> str | None:
    """Coalesce categorical string value from multiple possible column names."""
    for col in ("string_value", "stringValue", "value_string", "valueString"):
        v = row.get(col)
        if pd.notna(v) and str(v).strip():
            return str(v).strip()
    v = row.get("value")
    if pd.notna(v) and str(v) in {
        "true_positive", "false_positive", "true_negative", "false_negative",
    }:
        return str(v)
    return None


def compute_expected_cost_summary(
    df_error_type: pd.DataFrame,
    df_cost: pd.DataFrame,
    c_fp: float,
    c_fn: float,
) -> pd.DataFrame:
    """Per-model expected cost E_t[C](m) = C_t(m) + (C_FP*FP + C_FN*FN) / n."""
    if df_error_type.empty:
        return pd.DataFrame()

    confusion = _confusion_counts(df_error_type, ["model"])
    if confusion.empty:
        return pd.DataFrame()

    confusion["n"] = confusion["TP"] + confusion["FP"] + confusion["TN"] + confusion["FN"]
    confusion["Accuracy"] = (confusion["TP"] + confusion["TN"]) / confusion["n"]

    avg_cost = (
        df_cost.groupby("model")["value"].mean().reset_index()
        if not df_cost.empty
        else pd.DataFrame(columns=["model", "value"])
    )
    avg_cost.columns = ["model", "Avg Inference Cost"]

    summary = confusion.merge(avg_cost, on="model", how="left")
    summary["Avg Inference Cost"] = summary["Avg Inference Cost"].fillna(0.0)

    summary["Expected Cost"] = (
        summary["Avg Inference Cost"]
        + (c_fp * summary["FP"] + c_fn * summary["FN"]) / summary["n"]
    )

    return summary


# =============================================================================
# UI helpers
# =============================================================================

def render_date_range_inputs(
    *,
    from_key: str,
    to_key: str,
    default_from: date,
    default_to: date,
    max_value: date,
) -> tuple[date, date]:
    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input("From", value=default_from, max_value=max_value, key=from_key)
    with col2:
        date_to = st.date_input("To", value=default_to, max_value=max_value, key=to_key)

    if date_from > date_to:
        raise AppError("'From' date must be before 'To' date.")

    return date_from, date_to


def render_model_selector(df: pd.DataFrame, key: str) -> list[str]:
    models = sorted(df["model"].dropna().unique())
    selected = st.multiselect("Models", models, default=models, key=key)
    if not selected:
        raise AppError("Select at least one model.")
    return selected


def metric_chart(daily_df: pd.DataFrame, metric_label: str) -> alt.Chart:
    return (
        alt.Chart(daily_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %d", labelAngle=-45)),
            y=alt.Y(
                "metric_value:Q",
                title=metric_label,
                scale=alt.Scale(domain=[0, 1.05], clamp=True, nice=False),
            ),
            color=alt.Color("model:N", title="Model"),
            tooltip=[
                "date:T",
                "model:N",
                alt.Tooltip("metric_value:Q", title=metric_label, format=".2f"),
            ],
        )
        .properties(height=400)
        .interactive()
    )


# =============================================================================
# Tabs
# =============================================================================

def render_overview_tab(default_date: date, today: date) -> None:
    st.subheader("Model Metrics Over Time")

    metric_key = st.selectbox(
        "Metric",
        options=[k for k, _ in METRIC_OPTIONS],
        format_func=lambda k: _METRIC_LABELS[k],
        index=0,
        key="overview_metric",
    )
    metric_label = _METRIC_LABELS[metric_key]

    date_from, date_to = render_date_range_inputs(
        from_key="overview_from",
        to_key="overview_to",
        default_from=default_date,
        default_to=today,
        max_value=today,
    )

    df = fetch_scores(
        pd.Timestamp(date_from, tz="UTC"),
        pd.Timestamp(date_to, tz="UTC") + pd.DateOffset(days=1),
    )

    if df.empty:
        st.info("No scores found for the selected dates.")
        return

    df["timestamp"] = ensure_utc_timestamp(df["timestamp"])

    selected_models = render_model_selector(df, key="overview_models")
    df = df[df["model"].isin(selected_models)]

    df_error_type = df[df["score_name"] == "error_type"].copy()
    df_cost = df[df["score_name"] == "cost_usd"].copy()

    if df_error_type.empty:
        st.warning(
            "No error\\_type scores found. "
            "Re-run `export_langfuse_csv.py` to include `stringValue` for categorical scores."
        )
        return

    daily = build_daily_metric(df_error_type, metric_key)
    if daily.empty:
        st.warning(f"No valid data to compute {metric_label} for the selected models / dates.")
        return

    st.subheader(f"{metric_label} Over Time by Model")
    st.altair_chart(metric_chart(daily, metric_label), use_container_width=True)

    last_ts = df["timestamp"].max()
    if pd.notna(last_ts):
        last_date = last_ts.date()
        df_et_last = df_error_type[df_error_type["timestamp"].dt.date == last_date]
        df_cost_last = df_cost[df_cost["timestamp"].dt.date == last_date]

        if not df_et_last.empty:
            st.subheader(f"Last Launch ({last_date.isoformat()})")
            summary_last, cost_col_last, ml = build_metric_summary(
                df_et_last, df_cost_last, metric_key, cost_agg="sum",
            )
            if not summary_last.empty:
                st.dataframe(
                    summary_last[["model", ml, "Evaluations", cost_col_last, "Input Cost (1M Tokens)"]]
                    .style.format({ml: "{:.2%}", cost_col_last: "${:.6f}"}),
                    use_container_width=True,
                )

    st.subheader("Aggregated Results (Selected Period)")
    summary, cost_col, ml = build_metric_summary(
        df_error_type, df_cost, metric_key, cost_agg="mean",
    )
    if not summary.empty:
        st.dataframe(
            summary[["model", ml, "Evaluations", cost_col, "Input Cost (1M Tokens)"]]
            .style.format({ml: "{:.2%}", cost_col: "${:.6f}"}),
            use_container_width=True,
        )


def render_mcnemar_tab(default_date: date) -> None:
    st.subheader("McNemar Test on Trace Accuracy")

    col1, col2 = st.columns(2)
    with col1:
        ref_date = st.date_input("Reference date", value=default_date, key="mcnemar_ref_date")
    with col2:
        cmp_date = st.date_input("Comparison date", value=default_date, key="mcnemar_cmp_date")

    if ref_date == cmp_date:
        st.warning("Reference and comparison dates must be different.")
        return

    ref_start, ref_end = day_bounds(ref_date)
    cmp_start, cmp_end = day_bounds(cmp_date)

    df_ref = fetch_traces_and_scores_for_window(ref_start.isoformat(), ref_end.isoformat())
    df_cmp = fetch_traces_and_scores_for_window(cmp_start.isoformat(), cmp_end.isoformat())

    df_mcnemar = pd.concat([df_ref, df_cmp], ignore_index=True)
    if df_mcnemar.empty:
        st.info("No scores found for the selected McNemar window.")
        return

    results = compute_mcnemar_by_model(df_mcnemar, ref_date, cmp_date)
    if not results:
        st.info("No models have sufficient overlapping data between the two dates for a McNemar test.")
        return

    for model, result in results.items():
        st.markdown(f"### Model: `{model}`")
        st.markdown("**Contingency table (McNemar)**")
        st.table(result["contingency"])
        st.markdown(f"**Discordant pairs (b, c):** `{result['b']}, {result['c']}`")

        if pd.notna(result["chi2"]):
            st.markdown(f"**McNemar chi² (with continuity correction):** `{result['chi2']:.4f}`")
        else:
            st.markdown("**McNemar chi²:** undefined (no discordant pairs)")

        st.markdown(f"**Exact binomial p-value:** `{result['p_value']:.6g}`")


def render_expected_cost_tab(default_date: date, today: date) -> None:
    st.markdown("### Cost-Aware Deployment: Expected Cost $E_t[\\mathcal{C}](m)$")

    st.markdown(
        "The per-request expected cost combines inference expenditure "
        "with asymmetric misclassification penalties:"
    )
    st.latex(
        r"E_t[\mathcal{C}](m)"
        r" = C_t(m)"
        r" + \frac{C_{\mathrm{FP}} \cdot \mathrm{FP}_t(m)"
        r"       + C_{\mathrm{FN}} \cdot \mathrm{FN}_t(m)}{n}"
    )
    st.markdown(
        "- $C_t(m)$: average inference cost (USD per request)\n"
        "- $C_{\\mathrm{FP}}$: business cost of one false positive\n"
        "- $C_{\\mathrm{FN}}$: business cost of one false negative\n"
        "- For **ranking** models only the ratio "
        "$r = C_{\\mathrm{FN}} / C_{\\mathrm{FP}}$ matters.\n"
        "- The preferred model minimises $E_t[\\mathcal{C}](m)$."
    )

    date_from, date_to = render_date_range_inputs(
        from_key="cost_from",
        to_key="cost_to",
        default_from=default_date,
        default_to=today,
        max_value=today,
    )

    df = fetch_scores(
        pd.Timestamp(date_from, tz="UTC"),
        pd.Timestamp(date_to, tz="UTC") + pd.DateOffset(days=1),
    )

    if df.empty:
        st.info("No scores found for the selected dates.")
        return

    df["timestamp"] = ensure_utc_timestamp(df["timestamp"])

    selected_models = render_model_selector(df, key="cost_models")
    df = df[df["model"].isin(selected_models)]

    df_error_type = df[df["score_name"] == "error_type"].copy()
    df_cost = df[df["score_name"] == "cost_usd"].copy()

    if df_error_type.empty:
        st.warning(
            "No error\\_type scores found. "
            "Re-run `export_langfuse_csv.py` to include `stringValue` for categorical scores."
        )
        return

    st.divider()

    st.subheader("Misclassification trade-off (linked inputs)")

    if "_cost_last_changed" not in st.session_state:
        st.session_state["_cost_last_changed"] = "init"

    def _sync_from_r() -> None:
        st.session_state["_cost_last_changed"] = "r"
        r_val = float(st.session_state.get("cost_ratio_r", 0.0) or 0.0)
        c_fp_val = float(st.session_state.get("cost_c_fp_usd", 0.0) or 0.0)
        if r_val > 0:
            st.session_state["cost_c_fn_usd"] = c_fp_val / r_val
        else:
            # If R==0, only consistent solution is C_FP==0 (else undefined).
            st.session_state["cost_c_fn_usd"] = 0.0
            st.session_state["cost_c_fp_usd"] = 0.0

    def _sync_from_c_fp() -> None:
        st.session_state["_cost_last_changed"] = "c_fp"
        c_fp_val = float(st.session_state.get("cost_c_fp_usd", 0.0) or 0.0)
        c_fn_val = float(st.session_state.get("cost_c_fn_usd", 0.0) or 0.0)
        if c_fn_val > 0:
            st.session_state["cost_ratio_r"] = c_fp_val / c_fn_val
        else:
            st.session_state["cost_ratio_r"] = 0.0

    def _sync_from_c_fn() -> None:
        st.session_state["_cost_last_changed"] = "c_fn"
        c_fp_val = float(st.session_state.get("cost_c_fp_usd", 0.0) or 0.0)
        c_fn_val = float(st.session_state.get("cost_c_fn_usd", 0.0) or 0.0)
        if c_fn_val > 0:
            st.session_state["cost_ratio_r"] = c_fp_val / c_fn_val
        else:
            st.session_state["cost_ratio_r"] = 0.0

    col_cfp, col_cfn, col_r = st.columns([1, 1, 1])
    with col_cfp:
        st.number_input(
            "C_FP (USD)",
            min_value=0.0,
            value=float(st.session_state.get("cost_c_fp_usd", 1.0) or 1.0),
            step=0.10,
            format="%.2f",
            key="cost_c_fp_usd",
            on_change=_sync_from_c_fp,
        )
    with col_cfn:
        st.number_input(
            "C_FN (USD)",
            min_value=0.0,
            value=float(st.session_state.get("cost_c_fn_usd", 1.0) or 1.0),
            step=0.10,
            format="%.2f",
            key="cost_c_fn_usd",
            on_change=_sync_from_c_fn,
        )
    with col_r:
        st.number_input(
            "R = C_FP / C_FN",
            min_value=0.0,
            value=float(st.session_state.get("cost_ratio_r", 1.0) or 1.0),
            step=0.10,
            format="%.2f",
            key="cost_ratio_r",
            help="You can edit any two of (C_FP, C_FN, R). The third auto-updates.",
            on_change=_sync_from_r,
        )

    c_fp = float(st.session_state.get("cost_c_fp_usd", 0.0) or 0.0)
    c_fn = float(st.session_state.get("cost_c_fn_usd", 0.0) or 0.0)
    r_val = float(st.session_state.get("cost_ratio_r", 0.0) or 0.0)

    st.caption(
        f"Linked: **C_FP=${c_fp:.2f}**, **C_FN=${c_fn:.2f}**, **R={r_val:.2f}** "
        f"(meaning 1 FP costs like {r_val:.2f} FNs)."
    )

    if df_cost.empty:
        st.info("No cost data found; inference cost assumed $0 for all models.")

    summary = compute_expected_cost_summary(df_error_type, df_cost, c_fp=float(c_fp), c_fn=float(c_fn))
    if summary.empty:
        st.warning(
            "Could not compute confusion matrix. "
            "Ensure `stringValue` is present in `langfuse_scores.csv` for error\\_type rows."
        )
        return

    cost_col = "Expected Cost"
    acc_col = "Accuracy"
    inf_col = "Avg Inference Cost"

    best_model: str | None = None
    if summary[cost_col].notna().any():
        best_row = summary.loc[summary[cost_col].idxmin()]
        best_model = str(best_row["model"])
        st.success(
            f"Best model: **{best_model}** with E[C] = ${best_row[cost_col]:.6f}/req "
            f"(R={r_val:.2f}, C_FN=${c_fn:.2f}, C_FP=${c_fp:.2f})."
        )

    st.subheader("Expected cost")
    chart = (
        alt.Chart(summary)
        .mark_bar()
        .encode(
            x=alt.X("model:N", title="Model", sort="-y"),
            y=alt.Y(f"{cost_col}:Q", title="Expected Cost (USD per request)"),
            color=alt.condition(
                alt.datum.model == best_model,
                alt.value("#2ca02c"),
                alt.value("#1f77b4"),
            ),
            tooltip=[
                "model:N",
                alt.Tooltip(f"{cost_col}:Q", format="$.6f"),
                alt.Tooltip(f"{inf_col}:Q", format="$.6f"),
                alt.Tooltip(f"{acc_col}:Q", format=".2%"),
                "FP:Q",
                "FN:Q",
                "n:Q",
            ],
        )
        .properties(height=360)
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Per-model confusion matrix and expected cost")
    display_cols = ["model", "TP", "FP", "TN", "FN", "n", acc_col, inf_col, cost_col]
    styled = (
        summary[display_cols]
        .style.format(
            {
                acc_col: "{:.2%}",
                inf_col: "${:.6f}",
                cost_col: "${:.6f}",
            }
        )
        .highlight_min(subset=[cost_col], color="#d1ffd1")
    )
    st.dataframe(styled, use_container_width=True)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    today = datetime.now().date()
    default_date = date(2026, 3, 8)

    require_google_login(CONFIG)

    st.title("Model Accuracy Dashboard")

    tab_overview, tab_mcnemar, tab_cost = st.tabs(
        ["Overview", "McNemar Test", "Expected Cost"]
    )

    with tab_overview:
        render_overview_tab(default_date, today)

    with tab_mcnemar:
        render_mcnemar_tab(default_date)

    with tab_cost:
        render_expected_cost_tab(default_date, today)


try:
    main()
except AppError as exc:
    st.error(str(exc))