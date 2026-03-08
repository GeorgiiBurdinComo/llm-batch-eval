import streamlit as st
import pandas as pd
import altair as alt
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.auth import HTTPBasicAuth
from datetime import datetime
from dotenv import load_dotenv

st.set_page_config(page_title="Model Accuracy Dashboard", page_icon="📈", layout="wide")
st.title("Model Accuracy Over Time")

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))


def _secret(key, default=""):
    if key in os.environ:
        return os.environ[key]
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


LANGFUSE_PUBLIC_KEY = _secret("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = _secret("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = _secret("LANGFUSE_HOST", "https://cloud.langfuse.com")
SCORES_URL = f"{LANGFUSE_HOST}/api/public/scores"

_SESSION = requests.Session()

INPUT_PRICES = {
    "gpt-5.2": "$0.88", "gpt-5.1": "$0.63", "gpt-5": "$0.63", "gpt-5-mini": "$0.13",
    "gpt-5-nano": "$0.03", "gpt-4.1-mini": "$0.20", "gpt-4.1-nano": "$0.05", "gpt-4o-mini": "$0.08",
    "gemini-3-flash-preview": "$0.25", "gemini-2.5-flash": "$0.15",
    "gemini-2.5-flash-preview-09-2025": "$0.15", "gemini-2.0-flash": "$0.05",
    "claude-sonnet-4-6": "$1.50", "claude-sonnet-4-5": "$1.50", "claude-sonnet-4": "$1.50",
    "claude-haiku-4-5": "$0.50", "claude-3-5-haiku-20241022": "$0.40",
}


def _extract_model(trace):
    """Extract model name from trace tags or metadata."""
    for tag in trace.get("tags") or []:
        if tag.startswith("model:"):
            return tag.split("model:", 1)[1]
    meta = trace.get("metadata") or {}
    if isinstance(meta, dict) and "model" in meta:
        return meta["model"]
    return "unknown"


def _fetch_chunk(auth, chunk_start, chunk_end):
    """Fetch accuracy & cost_usd scores for one time window, paginating fully."""
    scores = []
    for name in ("accuracy", "cost_usd"):
        page = 1
        while True:
            res = _SESSION.get(SCORES_URL, auth=auth, params={
                "name": name, "traceTags": "batch_evaluation",
                "fromTimestamp": chunk_start, "toTimestamp": chunk_end,
                "limit": 100, "page": page,
            })
            if res.status_code != 200:
                break
            data = res.json()
            items = data.get("data", [])
            if not items:
                break
            for item in items:
                ts = item.get("timestamp") or item.get("createdAt")
                if not ts:
                    continue
                scores.append({
                    "timestamp": pd.to_datetime(ts),
                    "score_name": name,
                    "value": float(item.get("value", 0)),
                    "model": _extract_model(item.get("trace") or {}),
                })
            meta = data.get("meta", {})
            if meta.get("page", page) >= meta.get("totalPages", page):
                break
            page += 1
    return scores


@st.cache_data(ttl=900)
def fetch_scores(from_ts, to_ts):
    """Fetch scores in parallel daily chunks."""
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        st.error("Langfuse API keys not set. Configure in environment or Streamlit Secrets.")
        st.stop()

    auth = HTTPBasicAuth(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY)
    days = pd.date_range(from_ts.normalize(), to_ts.normalize(), freq="D")
    total = len(days)
    scores = []
    progress = st.progress(0, text="Fetching data from Langfuse...")
    done = 0

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(_fetch_chunk, auth, d.isoformat(), (d + pd.DateOffset(days=1)).isoformat()): d
            for d in days
        }
        for future in as_completed(futures):
            scores.extend(future.result())
            done += 1
            progress.progress(done / total, text=f"Fetched {done}/{total} days...")

    progress.empty()
    return pd.DataFrame(scores)


# ── Filters ──────────────────────────────────────────────────────────────────

today = datetime.now().date()
col1, col2 = st.columns(2)
with col1:
    date_from = st.date_input("From", value=today - pd.DateOffset(days=7), max_value=today)
with col2:
    date_to = st.date_input("To", value=today, max_value=today)

if date_from > date_to:
    st.error("'From' date must be before 'To' date.")
    st.stop()

df = fetch_scores(
    pd.Timestamp(date_from, tz="UTC"),
    pd.Timestamp(date_to, tz="UTC") + pd.DateOffset(days=1),
)

if df.empty:
    st.info("No scores found for the selected dates.")
    st.stop()

if df["timestamp"].dt.tz is None:
    df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

selected = st.multiselect("Models", sorted(df["model"].unique()), default=sorted(df["model"].unique()))
if not selected:
    st.warning("Select at least one model.")
    st.stop()

df = df[df["model"].isin(selected)]
df_acc = df[df["score_name"] == "accuracy"]
df_cost = df[df["score_name"] == "cost_usd"]

if df_acc.empty:
    st.warning("No accuracy data for the selected models / dates.")
    st.stop()

# ── Chart ────────────────────────────────────────────────────────────────────

daily = df_acc.groupby([df_acc["timestamp"].dt.date, "model"])["value"].mean().reset_index()
daily.columns = ["date", "model", "accuracy"]

st.subheader("Accuracy Over Time by Model")
st.altair_chart(
    alt.Chart(daily).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %d", labelAngle=-45)),
        y=alt.Y("accuracy:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1.05], clamp=True, nice=False)),
        color=alt.Color("model:N", title="Model", scale=alt.Scale(scheme="category10")),
        tooltip=["date:T", "model:N", alt.Tooltip("accuracy:Q", format=".2f")],
    ).properties(height=400).interactive()
)

# ── Table ────────────────────────────────────────────────────────────────────

st.subheader("Aggregated Results")

acc = df_acc.groupby("model")["value"].agg(["mean", "count"]).reset_index()
acc.columns = ["model", "Average Accuracy", "Evaluations"]

cost = df_cost.groupby("model")["value"].mean().reset_index()
cost.columns = ["model", "Avg Eval Cost (USD)"]

summary = acc.merge(cost, on="model", how="left")
summary["Input Cost (1M Tokens)"] = summary["model"].map(INPUT_PRICES).fillna("N/A")

st.dataframe(
    summary.style.format({"Average Accuracy": "{:.2%}", "Avg Eval Cost (USD)": "${:.6f}"}),
    width="stretch",
)
