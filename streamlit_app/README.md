# Streamlit Model Accuracy Dashboard

Streamlit app that visualizes benchmark accuracy, cost, and related stats from **offline CSV snapshots** of Langfuse scores and traces (`langfuse_scores.csv`, `langfuse_traces.csv` next to `app.py`). Stakeholders see charts without Langfuse access; the heavy lifting is exporting data once and shipping it with the app.

**Dashboard:** [llm-eval-dashboard.streamlit.app](https://llm-eval-dashboard.streamlit.app/) · **repo:** [GeorgiiBurdinComo/llm-eval-dashboard](https://github.com/GeorgiiBurdinComo/llm-eval-dashboard)

On the scheduled run (Monday 07:00 UTC) and when `main` changes under `streamlit_app/` in the **benchmark_eval** repo, the workflow `sync-streamlit-dashboard.yml` exports scores and traces from Langfuse into those CSV files, commits them with the Streamlit app, and pushes to the dashboard repository above. Streamlit Cloud builds and hosts the public app from that repo; the dashboard reads the CSVs committed there.

---

## Refreshing CSVs from Langfuse (export)

From the **benchmark_eval** repo root (same as CI: `sync-streamlit-dashboard.yml`):

```bash
pip install -r requirements.txt -r streamlit_app/requirements.txt
python streamlit_app/export_langfuse_csv.py
```

Writes `streamlit_app/langfuse_scores.csv` and `streamlit_app/langfuse_traces.csv`. Credentials via `.env` at repo root or env:

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"   # optional
```

---

## Running locally

1. **Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Data files** — `langfuse_scores.csv` and `langfuse_traces.csv` must exist next to `app.py` (export with `export_langfuse_csv.py` or copy from a recent sync).

3. **Google OAuth** — the app gates access with Google sign-in (`@gocomo.io` by default). Set:

   ```bash
   export GOOGLE_CLIENT_ID="..."
   export GOOGLE_CLIENT_SECRET="..."
   export GOOGLE_REDIRECT_URI="http://localhost:8501/"
   ```

   For Streamlit Cloud, put the same keys in **Secrets** as TOML (see below).

4. **Run**

   ```bash
   streamlit run app.py --server.runOnSave true
   ```

   Opens at `http://localhost:8501` (with reload on save).

---

## Deploying on Streamlit Community Cloud

The production app is the [llm-eval-dashboard](https://github.com/GeorgiiBurdinComo/llm-eval-dashboard) repo; CI keeps it in sync with this folder. To deploy or reconfigure:

1. Connect the repo at [share.streamlit.io](https://share.streamlit.io). The sync flattens this folder into the dashboard repo root, so the main file is `app.py` there (not `streamlit_app/app.py`).
2. **Secrets** — configure Google OAuth, for example:

   ```toml
   GOOGLE_CLIENT_ID = "..."
   GOOGLE_CLIENT_SECRET = "..."
   GOOGLE_REDIRECT_URI = "https://<your-app>.streamlit.app/"
   ```

   Chart data comes from the committed CSVs; you do **not** need Langfuse keys on Cloud for serving the dashboard (only for running `export_langfuse_csv.py` in CI or locally).
