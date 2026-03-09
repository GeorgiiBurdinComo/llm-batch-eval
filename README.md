# Model drift & quality monitoring

Batch-based evaluation (OpenAI + Gemini + Claude Batch APIs), fixed eval subset from Langfuse, Langfuse ingestion, and GitHub Actions for weekly/manual runs.

## API keys and secrets

Set these in your environment (or in a `.env` file at project root; see table below):

| Variable | Used by | Purpose |
|----------|--------|---------|
| `OPENAI_API_KEY` | batch_openai, run_eval, poll_and_ingest | OpenAI Batch API |
| `GOOGLE_API_KEY` | batch_gemini, upload_gemini_images | Gemini Batch API + Files (images) |
| `ANTHROPIC_API_KEY` | batch_claude, run_eval, poll_and_ingest | Claude (Anthropic) Message Batches |
| `LANGFUSE_PUBLIC_KEY` | sync_dataset, ingest, poll_and_ingest | Langfuse project |
| `LANGFUSE_SECRET_KEY` | sync_dataset, ingest, poll_and_ingest | Langfuse auth |
| `LANGFUSE_HOST` or `LANGFUSE_BASE_URL` | sync_dataset (optional) | Langfuse server URL if not cloud |

**Do not commit keys.** Use `.env` at project root (in `.gitignore`). For CI, use GitHub Actions secrets only.

**Scripts layout:** Entrypoints you run live in `scripts/` (e.g. `run_eval.py`, `poll_and_ingest.py`, `ingest.py`, `report.py`, `sync_dataset.py`, `batch_openai.py`, `batch_gemini.py`). Helpers used only by other scripts are in `scripts/lib/` (`load_dataset`, `image_cache`, `upload_gemini_images`); you don't need to run those directly.

### Dataset source (default: Langfuse)

- **Default**: The eval subset is loaded from a Langfuse dataset. Override the dataset name with `--langfuse-dataset` or env `LANGFUSE_DATASET_NAME`.
- **Creating the eval subset**: Create a Langfuse dataset manually with your chosen examples (e.g. disagreement-based selection). `run_eval.py` will use all items from that dataset.
- **Override with local CSV**: Pass `--csv input/dataset.csv` to `run_eval`, `poll_and_ingest`, `report`, or `upload_gemini_images` to use the local CSV instead.
- **Uploading full dataset to Langfuse**: run `python scripts/sync_dataset.py --csv input/dataset.csv --name campaign_relevance` to push your CSV to a Langfuse dataset.

---

## Step-by-step

From **project root**.

### 1. Install and env

```bash
pip install -r requirements.txt
# Create .env with OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
```

### 2. Submit batches

```bash
python scripts/run_eval.py --langfuse-dataset eval_subset
```

- Loads all examples from the `eval_subset` Langfuse dataset.
- Creates batch jobs for all models in `config/models.yaml`.
- Writes `data/batch_ids.json`.

### 3. Wait for batches, then poll, download, and ingest

Batches usually finish within few hours. Then:

```bash
python scripts/poll_and_ingest.py
```

- Polls every 5 min, downloads results to `data/results/`, ingests traces and accuracy scores into Langfuse.

### 4. View traces and dashboard

Traces and accuracy scores live in **Langfuse**. The Streamlit app is a read-only viewer that displays them:

- **Dashboard:** https://llm-eval-dashboard.streamlit.app/

Details: [docs/DASHBOARDS.md](docs/DASHBOARDS.md).

---

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_eval.py --langfuse-dataset eval_subset
# After batches complete:
python scripts/poll_and_ingest.py --batch-ids data/batch_ids.json
```

See [docs/RUNBOOK.md](docs/RUNBOOK.md) for first-run vs baseline, failure handling, and GitHub Actions.
