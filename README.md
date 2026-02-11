# Model drift & quality monitoring

Batch-based evaluation (OpenAI + Gemini Batch APIs), disagreement-weighted subset sampling, Langfuse ingestion, and GitHub Actions for weekly/manual runs.

## API keys and secrets

Set these in your environment (or in a `.env` file at project root; see table below):

| Variable | Used by | Purpose |
|----------|--------|---------|
| `OPENAI_API_KEY` | batch_openai, run_eval, poll_and_ingest | OpenAI Batch API |
| `GOOGLE_API_KEY` | batch_gemini, upload_gemini_images | Gemini Batch API + Files (images) |
| `LANGFUSE_PUBLIC_KEY` | sync_dataset, ingest, poll_and_ingest | Langfuse project |
| `LANGFUSE_SECRET_KEY` | sync_dataset, ingest, poll_and_ingest | Langfuse auth |
| `LANGFUSE_HOST` or `LANGFUSE_BASE_URL` | sync_dataset (optional) | Langfuse server URL if not cloud |

**Do not commit keys.** Use `.env` at project root (in `.gitignore`). For CI, use GitHub Actions secrets only.

**Scripts layout:** Entrypoints you run live in `scripts/` (e.g. `run_eval.py`, `poll_and_ingest.py`, `ingest.py`, `report.py`, `sync_dataset.py`, `batch_openai.py`, `batch_gemini.py`, `sample.py`). Helpers used only by other scripts are in `scripts/lib/` (`load_dataset`, `image_cache`, `upload_gemini_images`); you don’t need to run those directly.

### Dataset source (default: Langfuse)

- **Default**: Examples and ground truth are loaded from the Langfuse dataset `campaign_relevance_02e1a68ccb0f`. Override the dataset name with env `LANGFUSE_DATASET_NAME`.
- **Override with local CSV**: Pass `--csv input/dataset.csv` to `run_eval`, `sample`, `poll_and_ingest`, `report`, or `upload_gemini_images` to use the local CSV instead.
- **Uploading to Langfuse** is a separate step: run `python scripts/sync_dataset.py --csv input/dataset.csv --name campaign_relevance` to push your CSV to a Langfuse dataset. When using the Langfuse source, the request body shape (e.g. `text`, `store`, `model`) is taken from the first row of `input/dataset.csv` as a template; that file must exist even when data comes from Langfuse.

---

## Step-by-step: run with 5 examples and see the dashboard

From **project root**. Use a subset of **5** for a fast end-to-end run.

### 1. Install and env

```bash
pip install -r requirements.txt
# Create .env with OPENAI_API_KEY, GOOGLE_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
```

### 2. Sync dataset to Langfuse (once per dataset)

```bash
python scripts/sync_dataset.py --csv input/dataset.csv --name campaign_relevance
```

### 3. Submit batches (subset of 5)

```bash
python scripts/run_eval.py --subset-size 5
```

- Samples 5 examples, creates batch jobs for all models in `config/models.yaml`.
- For Gemini: uploads dataset images to Gemini Files API (or uses cache in `data/gemini_image_cache.json`) then submits batch.
- Writes `data/batch_ids.json`.

### 4. Wait for batches, then poll, download, and ingest

Batches usually finish within minutes to a few hours. Then:

```bash
python scripts/poll_and_ingest.py --batch-ids data/batch_ids.json --results-dir data/results --interval 300 --max-wait 7200
```

- Polls every 5 min, downloads results to `data/results/`, ingests traces and accuracy scores into Langfuse, updates `data/baseline_predictions.json`.

### 5. Open the dashboard in Langfuse

1. Go to your Langfuse project.
2. **Scores** (or Analytics): filter **Score name** = `accuracy`, group by **metadata.model**, set time range.
3. **Traces**: filter by tag `batch_evaluation` and metadata `batch_eval: true` to see only this run.

Details: [docs/DASHBOARDS.md](docs/DASHBOARDS.md).

---

## Quick start (larger subset)

```bash
pip install -r requirements.txt
python scripts/sync_dataset.py --csv input/dataset.csv
python scripts/run_eval.py --subset-size 300
# After batches complete:
python scripts/poll_and_ingest.py --batch-ids data/batch_ids.json
```

See [docs/RUNBOOK.md](docs/RUNBOOK.md) for first-run vs baseline, failure handling, and GitHub Actions.
