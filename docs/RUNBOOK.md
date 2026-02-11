# Model drift monitoring – runbook

## One-time setup

1. **API keys and secrets**  
   See [README → API keys and secrets](../README.md#api-keys-and-secrets) for the full list. For local runs set env vars; for GitHub Actions add them as repo secrets (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`).

2. **Langfuse**  
   Set env (or GitHub secrets): `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, optionally `LANGFUSE_BASE_URL`.

3. **Sync dataset to Langfuse** (once per dataset version):
   ```bash
   python scripts/sync_dataset.py --csv input/dataset.csv --name campaign_relevance
   ```

4. **Add a new model**  
   Edit `config/models.yaml`: add the model id under `openai` or `gemini` and add a `pricing` entry (input/output per 1M tokens, `batch_discount` 0–1).

## Normal runs

### Local: submit batches

```bash
# All models, subset 300 (uses data/baseline_predictions.json if present)
python scripts/run_eval.py --subset-size 300

# Specific models only
python scripts/run_eval.py --subset-size 50 --models gpt-5-mini,gemini-2.5-flash
```

This writes `data/batch_ids.json`. Batches typically complete within 24h.

### Local: poll, download, ingest

After batches are done:

```bash
python scripts/poll_and_ingest.py --batch-ids data/batch_ids.json --results-dir data/results --interval 600
```

- Polls every 10 min, downloads completed results to `data/results/<provider>_<model>.jsonl`, ingests into Langfuse (traces + accuracy score), then overwrites `data/baseline_predictions.json` from this run.

### First run (no baseline yet)

- No `data/baseline_predictions.json` → sampling is **stratified** by `campaign_relevant` (class balance).
- After a **full** run (all models, e.g. 1200 examples), run `poll_and_ingest` then optionally build baseline from that run’s results:
  - Baseline is written automatically by `poll_and_ingest` from `data/results/`.
- Next runs will use **disagreement-weighted** sampling (67% high, 23% medium, 10% baseline) if `data/baseline_predictions.json` exists.

### Aggregate baseline from existing JSONL only

```bash
python scripts/sample.py --aggregate --results-dir data/results --baseline data/baseline_predictions.json
```

## GitHub Actions

- **Weekly:** `Weekly drift monitoring` runs Monday 02:00 UTC: sample 300, submit all models, poll until done, ingest, update baseline.
- **Manual:** `Manual evaluation` → Run workflow, optional inputs: `models` (comma-separated), `subset_size` (default 300).

Secrets: `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`.

## When a batch fails

1. **Check batch status**  
   OpenAI: `python scripts/batch_openai.py <batch_id>`  
   Gemini: `python scripts/batch_gemini.py <batch_id>`

2. **Error file**  
   Status includes `error_file_id` when the batch has per-request errors. Download via API (e.g. `client.files.content(error_file_id)`) and inspect JSONL for failed `custom_id`s and error messages.

3. **Logs**  
   In GitHub Actions, open the “Submit batches” / “Poll, download, ingest” job logs. Failed batches are printed with status `failed` or `cancelled`; ingest only runs for successfully downloaded files.

4. **Retry**  
   Re-run the workflow or re-run `run_eval.py` with the same or smaller `--models` list. No automatic retry of single batches.

## Drift alerts

- In Langfuse, filter scores by **name = accuracy**, group by **metadata.model**, set time range.
- A **>5% drop** in accuracy week-over-week for a given model is a common threshold to investigate (prompt change, model version, or data shift).
- Cost: use usage (input/output tokens) from traces and multiply by `config/models.yaml` pricing (with batch_discount for OpenAI).

## E2E test (2 models, 50 examples)

```bash
# From repo root
python scripts/run_eval.py --subset-size 50 --models gpt-5-mini,gemini-2.5-flash --batch-ids data/batch_ids.json
# Wait for batches to complete (or poll manually), then:
python scripts/poll_and_ingest.py --batch-ids data/batch_ids.json --results-dir data/results --interval 300 --max-wait 7200
# Check Langfuse for new traces tagged batch_evaluation and scores name=accuracy.
```
