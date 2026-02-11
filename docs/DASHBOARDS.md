# Langfuse dashboards for drift monitoring

## Quality over time

1. In Langfuse, open **Scores** (or Analytics).
2. Filter: **Score name** = `accuracy`.
3. Group by: **metadata.model** (or tag `model:<id>`).
4. Set time range to last 7/30 days.
5. Use a line or bar chart to compare models and spot week-over-week drops.

Traces are created with `metadata.model` and `metadata.provider`; the ingestion script sets `trace.score(name="accuracy", value=0|1)`.

## Cost tracking

1. Traces include a **generation** span with `usage`: `input`, `output`, `total` (tokens), `unit: TOKENS`.
2. In Langfuse, use the usage/token metrics and group by `metadata.model`.
3. To get USD: multiply token counts by pricing from `config/models.yaml` (per 1M tokens, and apply `batch_discount` for OpenAI). This can be done in a spreadsheet or a small script; Langfuse does not apply custom pricing by default.

## Quick check

After a run, filter traces by tag **batch_evaluation** and metadata **batch_eval: true** to see only batch-eval traces.
