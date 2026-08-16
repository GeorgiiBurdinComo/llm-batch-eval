"""
Error composition through the March gpt-4.1-mini episode, and the monitoring
cost ratio quoted in Chapter 7.

Companion to drift_analysis.py. That script pairs correctness only (b/c counts);
this one splits the same panel runs into TP/FP/TN/FN so the episode can be read
as a recall collapse rather than an undifferentiated accuracy drop.

Note on the baseline: the 8 March run carries `accuracy` scores but no
`error_type` scores, so the confusion series starts at 10 March. The paired
McNemar counts in drift_analysis.py still use 8 March as the baseline.

Outputs (assets/drift/):
  episode_confusion.tex, monitoring_cost.json
"""
import json
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "assets" / "drift"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-4.1-mini"
PRODUCTION_MODELS = ("gpt-5-mini", "gpt-5-nano", "gpt-5")
C_FP = 0.155
C_FN = 0.031
DAYS = ["2026-03-10", "2026-03-15", "2026-03-22", "2026-03-23", "2026-03-28"]
BASELINE = "2026-03-08"
PANEL_MAX = 400

scores = pd.read_csv(ROOT / "streamlit_app" / "langfuse_scores.csv")
traces = pd.read_csv(ROOT / "streamlit_app" / "langfuse_traces.csv")


def joined(score_name):
    x = scores[scores["name"] == score_name].merge(
        traces[["id", "timestamp", "metadata.model", "metadata.custom_id"]],
        left_on="traceId", right_on="id", how="left", suffixes=("_s", "_t"),
    )
    x["model"] = x["metadata.model"].astype(str)
    x["item"] = x["metadata.custom_id"]
    x["day"] = pd.to_datetime(x["timestamp_t"]).dt.strftime("%Y-%m-%d")
    x = x[(x["model"] != "nan") & x["item"].notna()]
    return x.drop_duplicates(["model", "day", "item"])


err = joined("error_type")
acc = joined("accuracy")
cost = joined("cost_usd")

# ------------------------------------------------------- confusion series
rows = []
for day in DAYS:
    vc = err[(err.model == MODEL) & (err.day == day)].stringValue.value_counts()
    tp, tn, fp, fn = (int(vc.get(k, 0)) for k in
                      ("true_positive", "true_negative", "false_positive", "false_negative"))
    n = tp + tn + fp + fn
    rows.append(dict(day=day, n=n, tp=tp, fp=fp, tn=tn, fn=fn,
                     acc=(tp + tn) / n, rec=tp / (tp + fn), spec=tn / (tn + fp)))
conf = pd.DataFrame(rows)
print(conf.round(4).to_string(index=False))

# ------------------------------------------ where the discordant items went
base = acc[(acc.model == MODEL) & (acc.day == BASELINE)][["item", "value"]].rename(
    columns={"value": "base"})
legs = {}
for day in ("2026-03-22", "2026-03-23"):
    cur = acc[(acc.model == MODEL) & (acc.day == day)][["item", "value"]].rename(
        columns={"value": "cur"})
    kind = err[(err.model == MODEL) & (err.day == day)][["item", "stringValue"]]
    j = base.merge(cur, on="item").merge(kind, on="item")
    b = j[(j.base == 1) & (j.cur == 0)]
    legs[day] = dict(n=len(j), b=len(b), c=int(((j.base == 0) & (j.cur == 1)).sum()),
                     b_types=b.stringValue.value_counts().to_dict())
    print(day, legs[day])

# ------------------------------------------------- panel vs benchmark cost
g = cost.groupby(["model", "day"]).agg(n=("item", "nunique"), usd=("value", "sum")).reset_index()
bench = g[(g.day == "2026-05-22") & (g.n > PANEL_MAX)].set_index("model")
panel = (g[(g.day.isin(["2026-05-18", "2026-05-25"])) & (g.n <= PANEL_MAX)]
         .groupby("model").agg(n=("n", "mean"), usd=("usd", "mean")))
pair = bench[["n", "usd"]].join(panel, rsuffix="_pan", how="inner")
cost_facts = dict(
    models=len(pair),
    benchmark_usd=round(float(pair.usd.sum()), 2),
    panel_usd=round(float(pair.usd_pan.sum()), 2),
    cost_ratio=round(float(pair.usd.sum() / pair.usd_pan.sum()), 2),
    item_ratio=round(float(pair.n.sum() / pair.n_pan.sum()), 2),
    per_item_benchmark=round(float(pair.usd.sum() / pair.n.sum()), 5),
    per_item_panel=round(float(pair.usd_pan.sum() / pair.n_pan.sum()), 5),
    median_monitoring_day_usd=round(float(
        g[g.n <= PANEL_MAX].groupby("day")["usd"].sum().median()), 2),
)
cost_facts["per_item_premium"] = round(
    cost_facts["per_item_panel"] / cost_facts["per_item_benchmark"], 2)
three_model_costs = []
for models in combinations(pair.index.tolist(), 3):
    selected = pair.loc[list(models)]
    three_model_costs.append(
        {
            "models": list(models),
            "benchmark_usd": float(selected["usd"].sum()),
            "panel_usd": float(selected["usd_pan"].sum()),
        }
    )
cheapest_three = min(three_model_costs, key=lambda item: item["panel_usd"])
costliest_three = max(three_model_costs, key=lambda item: item["panel_usd"])
cost_facts.update(
    three_model_panel_min_usd=round(cheapest_three["panel_usd"], 2),
    three_model_panel_max_usd=round(costliest_three["panel_usd"], 2),
    three_model_benchmark_min_usd=round(cheapest_three["benchmark_usd"], 2),
    three_model_benchmark_max_usd=round(costliest_three["benchmark_usd"], 2),
    three_model_weekly_panel_max_annual_usd=round(costliest_three["panel_usd"] * 52, 2),
    three_model_panel_min_models=cheapest_three["models"],
    three_model_panel_max_models=costliest_three["models"],
)

production = pair.loc[list(PRODUCTION_MODELS)]
production_panel_usd = float(production["usd_pan"].sum())
production_benchmark_usd = float(production["usd"].sum())

baseline_errors = err[
    (err.model == MODEL) & (err.day == "2026-03-15")
][["item", "stringValue"]].rename(columns={"stringValue": "baseline_error"})
alert_errors = err[
    (err.model == MODEL) & (err.day == "2026-03-22")
][["item", "stringValue"]].rename(columns={"stringValue": "alert_error"})
episode_pair = baseline_errors.merge(alert_errors, on="item")
episode_n = len(episode_pair)
baseline_fp = int(episode_pair["baseline_error"].eq("false_positive").sum())
baseline_fn = int(episode_pair["baseline_error"].eq("false_negative").sum())
alert_fp = int(episode_pair["alert_error"].eq("false_positive").sum())
alert_fn = int(episode_pair["alert_error"].eq("false_negative").sum())
baseline_error_loss = (C_FP * baseline_fp + C_FN * baseline_fn) / episode_n
alert_error_loss = (C_FP * alert_fp + C_FN * alert_fn) / episode_n
incremental_error_loss = alert_error_loss - baseline_error_loss
production_model_scenarios = {}
for model, row in production.iterrows():
    model_panel_usd = float(row["usd_pan"])
    production_model_scenarios[model] = {
        "panel_usd": round(model_panel_usd, 2),
        "benchmark_usd": round(float(row["usd"]), 2),
        "weekly_panel_annual_usd": round(model_panel_usd * 52, 2),
        "weekly_break_even_affected_items": round(
            model_panel_usd / incremental_error_loss
        ),
        "annual_break_even_affected_items": round(
            model_panel_usd * 52 / incremental_error_loss
        ),
    }

cost_facts.update(
    production_models=list(PRODUCTION_MODELS),
    production_panel_usd=round(production_panel_usd, 2),
    production_benchmark_usd=round(production_benchmark_usd, 2),
    production_weekly_panel_annual_usd=round(production_panel_usd * 52, 2),
    episode_cost_n=episode_n,
    episode_baseline_fp=baseline_fp,
    episode_baseline_fn=baseline_fn,
    episode_alert_fp=alert_fp,
    episode_alert_fn=alert_fn,
    episode_incremental_error_loss_per_item=round(incremental_error_loss, 6),
    production_weekly_break_even_affected_items=round(
        production_panel_usd / incremental_error_loss
    ),
    production_annual_break_even_affected_items=round(
        production_panel_usd * 52 / incremental_error_loss
    ),
    production_model_scenarios=production_model_scenarios,
)
print(json.dumps(cost_facts, indent=2))
(OUT / "monitoring_cost.json").write_text(json.dumps(cost_facts, indent=2))

# ------------------------------------------------------------------ LaTeX
label = {"2026-03-10": "10~Mar", "2026-03-15": "15~Mar", "2026-03-22": "22~Mar (alert)",
         "2026-03-23": "23~Mar (alert)", "2026-03-28": "28~Mar"}
body = "\n".join(
    f"    {label[r.day]} & {r.n} & {r.tp} & {r.fp} & {r.tn} & {r.fn} & "
    f"${r.acc:.3f}$ & ${r.rec:.3f}$ & ${r.spec:.3f}$ \\\\"
    for r in conf.itertuples())
(OUT / "episode_confusion.tex").write_text(f"""\\begin{{table}}[htbp]
    \\centering
    \\footnotesize
    \\caption[Error composition, gpt-4.1-mini episode]{{Error composition for
    \\texttt{{gpt-4.1-mini}} on $\\mathcal{{D}}_n$ across the March episode. The
    8~March baseline run logs correctness but not error type, so the series
    starts on 10~March.}}
    \\label{{tab:episode-confusion}}
    \\begin{{tabular}}{{lrrrrrccc}}
    \\toprule
    Evaluation & $n$ & TP & FP & TN & FN & $\\widehat{{\\mathrm{{Acc}}}}$ &
    $\\widehat{{\\mathrm{{Rec}}}}$ & $\\widehat{{\\mathrm{{Spec}}}}$ \\\\
    \\midrule
{body}
    \\bottomrule
    \\end{{tabular}}
\\end{{table}}
""")
print("wrote assets/drift/episode_confusion.tex")
