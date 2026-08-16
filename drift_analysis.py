"""
Baseline-to-current drift analysis on the frozen monitoring panel.

Corrections relative to the earlier comparison_analysis.py:
  * de-duplicates (model, day, item) before pairing -- 2026-03-23 had each
    panel item logged twice, which cross-joined to n=596 and inflated b/c.
  * separates monitoring runs (panel, <=400 items) from full-benchmark runs
    (~1199 items). Only monitoring runs enter the drift series.
  * multiplicity control is Bonferroni (alpha/(T-1)) and is named as such.
  * CUSUM / Kolmogorov-Smirnov columns removed (neither was implemented
    correctly: no cumulative sum, and K-S discards the pairing).

Outputs (assets/drift/):
  alert_counts.tex, control_same_week.tex, contingency_trigger.tex,
  discordance.tex, drift_comparisons.csv, summary.json
"""
import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).parent
OUT = ROOT / "assets" / "drift"
OUT.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
PANEL_MAX = 400        # a model-day with more unique items is a full-benchmark run
MIN_OVERLAP = 10
FLOOR = 0.03           # operational relevance floor, 3 percentage points

# ---------------------------------------------------------------- load
scores = pd.read_csv(ROOT / "streamlit_app" / "langfuse_scores.csv")
traces = pd.read_csv(ROOT / "streamlit_app" / "langfuse_traces.csv")

df = scores[scores["name"] == "accuracy"].merge(
    traces[["id", "timestamp", "metadata.model", "metadata.custom_id"]],
    left_on="traceId", right_on="id", how="left", suffixes=("_s", "_t"),
)
df["model"] = df["metadata.model"].astype(str)
df["item"] = df["metadata.custom_id"]
df["correct"] = df["value"].astype(int)
df["day"] = pd.to_datetime(df["timestamp_t"]).dt.strftime("%Y-%m-%d")
df = df[(df["model"] != "nan") & df["item"].notna()]

# one row per (model, day, item)
df = df.drop_duplicates(["model", "day", "item"])

# ---------------------------------------------------------------- run types
size = df.groupby(["model", "day"])["item"].nunique().rename("n_items").reset_index()
size["run_type"] = size["n_items"].where(size["n_items"] > PANEL_MAX).notna()
size["run_type"] = size["run_type"].map({True: "benchmark", False: "monitoring"})
df = df.merge(size, on=["model", "day"])

benchmark_days = size[size.run_type == "benchmark"]
print(f"monitoring model-days: {(size.run_type=='monitoring').sum()}")
print(f"benchmark  model-days: {len(benchmark_days)}  (excluded from drift series)")
print(benchmark_days.to_string(index=False))

mon = df[df.run_type == "monitoring"]

# ---------------------------------------------------------------- pairing
def contingency(model, d1, d2, frame):
    x = frame[(frame.model == model) & (frame.day == d1)][["item", "correct"]]
    y = frame[(frame.model == model) & (frame.day == d2)][["item", "correct"]]
    j = x.rename(columns={"correct": "t1"}).merge(
        y.rename(columns={"correct": "t2"}), on="item")
    n = len(j)
    if n == 0:
        return None
    a = int(((j.t1 == 1) & (j.t2 == 1)).sum())
    b = int(((j.t1 == 1) & (j.t2 == 0)).sum())
    c = int(((j.t1 == 0) & (j.t2 == 1)).sum())
    d = int(((j.t1 == 0) & (j.t2 == 0)).sum())
    p = binomtest(b, b + c, 0.5, alternative="greater").pvalue if b + c else 1.0
    return dict(n=n, a=a, b=b, c=c, d=d, acc1=(a + b) / n, acc2=(a + c) / n,
                delta=(c - b) / n, ndis=b + c, discordance=(b + c) / n, p=p)


rows = []
for model in sorted(mon.model.unique()):
    days = sorted(mon[mon.model == model].day.unique())
    if len(days) < 2:
        continue
    baseline, later = days[0], days[1:]
    bonf = ALPHA / len(later)          # Bonferroni over this model's comparisons
    for day in later:
        r = contingency(model, baseline, day, mon)
        if r is None or r["n"] < MIN_OVERLAP:
            continue
        rows.append(dict(
            model=model, baseline=baseline, current=day, n_comparisons=len(later),
            bonferroni=bonf, **r,
            alert_raw=int(r["b"] > r["c"] and r["p"] < ALPHA),
            alert_bonf=int(r["b"] > r["c"] and r["p"] < bonf),
            alert_bonf_floor=int(r["b"] > r["c"] and r["p"] < bonf
                                 and r["delta"] <= -FLOOR),
            alert_3pp=int(r["delta"] <= -0.03),
            alert_5pp=int(r["delta"] <= -0.05),
        ))

res = pd.DataFrame(rows)
res.to_csv(OUT / "drift_comparisons.csv", index=False)

counts = {
    "raw": int(res.alert_raw.sum()),
    "bonf": int(res.alert_bonf.sum()),
    "bonf_floor": int(res.alert_bonf_floor.sum()),
    "3pp": int(res.alert_3pp.sum()),
    "5pp": int(res.alert_5pp.sum()),
}
print(f"\ncomparisons={len(res)}  models={res.model.nunique()}")
print("alert counts:", counts)

sig = res[res.alert_raw == 1]
print("\nraw-significant comparisons:")
print(sig[["model", "baseline", "current", "n", "b", "c", "delta", "p",
           "bonferroni", "alert_bonf", "alert_bonf_floor"]].to_string(index=False))

# ---------------------------------------------------------------- same-week control
TRIGGER = "2026-03-22"
ctrl = []
for model in sorted(mon.model.unique()):
    days = sorted(mon[mon.model == model].day.unique())
    if TRIGGER not in days:
        continue
    if days[0] >= TRIGGER:
        continue
    r = contingency(model, days[0], TRIGGER, mon)   # earliest baseline, as the rule specifies
    if r and r["n"] >= MIN_OVERLAP:
        ctrl.append(dict(model=model, baseline=days[0], n=r["n"], b=r["b"],
                         c=r["c"], delta=r["delta"], p=r["p"]))
ctrl = pd.DataFrame(ctrl).sort_values("delta")
print(f"\nsame-week control at {TRIGGER} ({len(ctrl)} models):")
print(ctrl.to_string(index=False))

# ---------------------------------------------------------------- trajectory
TRAJ_MODEL = "gpt-4.1-mini"
panel = set(mon[(mon.model == TRAJ_MODEL)].sort_values("day").iloc[:300]["item"])
traj = (df[(df.model == TRAJ_MODEL) & (df["item"].isin(panel))]
        .groupby(["day", "run_type"])["correct"].agg(acc="mean", n="size")
        .reset_index().sort_values("day"))
traj.to_csv(OUT / "trajectory_gpt41mini.csv", index=False)
print(f"\n{TRAJ_MODEL} trajectory on panel items:")
print(traj.to_string(index=False))

# ---------------------------------------------------------------- discordance
disc = (res.groupby("model")
        .agg(n_comp=("discordance", "size"), median=("discordance", "median"),
             lo=("discordance", "min"), hi=("discordance", "max"))
        .sort_values("median", ascending=False).reset_index())
print("\ndiscordance vs baseline by model:")
print(disc.round(4).to_string(index=False))
print(f"\npooled median discordance = {res.discordance.median():.4f}")

# ---------------------------------------------------------------- LaTeX
def tex(path, body):
    (OUT / path).write_text(body)
    print(f"wrote assets/drift/{path}")


tex("alert_counts.tex", f"""\\begin{{table}}[htbp]
    \\centering
    \\caption{{Alert counts over {len(res)} baseline-to-current comparisons
    ({res.model.nunique()} models, monitoring runs only). Bonferroni is applied
    per model over that model's own comparisons.}}
    \\label{{tab:alert-counts}}
    \\begin{{tabular}}{{lr}}
    \\toprule
    Rule & Alerts \\\\
    \\midrule
    $b>c$, $p_{{\\mathrm{{exact}}}}<0.05$ (per-comparison)        & {counts['raw']} \\\\
    $b>c$, $p_{{\\mathrm{{exact}}}}<\\alpha/(T-1)$ (Bonferroni)    & {counts['bonf']} \\\\
    Bonferroni and $\\Delta\\widehat{{A}}\\le-3$pp                 & {counts['bonf_floor']} \\\\
    \\midrule
    $\\Delta\\widehat{{A}}\\le-3$pp alone                          & {counts['3pp']} \\\\
    $\\Delta\\widehat{{A}}\\le-5$pp alone                          & {counts['5pp']} \\\\
    \\bottomrule
    \\end{{tabular}}
\\end{{table}}
""")

def fmt_p(p):
    if p < 1e-4:
        mant, exp = f"{p:.1e}".split("e")
        return f"${mant}\\times 10^{{{int(exp)}}}$"
    return f"${p:.3f}$"


ctrl_rows = "\n".join(
    f"    \\texttt{{{r.model}}} & {r.baseline} & {r.n} & {r.b} & {r.c} & "
    f"${r.delta*100:+.2f}$ & {fmt_p(r.p)} \\\\"
    for r in ctrl.itertuples())
tex("control_same_week.tex", f"""\\begin{{table}}[htbp]
    \\centering
    \\footnotesize
    \\caption{{Same-batch control at the 22~March~2026 trigger. Every monitored
    model is compared against its own pre-trigger baseline on
    $\\mathcal{{D}}_n$. Only \\texttt{{gpt-4.1-mini}} moves.}}
    \\label{{tab:control-same-week}}
    \\begin{{tabular}}{{llcccrr}}
    \\toprule
    Model & Baseline & $n$ & $b$ & $c$ & $\\Delta\\widehat{{A}}$ (pp) & $p_{{\\mathrm{{exact}}}}$ \\\\
    \\midrule
{ctrl_rows}
    \\bottomrule
    \\end{{tabular}}
\\end{{table}}
""")

t = contingency("gpt-4.1-mini", "2026-03-08", "2026-03-22", mon)
tex("contingency_trigger.tex", f"""\\begin{{table}}[htbp]
    \\centering
    \\caption{{Paired correctness for \\texttt{{gpt-4.1-mini}} on
    $\\mathcal{{D}}_n$: 8~March~2026 baseline against the 22~March~2026
    trigger ($n={t['n']}$, $n_{{\\mathrm{{dis}}}}={t['ndis']}$).}}
    \\label{{tab:contingency-trigger}}
    \\begin{{tabular}}{{lcc}}
    \\toprule
     & 22~March correct & 22~March wrong \\\\
    \\midrule
    8~March correct & $a={t['a']}$ & $b={t['b']}$ \\\\
    8~March wrong   & $c={t['c']}$ & $d={t['d']}$ \\\\
    \\bottomrule
    \\end{{tabular}}
\\end{{table}}
""")
print(f"\ntrigger contingency (03-08 -> 03-22): {t}")

disc_rows = "\n".join(
    f"    \\texttt{{{r.model}}} & {r.n_comp} & {r.median:.3f} & {r.lo:.3f} & {r.hi:.3f} \\\\"
    for r in disc.itertuples())
tex("discordance.tex", f"""\\begin{{table}}[htbp]
    \\centering
    \\footnotesize
    \\caption{{Discordance rate $(b+c)/n$ between the baseline and each later
    monitoring run. Pooled median {res.discordance.median():.3f}, against the
    pilot estimate $\\hat\\rho=0.114$ used for sizing in
    Section~\\ref{{sec:monitoring-sample-size}}.}}
    \\label{{tab:discordance}}
    \\begin{{tabular}}{{lcccc}}
    \\toprule
    Model & Comparisons & Median & Min & Max \\\\
    \\midrule
{disc_rows}
    \\bottomrule
    \\end{{tabular}}
\\end{{table}}
""")

json.dump({"comparisons": len(res), "models": int(res.model.nunique()),
           "alert_counts": counts,
           "pooled_median_discordance": float(res.discordance.median()),
           "trigger_contingency": t,
           "benchmark_days_excluded": benchmark_days.to_dict("records")},
          open(OUT / "summary.json", "w"), indent=2, default=str)
print("\ndone")
