"""
Thesis figures (print, light surface only).

fig_panel_trajectory : panel accuracy over the monitoring period, gpt-4.1-mini
                       highlighted against the other monitored models.
fig_gepa_trajectory  : GEPA accepted-candidate vs running-best validation score,
                       from prompt_optimization/runs/gpt-4.1-nano/20260713-184939Z.

Palette: validated categorical slots 1-2 (#2a78d6, #eb6834); chrome from the
reference instance. Validator: all checks PASS (light, all-pairs).
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "assets" / "drift"
OUT.mkdir(parents=True, exist_ok=True)

SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
PANEL_MAX = 400

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": MUTED,
    "ytick.labelcolor": MUTED,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(True, axis="y")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote assets/drift/{name}.pdf / .png")


# ----------------------------------------------------------- figure 1
scores = pd.read_csv(ROOT / "streamlit_app" / "langfuse_scores.csv")
traces = pd.read_csv(ROOT / "streamlit_app" / "langfuse_traces.csv")
d = scores[scores["name"] == "accuracy"].merge(
    traces[["id", "timestamp", "metadata.model", "metadata.custom_id"]],
    left_on="traceId", right_on="id", how="left", suffixes=("_s", "_t"))
d["model"] = d["metadata.model"].astype(str)
d["item"] = d["metadata.custom_id"]
d["correct"] = d["value"].astype(int)
d["date"] = pd.to_datetime(d["timestamp_t"]).dt.tz_localize(None).dt.normalize()
d = d[(d.model != "nan") & d["item"].notna()].drop_duplicates(["model", "date", "item"])

n_items = d.groupby(["model", "date"])["item"].transform("nunique")
mon = d[n_items <= PANEL_MAX]                      # monitoring runs only

series = (mon.groupby(["model", "date"])["correct"].mean()
          .rename("acc").reset_index())
long = series.groupby("model")["date"].size()
series = series[series.model.isin(long[long >= 10].index)]

focus = "gpt-4.1-mini"
fig, ax = plt.subplots(figsize=(6.6, 3.0))
style(ax)

for m, g in series[series.model != focus].groupby("model"):
    g = g.sort_values("date")
    ax.plot(g.date, g.acc, color=MUTED, lw=0.8, alpha=0.45,
            solid_capstyle="round", zorder=1)

g = series[series.model == focus].sort_values("date")
ax.plot(g.date, g.acc, color=SERIES_1, lw=1.8, marker="o", markersize=3.4,
        markeredgecolor="white", markeredgewidth=0.5, solid_capstyle="round",
        zorder=3, label="gpt-4.1-mini")
ax.plot([], [], color=MUTED, lw=0.8, alpha=0.7,
        label="other monitored models")

dip = g[g.acc < 0.70]
lo = dip.loc[dip.acc.idxmin()]
ax.annotate("22–23 March\n$-13$pp, $-28$pp",
            xy=(lo.date, lo.acc), xytext=(14, 6), textcoords="offset points",
            fontsize=8, color=INK_2, ha="left", va="bottom",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7,
                            shrinkA=3, shrinkB=3))

ax.set_ylabel("panel accuracy")
ax.set_ylim(0.45, 1.0)
ax.legend(loc="lower right", fontsize=8, labelcolor=INK_2, ncol=2,
          handlelength=1.6, columnspacing=1.2)
ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%d %b"))
ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
save(fig, "fig_panel_trajectory")

# ----------------------------------------------------------- figure 2
gepa_run_dir = ROOT / "prompt_optimization" / "runs" / "gpt-4.1-nano" / "20260713-184939Z"
summary = json.loads((gepa_run_dir / "summary.json").read_text())
history = pd.read_csv(gepa_run_dir / "metrics_history.csv")
val_rows = history[history["event"] == "valset_evaluated"].sort_values("candidate_idx")
val = val_rows["val_accuracy"].astype(float).tolist()
best = pd.Series(val).cummax().tolist()
best_idx = int(summary["best_by_val"]["candidate_idx"])

fig, ax = plt.subplots(figsize=(4.4, 2.6))
style(ax)
x = list(range(len(val)))
ax.plot(x, best, color=SERIES_2, lw=1.8, solid_capstyle="round",
        zorder=2, label="running best")
ax.plot(x, val, color=SERIES_1, lw=1.8, marker="o", markersize=4,
        markeredgecolor="white", markeredgewidth=0.6, solid_capstyle="round",
        zorder=3, label="accepted candidate")
ax.plot([best_idx], [val[best_idx]], marker="o", markersize=7,
        color=SERIES_1, markeredgecolor="white", markeredgewidth=0.9, zorder=4)
ax.annotate(f"best: {val[best_idx]:.3f}",
            xy=(best_idx, val[best_idx]), xytext=(6, 6),
            textcoords="offset points", fontsize=8, color=INK_2)
ax.annotate(f"seed: {val[0]:.3f}", xy=(0, val[0]), xytext=(2, 8),
            textcoords="offset points", fontsize=8, color=INK_2)

ax.set_xlabel("accepted candidate index")
ax.set_ylabel("validation score")
ax.set_xticks(x)
ax.legend(loc="lower left", fontsize=8, labelcolor=INK_2, handlelength=1.6)
save(fig, "fig_gepa_trajectory")

print(f"\ngepa run: seed={val[0]:.4f} best={val[best_idx]:.4f} "
      f"(idx {best_idx}), candidates={len(val)}")
