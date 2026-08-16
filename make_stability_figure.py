"""
How stable is the paired (McNemar) comparison per model over the monitoring
period?  Reads assets/drift/drift_comparisons.csv (written by drift_analysis.py).

fig_stability : left  -- discordance psi = (b+c)/n of every baseline-to-current
                         comparison over time, most / least stable models
                         highlighted against the rest of the fleet.
                right -- median psi per model, min-max whisker, with the count
                         of raw-significant McNemar verdicts.

Palette: validated categorical slots 1-3 (#2a78d6, #eb6834, #1baf7a); chrome
from the reference instance.  Validator: all checks PASS (light, all-pairs);
slot 3 carries the contrast WARN, relieved by direct labels.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "assets" / "drift"

MIN_COMPARISONS = 10   # a median over 2-3 points is not a stability estimate
ALPHA = 0.05

SERIES_1, SERIES_2, SERIES_3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

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


def style(ax, axis="y"):
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- load
res = pd.read_csv(OUT / "drift_comparisons.csv")
res["current"] = pd.to_datetime(res["current"])

counts = res.groupby("model").size()
kept = counts[counts >= MIN_COMPARISONS].index
dropped = counts[counts < MIN_COMPARISONS]
res = res[res.model.isin(kept)]
print(f"{len(kept)} models with >= {MIN_COMPARISONS} comparisons "
      f"({len(res)} comparisons)")
print(f"excluded (too few comparisons to rank): "
      f"{', '.join(f'{m} ({n})' for m, n in dropped.items())}")

rank = (res.groupby("model")
        .agg(n_comp=("discordance", "size"),
             median=("discordance", "median"),
             lo=("discordance", "min"),
             hi=("discordance", "max"),
             n_sig=("alert_raw", "sum"),
             worst_delta=("delta", "min"))
        .sort_values("median").reset_index())
print("\nstability ranking (low psi = stable):")
print(rank.round(4).to_string(index=False))

least, most = rank.iloc[-1].model, rank.iloc[0].model
mid = "gpt-5-mini"                      # low churn, yet repeatedly significant
focus = {least: SERIES_2, mid: SERIES_3, most: SERIES_1}
pooled = res.discordance.median()

# ---------------------------------------------------------------- figure
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(9.2, 3.4), gridspec_kw={"width_ratios": [1.7, 1],
                                           "wspace": 0.46})
style(ax1)

for m, g in res[~res.model.isin(focus)].groupby("model"):
    g = g.sort_values("current")
    ax1.plot(g.current, g.discordance, color=MUTED, lw=0.8, alpha=0.4,
             solid_capstyle="round", zorder=1)

ax1.axhline(pooled, color=AXIS, lw=0.9, ls=(0, (4, 3)), zorder=1)
ax1.annotate(f"pooled median $\\hat\\psi$ = {pooled:.3f}",
             xy=(0.005, pooled), xycoords=("axes fraction", "data"),
             xytext=(0, 4), textcoords="offset points",
             fontsize=7.5, color=MUTED, va="bottom")

for m, color in focus.items():
    g = res[res.model == m].sort_values("current")
    ax1.plot(g.current, g.discordance, color=color, lw=1.8, marker="o",
             markersize=3.4, markeredgecolor="white", markeredgewidth=0.5,
             solid_capstyle="round", zorder=3)
    last = g.iloc[-1]
    ax1.annotate(m, xy=(last.current, last.discordance),
                 xytext=(6, 0), textcoords="offset points",
                 fontsize=8, color=INK_2, va="center")

sig = res[(res.alert_raw == 1) & (res.model.isin(focus))]
ax1.scatter(sig.current, sig.discordance, s=42, facecolors="none",
            edgecolors=INK, linewidths=0.9, zorder=4)
ax1.plot([], [], color=MUTED, lw=0.8, alpha=0.7,
         label="other monitored models")
ax1.scatter([], [], s=42, facecolors="none", edgecolors=INK, linewidths=0.9,
            label=f"McNemar $p<{ALPHA}$")

episode = res.loc[res.discordance.idxmax()]
ax1.annotate(f"{episode.model}, {episode.current:%d %b}\n"
             f"$\\Delta\\widehat{{A}}={episode.delta * 100:.0f}$pp",
             xy=(episode.current, episode.discordance), xytext=(10, -6),
             textcoords="offset points", fontsize=7.5, color=INK_2,
             ha="left", va="top",
             arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7,
                             shrinkA=2, shrinkB=3))

ax1.set_ylabel("discordance $\\hat\\psi = (b+c)/n$")
ax1.set_ylim(0, None)
ax1.set_xlim(res.current.min() - pd.Timedelta(days=6),
             res.current.max() + pd.Timedelta(days=34))
ax1.set_title("Per-comparison churn against the frozen baseline",
              fontsize=9, color=INK, loc="left", pad=8)
ax1.legend(loc="upper right", fontsize=8, labelcolor=INK_2, handlelength=1.6)
ax1.xaxis.set_major_formatter(mpl.dates.DateFormatter("%d %b"))
ax1.xaxis.set_major_locator(mpl.dates.MonthLocator())

# ------------------------------------------------------------ ranking
style(ax2, axis="x")
y = range(len(rank))
for i, r in enumerate(rank.itertuples()):
    color = focus.get(r.model, MUTED)
    ax2.plot([r.lo, r.hi], [i, i], color=color, lw=1.4,
             alpha=1.0 if r.model in focus else 0.35,
             solid_capstyle="round", zorder=2)
    ax2.plot([r.median], [i], marker="o", markersize=5.5, color=color,
             markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    if r.n_sig:
        ax2.annotate(f"{r.n_sig}/{r.n_comp} sig.", xy=(r.hi, i),
                     xytext=(6, 0), textcoords="offset points",
                     fontsize=7.5, color=INK_2 if r.model in focus else MUTED,
                     va="center")

ax2.set_yticks(list(y))
ax2.set_yticklabels(rank.model, fontsize=8)
for tick, m in zip(ax2.get_yticklabels(), rank.model):
    tick.set_color(focus.get(m, MUTED))
ax2.set_ylim(-0.7, len(rank) - 0.3)
ax2.set_xlim(0, rank.hi.max() * 1.28)
ax2.set_xlabel("$\\hat\\psi$: median, min-max over comparisons")
ax2.set_title("Churn is a per-model property, not a fleet constant",
              fontsize=9, color=INK, loc="left", pad=8)

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_stability.{ext}")
plt.close(fig)
print(f"\nwrote assets/drift/fig_stability.pdf / .png")
print(f"least stable: {least} (median psi {rank.iloc[-1]['median']:.3f}), "
      f"most stable: {most} ({rank.iloc[0]['median']:.3f}) -- "
      f"{rank.iloc[-1]['median'] / rank.iloc[0]['median']:.0f}x")