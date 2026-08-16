#!/usr/bin/env python3
"""Generate thesis figures from the frozen canonical evidence export."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle

from canonical_evidence import exact_unconditional_power


ROOT = Path(__file__).resolve().parent
EXPORT = ROOT / "notebooks" / "canonical_evidence_export"
BLUE = "#2A78D6"
ORANGE = "#EB6834"
INK = "#333333"
GREY = "#A9AAA7"
LIGHT_GREY = "#E8E8E4"


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(EXPORT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(EXPORT / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def monitoring_trajectory() -> None:
    data = pd.read_csv(EXPORT / "baseline_to_later_mcnemar.csv")
    data["baseline_started"] = pd.to_datetime(data["baseline_started"], utc=True)
    data["later_started"] = pd.to_datetime(data["later_started"], utc=True)
    rows: list[dict[str, object]] = []
    for model, group in data.groupby("model", sort=True):
        first = group.iloc[0]
        rows.append(
            {
                "model": model,
                "started": first["baseline_started"],
                "accuracy": first["acc_baseline"],
            }
        )
        for item in group.itertuples(index=False):
            rows.append(
                {
                    "model": model,
                    "started": item.later_started,
                    "accuracy": item.acc_later,
                }
            )
    timeline = pd.DataFrame(rows).drop_duplicates(["model", "started"]).sort_values("started")
    nominal = data.loc[data["p_down"] < 0.05].sort_values("later_started")

    fig, ax = plt.subplots(figsize=(7.25, 3.7))
    focus = "gpt-4.1-mini"
    haiku = "claude-haiku-4-5"
    for model, group in timeline.groupby("model"):
        if model == focus:
            color, lw, alpha, marker, zorder = ORANGE, 2.2, 1.0, "o", 3
        elif model == haiku:
            color, lw, alpha, marker, zorder = BLUE, 1.6, 0.95, None, 2
        else:
            color, lw, alpha, marker, zorder = GREY, 0.75, 0.5, None, 1
        ax.plot(
            group["started"],
            group["accuracy"],
            color=color,
            lw=lw,
            alpha=alpha,
            marker=marker,
            ms=3.2,
            zorder=zorder,
        )
    for model, group in nominal.groupby("model", sort=False):
        edge = ORANGE if model == focus else BLUE
        ax.scatter(
            group["later_started"],
            group["acc_later"],
            marker="o",
            s=46,
            facecolors="white",
            edgecolors=edge,
            linewidth=1.3,
            zorder=5,
        )
    annotations = {
        ("gpt-4.1-mini", "2026-03-22"): ("22 Mar: -12.5 pp", (18, -26), ORANGE),
        ("claude-haiku-4-5", "2026-06-22"): ("22 Jun: -2.6 pp", (10, 16), BLUE),
        ("claude-haiku-4-5", "2026-07-06"): ("6 Jul: -2.3 pp", (-52, -22), BLUE),
    }
    for row in nominal.itertuples(index=False):
        key = (row.model, row.later_started.strftime("%Y-%m-%d"))
        text, offset, color = annotations[key]
        ax.annotate(
            text,
            (row.later_started, row.acc_later),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            color=color,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8},
        )
    ax.scatter(
        [],
        [],
        marker="o",
        s=46,
        facecolors="white",
        edgecolors=INK,
        linewidth=1.3,
        label=r"nominal $p<0.05$",
    )
    ax.plot([], [], color=ORANGE, lw=2.2, marker="o", ms=3.2, label=focus)
    ax.plot([], [], color=BLUE, lw=1.6, label=haiku)
    ax.plot([], [], color=GREY, lw=1.0, label="other monitored models")
    ax.set_ylabel("Panel accuracy")
    ax.set_ylim(0.45, 1.01)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(axis="y", color=LIGHT_GREY, lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        frameon=False,
        ncol=4,
        fontsize=7.2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
    )
    ax.set_title("Canonical panel trajectories, 15 March-6 July 2026", fontsize=10, pad=32)
    fig.tight_layout()
    save(fig, "fig_canonical_monitoring_trajectory")


def transition_matrix() -> None:
    summary = json.loads((EXPORT / "canonical_summary.json").read_text())
    row = summary["monitoring"]["trigger_rows"][0]
    n = int(row["n_pairs"])
    b = int(row["b_degrade"])
    c = int(row["c_improve"])
    baseline_correct = round(float(row["acc_baseline"]) * n)
    a = baseline_correct - b
    d = n - a - b - c
    counts = np.array([[a, b], [c, d]])
    colors = np.array([[0, 1], [2, 0]])
    cmap = ListedColormap([LIGHT_GREY, "#F4A582", "#92C5DE"])

    fig, ax = plt.subplots(figsize=(4.8, 3.65))
    ax.imshow(colors, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    labels = [
        ["stable correct", "regressions"],
        ["improvements", "stable wrong"],
    ]
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{labels[i][j]}\n{counts[i, j]}",
                ha="center",
                va="center",
                color=INK,
                fontsize=10,
                fontweight="bold" if (i, j) in {(0, 1), (1, 0)} else "normal",
            )
    ax.set_xticks([0, 1], ["correct", "wrong"])
    ax.set_yticks([0, 1], ["correct", "wrong"])
    ax.set_xlabel("22 March outcome")
    ax.set_ylabel("15 March baseline outcome")
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(
        f"Paired transition matrix (n={n})\n49 regressions vs 12 improvements",
        fontsize=10,
    )
    fig.tight_layout()
    save(fig, "fig_canonical_mcnemar_matrix")


def power_heatmaps() -> None:
    n = 300
    tests = 173
    rhos = np.linspace(0.03, 0.20, 36)
    effects = np.linspace(0.005, 0.12, 40)
    alphas = [0.05, 0.05 / tests]
    titles = [r"Nominal $\alpha=0.05$", r"Conservative $\alpha=0.05/173$"]
    matrices: list[np.ndarray] = []
    for alpha in alphas:
        values = np.full((len(effects), len(rhos)), np.nan)
        for i, g in enumerate(effects):
            for j, rho in enumerate(rhos):
                if g <= rho:
                    values[i, j] = exact_unconditional_power(n, float(rho), float(g), alpha)
        matrices.append(values)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.35), sharex=True, sharey=True)
    image = None
    for ax, values, title in zip(axes, matrices, titles):
        image = ax.pcolormesh(
            rhos,
            effects * 100,
            values,
            shading="auto",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        ax.contour(rhos, effects * 100, values, levels=[0.8], colors="white", linewidths=1.3)
        for guide, dash in ((5.0, (4, 2)), (8.0, (1.5, 2.5))):
            line = ax.axhline(guide, color="white", linewidth=1.0, alpha=0.9)
            line.set_dashes(dash)
        ax.plot(0.12, 5.0, marker="x", color=ORANGE, ms=7, mew=2)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel(r"discordance $\rho$")
        ax.set_xticks([0.04, 0.08, 0.12, 0.16, 0.20])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("accuracy degradation g (pp)")
    axes[1].text(0.198, 5.15, "5pp", ha="right", va="bottom", fontsize=7.5, color="white")
    axes[1].text(0.198, 8.15, "8pp", ha="right", va="bottom", fontsize=7.5, color="white")
    assert image is not None
    cbar = fig.colorbar(image, ax=axes, fraction=0.035, pad=0.05)
    cbar.set_label("exact power")
    fig.suptitle("Exact unconditional McNemar power (n=300); white contour = 80%", fontsize=10)
    fig.subplots_adjust(left=0.09, right=0.86, bottom=0.17, top=0.82, wspace=0.12)
    save(fig, "fig_canonical_power_heatmap")


def cost_decision_map() -> None:
    data = pd.read_csv(EXPORT / "deployment_cost_map_source.csv")
    summary = json.loads((EXPORT / "canonical_summary.json").read_text())
    operating_r = float(summary["cost"]["R"])
    operating_lambda = float(summary["cost"]["lambda"])
    winners = data.loc[data["winner"]].copy()
    winner_names = sorted(winners["model"].unique())
    code = {name: i for i, name in enumerate(winner_names)}
    pivot = winners.pivot(index="lambda", columns="R", values="model")
    z = pivot.replace(code).to_numpy(dtype=float)
    x = pivot.columns.to_numpy(dtype=float)
    y = pivot.index.to_numpy(dtype=float)
    palette = ["#2A78D6", "#EB6834", "#5AAE61", "#9970AB", "#D8A03D", "#4D4D4D"]
    cmap = ListedColormap(palette[: len(winner_names)])
    norm = BoundaryNorm(np.arange(-0.5, len(winner_names) + 0.5), cmap.N)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.pcolormesh(x, y, z, shading="nearest", cmap=cmap, norm=norm)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"relative error cost $R=C_{\mathrm{FN}}/C_{\mathrm{FP}}$")
    ax.set_ylabel(r"total error severity $\lambda$ (USD)")
    ax.plot(operating_r, operating_lambda, marker="x", color="black", ms=8, mew=2)
    ax.annotate(
        "reported point",
        (operating_r, operating_lambda),
        xytext=(operating_r * 1.8, operating_lambda * 1.8),
        fontsize=8,
    )
    handles = [Patch(facecolor=palette[code[name]], label=name) for name in winner_names]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    ax.set_title("Cost-minimising model on the common 1113-item benchmark set", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_canonical_cost_decision_map")


def gepa_effect_plot() -> None:
    selected_runs = (
        ("gpt-4.1-nano", "20260713-184939Z"),
        ("gpt-5.4-nano", "20260713-184940Z"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    for ax, (model, run_id) in zip(axes, selected_runs, strict=True):
        run_dir = ROOT / "prompt_optimization" / "runs" / model / run_id
        summary = json.loads((run_dir / "summary.json").read_text())
        candidates = pd.read_csv(run_dir / "candidates.csv").set_index("candidate_idx")
        selected = int(summary["best_by_val"]["candidate_idx"])

        lineage = [selected]
        while lineage[-1] != 0:
            parent_text = str(candidates.loc[lineage[-1], "parent_ids"])
            parent = int(parent_text.strip("[]").split(",")[0])
            lineage.append(parent)
        lineage.reverse()

        validation = [100 * float(candidates.loc[index, "val_accuracy"]) for index in lineage]
        test = [100 * float(candidates.loc[index, "test_accuracy"]) for index in lineage]
        validation[0] = 100 * float(summary["baseline"]["val_accuracy"])
        test[0] = 100 * float(summary["baseline"]["test_accuracy"])
        x = np.arange(len(lineage))

        ax.plot(x, validation, color=BLUE, marker="o", lw=1.8, label="validation accuracy")
        ax.plot(x, test, color=ORANGE, marker="D", lw=1.8, label="descriptive test accuracy")
        ax.scatter([x[-1]], [validation[-1]], color=BLUE, marker="*", s=110, zorder=4)
        ax.scatter([x[-1]], [test[-1]], color=ORANGE, marker="*", s=110, zorder=4)
        ax.set_xticks(x, ["Seed", "Mutation 1", "Selected\nmutation 2"])
        test_delta = test[-1] - test[0]
        ax.set_title(f"{model}\ntest change {test_delta:+.1f} pp", fontsize=9)
        ax.set_xlabel("selected-candidate lineage")
        ax.grid(axis="y", color=LIGHT_GREY, lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("accuracy (%)")
    axes[0].set_ylim(65, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
    )
    fig.suptitle(
        "GEPA prompt trajectories: validation selection may or may not improve test accuracy",
        fontsize=10,
        y=0.99,
    )
    fig.subplots_adjust(top=0.66, bottom=0.18, wspace=0.16)
    save(fig, "fig_canonical_gepa_effects")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "pdf.fonttype": 42,
        }
    )
    monitoring_trajectory()
    transition_matrix()
    power_heatmaps()
    cost_decision_map()
    gepa_effect_plot()
    print("generated 5 canonical figure pairs")


if __name__ == "__main__":
    main()
