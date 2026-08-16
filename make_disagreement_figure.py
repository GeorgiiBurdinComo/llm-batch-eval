#!/usr/bin/env python3
"""Build a real-runs disagreement matrix on the 1199-item benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle

from canonical_evidence import find_root, load_benchmark, load_measurements


ROOT = Path(__file__).resolve().parent
EXPORT = ROOT / "notebooks" / "canonical_evidence_export"
FIGURE_STEM = "fig_canonical_disagreement_matrix"
SOURCE_NAME = "canonical_disagreement_matrix_source.csv"
SUMMARY_NAME = "canonical_disagreement_matrix_summary.json"
MIN_ITEMS = 1180
BLUE = "#4e79a7"
RED = "#f28e2b"
MISSING = "#d9d9d9"
INK = "#333333"


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(EXPORT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(EXPORT / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def select_best_runs(measurements: pd.DataFrame, min_items: int = MIN_ITEMS) -> pd.DataFrame:
    runs = (
        measurements.groupby(["model", "run_id"], sort=False)
        .agg(started=("timestamp", "min"), n_items=("custom_id", "nunique"))
        .reset_index()
    )
    best = (
        runs.sort_values(
            ["model", "n_items", "started", "run_id"],
            ascending=[True, False, True, True],
        )
        .groupby("model", as_index=False)
        .first()
    )
    return (
        best.loc[best["n_items"].ge(min_items)]
        .sort_values(["n_items", "model"], ascending=[False, True])
        .reset_index(drop=True)
    )


def order_rows(
    labels: pd.Series, predictions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = pd.DataFrame(index=labels.index)
    stats["expected"] = labels.astype(bool)
    stats["coverage"] = predictions.notna().sum(axis=1)
    stats["vote_share"] = predictions.astype(float).mean(axis=1, skipna=True)
    stats["mixed"] = (
        stats["coverage"].gt(0)
        & stats["vote_share"].gt(0.0)
        & stats["vote_share"].lt(1.0)
    )
    stats["disagreement"] = np.where(
        stats["coverage"].gt(0),
        stats["vote_share"] * (1.0 - stats["vote_share"]),
        np.nan,
    )
    stats["sort_group"] = np.select(
        [
            stats["coverage"].eq(0),
            stats["vote_share"].eq(0.0),
            stats["mixed"],
            stats["vote_share"].eq(1.0),
        ],
        [3, 0, 1, 2],
        default=3,
    )
    ordered_ids = (
        stats.assign(
            vote_share_sort=stats["vote_share"].fillna(2.0),
            disagreement_sort=-stats["disagreement"].fillna(-1.0),
            expected_sort=stats["expected"].astype(int),
        )
        .reset_index(names="custom_id")
        .sort_values(
            [
                "sort_group",
                "vote_share_sort",
                "disagreement_sort",
                "expected_sort",
                "custom_id",
            ],
            kind="stable",
        )
        ["custom_id"]
        .tolist()
    )
    ordered_stats = stats.loc[ordered_ids].copy()
    ordered_stats.insert(0, "row_index", np.arange(len(ordered_stats)))
    ordered_predictions = predictions.loc[ordered_ids].copy()
    return ordered_stats, ordered_predictions


def build_outputs(root: Path, min_items: int = MIN_ITEMS) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    benchmark = load_benchmark(root).set_index("custom_id")
    measurements, conflicts = load_measurements(root)
    selected = select_best_runs(measurements, min_items=min_items)
    if selected.empty:
        raise RuntimeError(f"no model runs reached the minimum coverage threshold of {min_items}")

    subset = measurements.merge(selected[["model", "run_id"]], on=["model", "run_id"], how="inner")
    model_order = (
        subset.groupby("model")["predicted"]
        .mean()
        .sort_values(kind="stable")
        .index.tolist()
    )
    wide = (
        subset.pivot(index="custom_id", columns="model", values="predicted")
        .reindex(index=benchmark.index, columns=model_order)
        .astype("boolean")
    )
    ordered_stats, ordered_predictions = order_rows(benchmark["label"], wide)
    mixed_positions = np.flatnonzero(ordered_stats["mixed"].to_numpy())
    mixed_start = int(mixed_positions[0]) if mixed_positions.size else None
    mixed_end = int(mixed_positions[-1]) if mixed_positions.size else None

    rates = subset.groupby("model")["predicted"].mean()
    selected_lookup = selected.set_index("model")
    selected_models = []
    for model in model_order:
        row = selected_lookup.loc[model]
        selected_models.append(
            {
                "model": model,
                "run_id": row["run_id"],
                "items": int(row["n_items"]),
                "started": row["started"].isoformat(),
                "positive_rate": float(rates.loc[model]),
            }
        )

    source = ordered_stats.reset_index(names="custom_id").rename(columns={"expected": "label"})
    source["label"] = source["label"].astype(bool)
    source["fully_observed"] = source["coverage"].eq(len(model_order))
    source = source.join(ordered_predictions.reset_index(drop=True))

    summary = {
        "figure_stem": FIGURE_STEM,
        "min_items_threshold": int(min_items),
        "benchmark_rows": int(len(source)),
        "selected_model_count": int(len(model_order)),
        "observed_rows": int(source["coverage"].gt(0).sum()),
        "fully_observed_rows": int(source["fully_observed"].sum()),
        "partially_observed_rows": int(
            source["coverage"].gt(0).sum() - source["fully_observed"].sum()
        ),
        "fully_missing_rows": int(source["coverage"].eq(0).sum()),
        "mixed_rows": int(source["mixed"].sum()),
        "band_start_row": mixed_start,
        "band_end_row": mixed_end,
        "excluded_conflicting_runs": int(len(conflicts)),
        "selected_models": selected_models,
    }
    return source, ordered_predictions, summary


def render_figure(
    source: pd.DataFrame, ordered_predictions: pd.DataFrame, summary: dict[str, object]
) -> plt.Figure:
    cmap = ListedColormap([BLUE, RED])
    cmap.set_bad(MISSING)
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    data = ordered_predictions.astype(float).to_numpy()
    truth = source["label"].astype(float).to_numpy()[:, None]
    n_models = data.shape[1]

    fig = plt.figure(figsize=(7.0, 9.4))
    grid = fig.add_gridspec(1, 2, width_ratios=[0.38, 5.0], wspace=0.05)
    ax_truth = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[0, 1], sharey=ax_truth)

    ax_truth.imshow(truth, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)

    ax_truth.set_xticks([0], ["truth"])
    ax_truth.tick_params(axis="x", labelrotation=90)
    ax.set_xticks(np.arange(n_models), ordered_predictions.columns, rotation=35, ha="right")
    ticks = [0, 299, 599, 899, 1198]
    ax_truth.set_yticks(ticks, [str(value + 1) for value in ticks])
    ax.set_yticks(ticks)
    ax.set_yticklabels([])
    ax_truth.set_ylabel("Benchmark posts (ordered)")
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("Real-runs disagreement matrix on the frozen 1199-item benchmark", fontsize=10, pad=16)

    for axis in (ax_truth, ax):
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=0)

    band_start = summary["band_start_row"]
    band_end = summary["band_end_row"]
    if band_start is not None and band_end is not None:
        y_center = (band_start + band_end) / 2
        box = Rectangle(
            xy=(-0.48, band_start - 0.48),
            width=n_models - 0.04,
            height=band_end - band_start + 0.96,
            fill=False,
            edgecolor=INK,
            linewidth=1.4,
        )
        ax.add_patch(box)
        ax.annotate(
            "disagreement band",
            xy=(n_models - 0.55, y_center),
            xytext=(18, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK,
            arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.8},
            annotation_clip=False,
        )

    handles = [
        Patch(facecolor=BLUE, label="predicted not relevant"),
        Patch(facecolor=RED, label="predicted relevant"),
        Patch(facecolor=MISSING, label="missing prediction"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.56, 0.98))
    fig.text(
        0.5,
        0.03,
        (
            f"{summary['selected_model_count']} models; "
            f"{summary['fully_observed_rows']} complete rows; "
            f"{summary['partially_observed_rows']} partial rows; "
            f"{summary['fully_missing_rows']} fully missing rows"
        ),
        ha="center",
        fontsize=8,
        color=INK,
    )
    fig.subplots_adjust(top=0.9, bottom=0.17, left=0.12, right=0.87)
    return fig


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
    EXPORT.mkdir(parents=True, exist_ok=True)
    root = find_root(ROOT)
    source, ordered_predictions, summary = build_outputs(root)
    source.to_csv(EXPORT / SOURCE_NAME, index=False)
    (EXPORT / SUMMARY_NAME).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fig = render_figure(source, ordered_predictions, summary)
    save(fig, FIGURE_STEM)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
