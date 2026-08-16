"""Rebuild the frozen, canonical thesis evidence from repository-local inputs.

This module performs no API calls.  All generated artifacts are written below
``notebooks/canonical_evidence_export``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import binom


ANALYSIS_VERSION = "canonical-evidence-v1.0.0"
WINDOW_START = pd.Timestamp("2026-03-15T00:00:00Z")
WINDOW_END = pd.Timestamp("2026-07-06T23:59:59.999999Z")
MAY22 = pd.Timestamp("2026-05-22", tz="UTC")
ALPHA = 0.05
MIN_PAIRS = 250
PANEL_MIN = 250
PANEL_MAX = 300
PANEL_N = 300
BENCHMARK_N = 1199
OPERATIONAL_FLOOR = 0.05
REFERENCE_DEGRADATION_LO = 0.05
REFERENCE_DEGRADATION_HI = 0.08
C_FP = 0.155
C_FN = 0.031
R_GRID = np.geomspace(0.125, 64.0, 90)
LAMBDA_GRID = np.geomspace(0.02, 40.0, 90)
INPUT_PATHS = (
    "streamlit_app/langfuse_traces.csv",
    "streamlit_app/langfuse_scores.csv",
    "prompt_optimization/splits/train.json",
    "prompt_optimization/splits/val.json",
    "prompt_optimization/splits/test.json",
    "prompt_optimization/splits/split_manifest.json",
)
ERROR_TYPES = {
    "true_positive": (True, True),
    "false_positive": (False, True),
    "true_negative": (False, False),
    "false_negative": (True, False),
}


@dataclass(frozen=True)
class BuildResult:
    export_dir: str
    panel_hash: str
    comparisons: int
    models: int
    raw_significant: int
    holm_significant: int
    operational_alerts: int
    cost_common_n: int
    cost_winner: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(values)).encode()
    return hashlib.sha256(payload).hexdigest()


def find_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / INPUT_PATHS[0]).is_file():
            return candidate
    raise FileNotFoundError("could not locate benchmark_eval repository root")


def load_benchmark(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        with (root / "prompt_optimization" / "splits" / f"{split}.json").open() as handle:
            items = json.load(handle)
        for item in items:
            context = item["additional_context"]
            rows.append(
                {
                    "custom_id": str(context["custom_id"]),
                    "split": split,
                    "label": item["answer"] == "TRUE",
                }
            )
    benchmark = pd.DataFrame(rows)
    assert len(benchmark) == BENCHMARK_N
    assert benchmark["custom_id"].is_unique
    return benchmark


def _score_table(scores: pd.DataFrame, name: str, columns: list[str]) -> pd.DataFrame:
    table = scores.loc[scores["name"].eq(name), ["traceId", *columns]].copy()
    duplicated = table["traceId"].duplicated(keep=False)
    if duplicated.any():
        conflicts = table.loc[duplicated].groupby("traceId", dropna=False)[columns].nunique(dropna=False)
        if (conflicts > 1).any(axis=None):
            raise ValueError(f"conflicting duplicate {name!r} scores")
        table = table.drop_duplicates("traceId")
    return table


def load_measurements(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    traces = pd.read_csv(root / INPUT_PATHS[0], low_memory=False, dtype={"metadata.custom_id": str})
    scores = pd.read_csv(root / INPUT_PATHS[1], low_memory=False)
    traces["timestamp"] = pd.to_datetime(traces["timestamp"], format="mixed", utc=True, errors="coerce")

    accuracy = _score_table(scores, "accuracy", ["value"]).rename(columns={"value": "accuracy"})
    error = _score_table(scores, "error_type", ["stringValue"]).rename(
        columns={"stringValue": "error_type"}
    )
    cost = _score_table(scores, "cost_usd", ["value", "comment"]).rename(
        columns={"value": "cost_usd", "comment": "cost_detail"}
    )

    evaluations = (
        traces.loc[traces["tags"].fillna("").str.contains("batch_evaluation", regex=False)]
        .rename(
            columns={
                "id": "trace_id",
                "metadata.model": "model",
                "metadata.custom_id": "custom_id",
                "metadata.run_id": "run_id",
            }
        )[["trace_id", "timestamp", "model", "custom_id", "run_id"]]
        .merge(accuracy, left_on="trace_id", right_on="traceId", how="left")
        .drop(columns="traceId")
        .merge(error, left_on="trace_id", right_on="traceId", how="left")
        .drop(columns="traceId")
        .merge(cost, left_on="trace_id", right_on="traceId", how="left")
        .drop(columns="traceId")
    )
    required = ["timestamp", "model", "custom_id", "run_id", "accuracy", "error_type"]
    complete = evaluations[required].notna().all(axis=1)
    evaluations = evaluations.loc[complete & evaluations["error_type"].isin(ERROR_TYPES)].copy()
    evaluations["custom_id"] = evaluations["custom_id"].astype(str)
    evaluations["accuracy"] = pd.to_numeric(evaluations["accuracy"], errors="raise").astype(int)
    evaluations["cost_usd"] = pd.to_numeric(evaluations["cost_usd"], errors="coerce")
    evaluations[["expected", "predicted"]] = pd.DataFrame(
        evaluations["error_type"].map(ERROR_TYPES).tolist(), index=evaluations.index
    )
    evaluations["correct"] = evaluations["expected"].eq(evaluations["predicted"]).astype(int)
    if not evaluations["correct"].eq(evaluations["accuracy"]).all():
        raise ValueError("accuracy and error_type scores disagree")

    signature_columns = ["accuracy", "error_type", "cost_usd", "cost_detail"]
    signature = evaluations[signature_columns].fillna("<NA>").astype(str).agg("|".join, axis=1)
    evaluations["signature"] = signature
    keys = ["model", "run_id", "custom_id"]
    grouped = evaluations.groupby(keys, sort=False)["signature"]
    evaluations["signature_count"] = grouped.transform("nunique")
    conflicting_runs = (
        evaluations.loc[evaluations["signature_count"].gt(1), ["model", "run_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    conflict_index = pd.MultiIndex.from_frame(conflicting_runs)
    row_index = pd.MultiIndex.from_frame(evaluations[["model", "run_id"]])
    measurements = (
        evaluations.loc[~row_index.isin(conflict_index)]
        .sort_values(["timestamp", "trace_id"])
        .drop_duplicates(keys, keep="first")
        .drop(columns="signature_count")
        .reset_index(drop=True)
    )
    assert not measurements.duplicated(keys).any()
    return measurements, conflicting_runs


def reconstruct_panel(
    measurements: pd.DataFrame, benchmark_ids: set[str]
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    in_window = measurements["timestamp"].between(WINDOW_START, WINDOW_END)
    scheduled = measurements["run_id"].astype(str).str.startswith("gha-")
    candidate_data = measurements.loc[in_window & scheduled].copy()
    runs = (
        candidate_data.groupby(["model", "run_id"], sort=False)
        .agg(started=("timestamp", "min"), n_items=("custom_id", "nunique"))
        .reset_index()
    )
    candidates = runs.loc[runs["n_items"].between(PANEL_MIN, PANEL_MAX)].copy()
    candidate_keys = pd.MultiIndex.from_frame(candidates[["model", "run_id"]])
    candidate_rows = candidate_data.loc[
        pd.MultiIndex.from_frame(candidate_data[["model", "run_id"]]).isin(candidate_keys)
    ]
    panel_ids = sorted(candidate_rows["custom_id"].unique())
    if len(panel_ids) != PANEL_N:
        raise AssertionError(f"reconstructed panel has {len(panel_ids)} IDs, expected {PANEL_N}")
    if not set(panel_ids) <= benchmark_ids:
        raise AssertionError("monitoring panel is not a subset of the frozen benchmark")

    panel_set = set(panel_ids)
    membership = candidate_rows.groupby(["model", "run_id"])["custom_id"].agg(
        lambda values: set(values) <= panel_set
    )
    eligible = candidates.merge(
        membership.rename("panel_only").reset_index(), on=["model", "run_id"], how="left"
    )
    eligible = eligible.loc[eligible["panel_only"]].sort_values(["model", "started"]).reset_index(drop=True)
    eligible_keys = pd.MultiIndex.from_frame(eligible[["model", "run_id"]])
    panel_data = candidate_data.loc[
        pd.MultiIndex.from_frame(candidate_data[["model", "run_id"]]).isin(eligible_keys)
    ].copy()
    return panel_ids, eligible, panel_data


def exact_mcnemar_p(b: int, c: int) -> float:
    discordant = b + c
    return float(binom.sf(b - 1, discordant, 0.5)) if discordant else 1.0


def holm_adjust(pvalues: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvalues), dtype=float)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def build_comparisons(eligible: pd.DataFrame, panel_data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, model_runs in eligible.groupby("model", sort=True):
        model_runs = model_runs.sort_values(["started", "run_id"])
        if len(model_runs) < 2:
            continue
        baseline = model_runs.iloc[0]
        left = panel_data.loc[
            panel_data["model"].eq(model) & panel_data["run_id"].eq(baseline["run_id"]),
            ["custom_id", "correct", "expected"],
        ].set_index("custom_id")
        for later in model_runs.iloc[1:].itertuples(index=False):
            right = panel_data.loc[
                panel_data["model"].eq(model) & panel_data["run_id"].eq(later.run_id),
                ["custom_id", "correct", "expected"],
            ].set_index("custom_id")
            paired = left.join(right, how="inner", lsuffix="_a", rsuffix="_b")
            if len(paired) < MIN_PAIRS:
                continue
            if not paired["expected_a"].eq(paired["expected_b"]).all():
                raise AssertionError("gold labels differ across paired runs")
            a_correct = paired["correct_a"]
            b_correct = paired["correct_b"]
            b = int((a_correct.eq(1) & b_correct.eq(0)).sum())
            c = int((a_correct.eq(0) & b_correct.eq(1)).sum())
            rows.append(
                {
                    "model": model,
                    "baseline_run_id": baseline["run_id"],
                    "later_run_id": later.run_id,
                    "baseline_started": baseline["started"],
                    "later_started": later.started,
                    "n_pairs": len(paired),
                    "acc_baseline": float(a_correct.mean()),
                    "acc_later": float(b_correct.mean()),
                    "b_degrade": b,
                    "c_improve": c,
                    "discordance": (b + c) / len(paired),
                    "delta": (c - b) / len(paired),
                    "p_down": exact_mcnemar_p(b, c),
                }
            )
    comparisons = pd.DataFrame(rows)
    if comparisons.empty:
        raise AssertionError("no eligible baseline-to-later comparisons")
    comparisons["p_down_holm"] = holm_adjust(comparisons["p_down"])
    comparisons["holm_reject"] = comparisons["p_down_holm"].le(ALPHA)
    comparisons["operational_floor_met"] = comparisons["delta"].le(-OPERATIONAL_FLOOR)
    comparisons["operational_alert"] = (
        comparisons["holm_reject"] & comparisons["operational_floor_met"]
    )
    return comparisons.sort_values(["model", "later_started"]).reset_index(drop=True)


@lru_cache(maxsize=None)
def _mcnemar_rejection_thresholds(n: int, alpha: float) -> np.ndarray:
    thresholds = np.full(n + 1, n + 1, dtype=int)
    for discordant in range(n + 1):
        # Search from the upper tail; n <= 300 in the canonical analysis and
        # this cached calculation runs only once for each alpha.
        for b in range(discordant + 1):
            if binom.sf(b - 1, discordant, 0.5) <= alpha:
                thresholds[discordant] = b
                break
    thresholds.setflags(write=False)
    return thresholds


def exact_unconditional_power(n: int, rho: float, g: float, alpha: float) -> float:
    """Exact power of the one-sided conditional McNemar test.

    ``rho=P(b)+P(c)`` is discordance and ``g=P(b)-P(c)`` is the degradation
    effect.  The multinomial sampling distribution is integrated over every
    possible discordant count instead of conditioning on an expected count.
    """
    if not (0.0 <= g <= rho <= 1.0):
        raise ValueError("exact McNemar power requires 0 <= g <= rho <= 1")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly between zero and one")
    if rho == 0:
        return 0.0
    q = (rho + g) / (2.0 * rho)
    d = np.arange(n + 1)
    thresholds = _mcnemar_rejection_thresholds(n, alpha)
    reject_given_d = binom.sf(thresholds - 1, d, q)
    return float(np.dot(binom.pmf(d, n, rho), reject_given_d))


def exact_sensitivity(
    n: int, rho: float, alpha: float, target_power: float = 0.8, tolerance: float = 1e-10
) -> float:
    if exact_unconditional_power(n, rho, rho, alpha) < target_power:
        return float("nan")
    low, high = 0.0, rho
    while high - low > tolerance:
        middle = (low + high) / 2
        if exact_unconditional_power(n, rho, middle, alpha) >= target_power:
            high = middle
        else:
            low = middle
    if high > rho + tolerance:
        raise AssertionError("computed sensitivity violates g <= rho")
    return high


def build_power_table(number_of_tests: int, comparisons: pd.DataFrame) -> pd.DataFrame:
    observed = comparisons.groupby("model")["discordance"].median().sort_index()
    rows: list[dict[str, object]] = []
    family_alpha = ALPHA / number_of_tests
    design_rhos = [0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
    cases = [
        ("design_grid", f"rho={rho:.2f}", rho) for rho in design_rhos
    ] + [
        ("observed_model_median", model, float(rho)) for model, rho in observed.items()
    ]
    for scope, model, rho in cases:
        nominal = exact_sensitivity(PANEL_N, float(rho), ALPHA)
        family = exact_sensitivity(PANEL_N, float(rho), family_alpha)
        floor_is_admissible = OPERATIONAL_FLOOR <= rho
        reference_lo_admissible = REFERENCE_DEGRADATION_LO <= rho
        reference_hi_admissible = REFERENCE_DEGRADATION_HI <= rho
        rows.append(
            {
                "scope": scope,
                "model": model,
                "n": PANEL_N,
                "rho": rho,
                "target_power": 0.8,
                "nominal_alpha": ALPHA,
                "nominal_g_80": nominal,
                "nominal_80_attainable": not math.isnan(nominal),
                "family_alpha": family_alpha,
                "family_g_80": family,
                "family_80_attainable": not math.isnan(family),
                "floor_admissible": floor_is_admissible,
                "power_5pp_admissible": reference_lo_admissible,
                "power_8pp_admissible": reference_hi_admissible,
                "nominal_power_at_floor": (
                    exact_unconditional_power(PANEL_N, float(rho), OPERATIONAL_FLOOR, ALPHA)
                    if floor_is_admissible
                    else float("nan")
                ),
                "family_power_at_floor": (
                    exact_unconditional_power(
                        PANEL_N, float(rho), OPERATIONAL_FLOOR, family_alpha
                    )
                    if floor_is_admissible
                    else float("nan")
                ),
                "nominal_power_at_5pp": (
                    exact_unconditional_power(PANEL_N, float(rho), REFERENCE_DEGRADATION_LO, ALPHA)
                    if reference_lo_admissible
                    else float("nan")
                ),
                "family_power_at_5pp": (
                    exact_unconditional_power(
                        PANEL_N, float(rho), REFERENCE_DEGRADATION_LO, family_alpha
                    )
                    if reference_lo_admissible
                    else float("nan")
                ),
                "nominal_power_at_8pp": (
                    exact_unconditional_power(PANEL_N, float(rho), REFERENCE_DEGRADATION_HI, ALPHA)
                    if reference_hi_admissible
                    else float("nan")
                ),
                "family_power_at_8pp": (
                    exact_unconditional_power(
                        PANEL_N, float(rho), REFERENCE_DEGRADATION_HI, family_alpha
                    )
                    if reference_hi_admissible
                    else float("nan")
                ),
            }
        )
    table = pd.DataFrame(rows)
    for column in ("nominal_g_80", "family_g_80"):
        valid = table[column].notna()
        if not table.loc[valid, column].le(table.loc[valid, "rho"] + 1e-9).all():
            raise AssertionError(f"{column} violates g <= rho")
    return table


def build_cost_evidence(
    measurements: pd.DataFrame, benchmark_ids: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    on_day = measurements["timestamp"].dt.normalize().eq(MAY22)
    may = measurements.loc[on_day & measurements["custom_id"].isin(benchmark_ids)].copy()
    runs = (
        may.groupby(["model", "run_id"], sort=False)
        .agg(n_items=("custom_id", "nunique"), started=("timestamp", "min"))
        .reset_index()
    )
    full = runs.loc[runs["n_items"].gt(1000)].copy()
    if full.empty:
        raise AssertionError("no 22 May full-benchmark runs found")
    # There should be one full run per model.  If a repeated ingestion has a
    # different run id, refusing it is safer than choosing by timestamp.
    counts = full.groupby("model")["run_id"].nunique()
    if counts.gt(1).any():
        raise AssertionError(f"multiple 22 May full runs for models: {counts[counts.gt(1)].to_dict()}")
    full_keys = pd.MultiIndex.from_frame(full[["model", "run_id"]])
    data = may.loc[pd.MultiIndex.from_frame(may[["model", "run_id"]]).isin(full_keys)].copy()
    complete_cost = data["cost_usd"].notna()
    data = data.loc[complete_cost]
    ids_by_model = data.groupby("model")["custom_id"].agg(set)
    common_ids = set.intersection(*ids_by_model.tolist())
    if not common_ids:
        raise AssertionError("22 May full runs have no complete-case ID intersection")
    common = data.loc[data["custom_id"].isin(common_ids)].copy()
    expected_by_id = common.groupby("custom_id")["expected"].nunique()
    if expected_by_id.gt(1).any():
        raise AssertionError("gold labels disagree on the cost complete-case intersection")

    common = common.assign(
        TP=common["expected"] & common["predicted"],
        FP=~common["expected"] & common["predicted"],
        TN=~common["expected"] & ~common["predicted"],
        FN=common["expected"] & ~common["predicted"],
    )
    table = (
        common.groupby(["model", "run_id"], sort=True)
        .agg(
            n=("custom_id", "nunique"),
            TP=("TP", "sum"),
            FP=("FP", "sum"),
            TN=("TN", "sum"),
            FN=("FN", "sum"),
            mean_inference_cost=("cost_usd", "mean"),
        )
        .reset_index()
    )
    if not table["n"].eq(len(common_ids)).all():
        raise AssertionError("cost evidence is not on one common complete-case intersection")
    table["accuracy"] = (table["TP"] + table["TN"]) / table["n"]
    table["fp_rate_all"] = table["FP"] / table["n"]
    table["fn_rate_all"] = table["FN"] / table["n"]
    table["R"] = C_FN / C_FP
    table["lambda"] = C_FP + C_FN
    table["expected_cost"] = (
        table["mean_inference_cost"]
        + C_FP * table["fp_rate_all"]
        + C_FN * table["fn_rate_all"]
    )
    table = table.sort_values("expected_cost").reset_index(drop=True)

    grid_rows: list[dict[str, object]] = []
    for lam in LAMBDA_GRID:
        for ratio in R_GRID:
            c_fp = lam / (1.0 + ratio)
            c_fn = lam * ratio / (1.0 + ratio)
            costs = (
                table["mean_inference_cost"]
                + c_fp * table["fp_rate_all"]
                + c_fn * table["fn_rate_all"]
            )
            winner = table.loc[costs.idxmin(), "model"]
            for index, model in enumerate(table["model"]):
                grid_rows.append(
                    {
                        "R": ratio,
                        "lambda": lam,
                        "C_FP": c_fp,
                        "C_FN": c_fn,
                        "model": model,
                        "expected_cost": costs.iloc[index],
                        "winner": model == winner,
                    }
                )
    grid = pd.DataFrame(grid_rows)
    metadata = {
        "day": str(MAY22.date()),
        "common_n": len(common_ids),
        "common_ids_hash": sha256_lines(common_ids),
        "models": len(table),
        "R": C_FN / C_FP,
        "lambda": C_FP + C_FN,
        "winner": table.iloc[0]["model"],
        "winner_expected_cost": float(table.iloc[0]["expected_cost"]),
    }
    return table, grid, metadata


def inspect_gepa(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted((root / "prompt_optimization" / "runs").glob("*/*/summary.json")):
        try:
            with path.open() as handle:
                summary = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            rows.append({"path": str(path.relative_to(root)), "status": f"unreadable: {error}"})
            continue
        rows.append(
            {
                "model": path.parents[1].name,
                "run_id": path.parent.name,
                "path": str(path.relative_to(root)),
                "status": summary.get("status", "present"),
                "best_val": summary.get("best_val_accuracy", summary.get("best_score")),
                "test": summary.get("test_accuracy"),
                "interpretation": "exploratory_existing_artifact",
            }
        )
    return pd.DataFrame(rows)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def rebuild(root: Path | None = None, export_dir: Path | None = None) -> BuildResult:
    root = find_root(root)
    export_dir = export_dir or root / "notebooks" / "canonical_evidence_export"
    staging = export_dir.with_name(f".{export_dir.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    input_manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "window": {"start": WINDOW_START.isoformat(), "end": WINDOW_END.isoformat()},
        "inputs": [
            {
                "path": relative,
                "sha256": sha256_file(root / relative),
                "bytes": (root / relative).stat().st_size,
            }
            for relative in INPUT_PATHS
        ],
    }
    _write_json(staging / "input_manifest.json", input_manifest)

    benchmark = load_benchmark(root)
    measurements, conflicts = load_measurements(root)
    benchmark_ids = set(benchmark["custom_id"])
    panel_ids, eligible, panel_data = reconstruct_panel(measurements, benchmark_ids)
    panel_hash = sha256_lines(panel_ids)
    (staging / "panel_ids.txt").write_text("".join(f"{item}\n" for item in panel_ids))
    eligible.to_csv(staging / "eligible_monitoring_runs.csv", index=False)
    conflicts.to_csv(staging / "excluded_conflicting_runs.csv", index=False)

    comparisons = build_comparisons(eligible, panel_data)
    comparisons.to_csv(staging / "baseline_to_later_mcnemar.csv", index=False)
    power = build_power_table(len(comparisons), comparisons)
    power.to_csv(staging / "exact_unconditional_power_sensitivity.csv", index=False)

    cost, cost_grid, cost_metadata = build_cost_evidence(measurements, benchmark_ids)
    cost.to_csv(staging / "deployment_cost_may22.csv", index=False)
    cost_grid.to_csv(staging / "deployment_cost_map_source.csv", index=False)
    gepa = inspect_gepa(root)
    gepa.to_csv(staging / "gepa_existing_artifacts.csv", index=False)

    trigger = comparisons.loc[comparisons["operational_alert"]]
    power_reference = power.loc[
        power["scope"].eq("design_grid") & power["rho"].eq(0.12)
    ].iloc[0]
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "window": {"start": WINDOW_START.isoformat(), "end": WINDOW_END.isoformat()},
        "benchmark": {
            "n": len(benchmark),
            "positives": int(benchmark["label"].sum()),
            "prevalence": float(benchmark["label"].mean()),
        },
        "panel": {
            "n": len(panel_ids),
            "sha256": panel_hash,
            "positives": int(benchmark.set_index("custom_id").loc[panel_ids, "label"].sum()),
            "subset_of_benchmark": set(panel_ids) <= benchmark_ids,
        },
        "monitoring": {
            "eligible_runs": len(eligible),
            "models": int(comparisons["model"].nunique()),
            "comparisons": len(comparisons),
            "raw_significant": int(comparisons["p_down"].le(ALPHA).sum()),
            "holm_significant": int(comparisons["holm_reject"].sum()),
            "operational_alerts": int(comparisons["operational_alert"].sum()),
            "median_discordance": float(comparisons["discordance"].median()),
            "operational_floor": OPERATIONAL_FLOOR,
            "trigger_rows": trigger.to_dict("records"),
        },
        "power": {
            "method": "exact unconditional multinomial power for conditional exact McNemar rejection",
            "nominal_alpha": ALPHA,
            "family_conservative_alpha": ALPHA / len(comparisons),
            "target_power": 0.8,
            "constraint": "0 <= g <= rho",
            "reference_rho": 0.12,
            "nominal_g_80_at_reference_rho": float(power_reference["nominal_g_80"]),
            "family_g_80_at_reference_rho": float(power_reference["family_g_80"]),
        },
        "cost": cost_metadata,
        "gepa": {
            "status": "exploratory_existing_artifacts_only",
            "artifacts_inspected": len(gepa),
            "paid_api_calls": 0,
        },
    }
    _write_json(staging / "canonical_summary.json", summary)

    artifact_rows = []
    for path in sorted(staging.iterdir()):
        if path.name == "artifact_manifest.csv":
            continue
        artifact_rows.append(
            {
                "analysis_version": ANALYSIS_VERSION,
                "artifact": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    pd.DataFrame(artifact_rows).to_csv(staging / "artifact_manifest.csv", index=False)

    if export_dir.exists():
        shutil.rmtree(export_dir)
    staging.rename(export_dir)

    old_export = root / "notebooks" / "thesis_evidence_export"
    if old_export.exists():
        (old_export / "INCOMPATIBLE_WITH_CANONICAL_REBUILD.md").write_text(
            "# Retired evidence export\n\n"
            "These artifacts predate the strict canonical rebuild and must not be mixed with it. "
            "Use `../canonical_evidence_export/`, generated by `canonical_evidence.py` "
            f"({ANALYSIS_VERSION}).\n"
        )

    return BuildResult(
        export_dir=str(export_dir.relative_to(root)),
        panel_hash=panel_hash,
        comparisons=len(comparisons),
        models=int(comparisons["model"].nunique()),
        raw_significant=int(comparisons["p_down"].le(ALPHA).sum()),
        holm_significant=int(comparisons["holm_reject"].sum()),
        operational_alerts=int(comparisons["operational_alert"].sum()),
        cost_common_n=int(cost_metadata["common_n"]),
        cost_winner=str(cost_metadata["winner"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--export-dir", type=Path, default=None)
    args = parser.parse_args()
    result = rebuild(args.root, args.export_dir)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
