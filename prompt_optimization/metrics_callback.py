"""GEPA callbacks: step metrics CSV + held-out test tracking during search."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class MetricsHistoryCallback:
    FIELDNAMES = [
        "timestamp_utc",
        "event",
        "iteration",
        "candidate_idx",
        "val_accuracy",
        "test_accuracy",
        "val_examples_evaluated",
        "val_total_size",
        "is_best_program",
        "proposal_accepted",
        "best_val_accuracy_so_far",
        "metric_calls_used",
        "notes",
    ]

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.best_val_accuracy = 0.0
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.FIELDNAMES).writeheader()

    def _append(self, row: Dict[str, Any]) -> None:
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow({k: row.get(k, "") for k in self.FIELDNAMES})

    def on_optimization_start(self, event) -> None:
        self._append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "event": "optimization_start",
                "iteration": 0,
                "val_total_size": event.get("valset_size"),
                "notes": f"trainset_size={event.get('trainset_size')}",
            }
        )

    def on_valset_evaluated(self, event) -> None:
        val_acc = float(event.get("average_score") or 0.0)
        if val_acc > self.best_val_accuracy:
            self.best_val_accuracy = val_acc
        self._append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "event": "valset_evaluated",
                "iteration": event.get("iteration"),
                "candidate_idx": event.get("candidate_idx"),
                "val_accuracy": f"{val_acc:.6f}",
                "val_examples_evaluated": event.get("num_examples_evaluated"),
                "val_total_size": event.get("total_valset_size"),
                "is_best_program": event.get("is_best_program"),
                "best_val_accuracy_so_far": f"{self.best_val_accuracy:.6f}",
                "notes": "accepted_candidate" if event.get("is_best_program") else "",
            }
        )

    def on_iteration_end(self, event) -> None:
        self._append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "event": "iteration_end",
                "iteration": event.get("iteration"),
                "proposal_accepted": event.get("proposal_accepted"),
                "best_val_accuracy_so_far": f"{self.best_val_accuracy:.6f}",
            }
        )

    def on_optimization_end(self, event) -> None:
        self._append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "event": "optimization_end",
                "iteration": event.get("total_iterations"),
                "candidate_idx": event.get("best_candidate_idx"),
                "metric_calls_used": event.get("total_metric_calls"),
                "best_val_accuracy_so_far": f"{self.best_val_accuracy:.6f}",
            }
        )

    def log_manual(self, event: str, **fields) -> None:
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
        }
        row.update(fields)
        self._append(row)


class TestEvalCallback:
    """Evaluate each fully-scored candidate on held-out test (logging only during search)."""

    def __init__(
        self,
        testset: list,
        accuracy_fn: Callable[[dict, list], float],
        metrics: MetricsHistoryCallback,
        val_margin: float = 0.02,
    ):
        self.testset = testset
        self.accuracy_fn = accuracy_fn
        self.metrics = metrics
        self.val_margin = val_margin
        self.test_scores: Dict[int, float] = {}

    def on_valset_evaluated(self, event) -> None:
        evaluated = event.get("num_examples_evaluated")
        total = event.get("total_valset_size")
        if not evaluated or not total or evaluated != total:
            return
        idx = event.get("candidate_idx")
        candidate = event.get("candidate")
        if idx is None or not candidate:
            return
        if idx in self.test_scores:
            return

        test_acc = self.accuracy_fn(candidate, self.testset)
        val_acc = float(event.get("average_score") or 0.0)
        self.test_scores[idx] = test_acc
        self.metrics.log_manual(
            "test_eval_during_search",
            iteration=event.get("iteration"),
            candidate_idx=idx,
            val_accuracy=f"{val_acc:.6f}",
            test_accuracy=f"{test_acc:.6f}",
            notes="held_out_test_not_used_for_gepa_selection",
        )

    def pick_conservative_best(self, result, val_scores: List[float]) -> int:
        """Among candidates within val_margin of best val, pick highest test score."""
        if not val_scores:
            return result.best_idx
        best_val = max(val_scores)
        threshold = best_val - self.val_margin
        near_best = [i for i, s in enumerate(val_scores) if s >= threshold]
        if not near_best:
            return result.best_idx
        scored = [i for i in near_best if i in self.test_scores]
        if not scored:
            return result.best_idx
        return max(scored, key=lambda i: self.test_scores[i])


def save_candidates_table(result, out_path: Path, test_scores: Optional[Dict[int, float]] = None) -> None:
    rows: List[Dict[str, Any]] = []
    for idx, score in enumerate(result.val_aggregate_scores):
        rows.append(
            {
                "candidate_idx": idx,
                "val_accuracy": score,
                "test_accuracy": test_scores.get(idx) if test_scores else "",
                "is_best_val": idx == result.best_idx,
                "parent_ids": result.parents[idx] if idx < len(result.parents) else [],
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["candidate_idx", "val_accuracy", "test_accuracy", "is_best_val", "parent_ids"],
        )
        writer.writeheader()
        writer.writerows(rows)
