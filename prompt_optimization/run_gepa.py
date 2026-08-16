#!/usr/bin/env python3
"""Run GEPA prompt optimization for campaign relevance on train/val/test splits."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
PO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PO_ROOT))

import gepa  # noqa: E402

from campaign_adapter import CampaignRelevanceAdapter  # noqa: E402
from data_utils import (  # noqa: E402
    DEFAULT_DATASET,
    VAL_MARGIN_FOR_SELECTION,
    extract_seed_system_prompt,
    load_splits,
    prepare_splits,
)
from metrics_callback import (  # noqa: E402
    MetricsHistoryCallback,
    TestEvalCallback,
    save_candidates_table,
)


def _accuracy(adapter: CampaignRelevanceAdapter, candidate: dict, dataset: list) -> float:
    if not dataset:
        return 0.0
    batch_size = 32
    scores = []
    for i in range(0, len(dataset), batch_size):
        chunk = dataset[i : i + batch_size]
        result = adapter.evaluate(chunk, candidate, capture_traces=False)
        scores.extend(result.scores)
    return sum(scores) / len(scores)


def run_for_model(
    model: str,
    dataset_name: str,
    split_dir: Path,
    max_metric_calls: int,
    reflection_lm: str,
    max_workers: int,
    seed: int,
    skip_perfect_score: bool,
    val_margin: float,
    seed_prompt_path: Path | None = None,
) -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    run_dir = PO_ROOT / "runs" / model / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = load_splits(split_dir)
    if seed_prompt_path:
        seed_prompt_text = seed_prompt_path.read_text(encoding="utf-8").strip()
        print(f"[gepa] warm-start seed from {seed_prompt_path} ({len(seed_prompt_text)} chars)")
    else:
        seed_prompt_text = extract_seed_system_prompt(dataset_name)
    seed_candidate = {"system_prompt": seed_prompt_text}

    adapter = CampaignRelevanceAdapter(model=model, max_workers=max_workers)
    metrics_csv = run_dir / "metrics_history.csv"
    metrics_cb = MetricsHistoryCallback(metrics_csv)
    test_cb = TestEvalCallback(
        testset=test,
        accuracy_fn=lambda cand, data: _accuracy(adapter, cand, data),
        metrics=metrics_cb,
        val_margin=val_margin,
    )

    print(f"[gepa] model={model}")
    print(f"[gepa] reflection_lm={reflection_lm}")
    print(f"[gepa] train={len(train)} val={len(val)} test={len(test)}")
    print(f"[gepa] max_metric_calls={max_metric_calls} skip_perfect_score={skip_perfect_score}")
    print(f"[gepa] run_dir={run_dir}")

    seed_val_acc = _accuracy(adapter, seed_candidate, val)
    seed_test_acc = _accuracy(adapter, seed_candidate, test)
    test_cb.test_scores[0] = seed_test_acc
    metrics_cb.log_manual(
        "baseline_eval",
        iteration=0,
        candidate_idx=0,
        val_accuracy=f"{seed_val_acc:.6f}",
        test_accuracy=f"{seed_test_acc:.6f}",
    )
    print(f"[gepa] baseline val={seed_val_acc:.4f} test={seed_test_acc:.4f}")

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=train,
        valset=val,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=max_metric_calls,
        skip_perfect_score=skip_perfect_score,
        run_dir=str(run_dir / "gepa_state"),
        callbacks=[metrics_cb, test_cb],
        display_progress_bar=True,
        seed=seed,
    )

    val_scores = result.val_aggregate_scores
    best_idx_val = result.best_idx
    best_idx_conservative = test_cb.pick_conservative_best(result, val_scores)

    # Fill missing test scores for any candidate not evaluated during search.
    for idx, candidate in enumerate(result.candidates):
        if idx not in test_cb.test_scores:
            test_cb.test_scores[idx] = _accuracy(adapter, candidate, test)

    best_candidate_val = result.candidates[best_idx_val]
    best_candidate_conservative = result.candidates[best_idx_conservative]

    best_val = val_scores[best_idx_val]
    best_test_val_pick = test_cb.test_scores[best_idx_val]
    best_test_conservative = test_cb.test_scores[best_idx_conservative]
    conservative_val = val_scores[best_idx_conservative]

    metrics_cb.log_manual(
        "final_test_eval",
        iteration=result.total_metric_calls,
        candidate_idx=best_idx_val,
        val_accuracy=f"{best_val:.6f}",
        test_accuracy=f"{best_test_val_pick:.6f}",
        notes="selection=best_by_val",
    )
    metrics_cb.log_manual(
        "final_test_eval",
        iteration=result.total_metric_calls,
        candidate_idx=best_idx_conservative,
        val_accuracy=f"{conservative_val:.6f}",
        test_accuracy=f"{best_test_conservative:.6f}",
        notes=f"selection=conservative_within_{val_margin:.3f}_val_margin",
    )

    (run_dir / "best_prompt.txt").write_text(best_candidate_val["system_prompt"], encoding="utf-8")
    (run_dir / "best_prompt_conservative.txt").write_text(
        best_candidate_conservative["system_prompt"], encoding="utf-8"
    )
    (run_dir / "seed_prompt.txt").write_text(seed_prompt_text, encoding="utf-8")
    with open(run_dir / "gepa_result.json", "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    save_candidates_table(result, run_dir / "candidates.csv", test_scores=test_cb.test_scores)

    summary = {
        "model": model,
        "dataset_name": dataset_name,
        "run_id": run_id,
        "reflection_lm": reflection_lm,
        "max_metric_calls": max_metric_calls,
        "skip_perfect_score": skip_perfect_score,
        "val_margin_for_conservative_selection": val_margin,
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "baseline": {"val_accuracy": seed_val_acc, "test_accuracy": seed_test_acc},
        "best_by_val": {
            "candidate_idx": best_idx_val,
            "val_accuracy": best_val,
            "test_accuracy": best_test_val_pick,
        },
        "best_conservative": {
            "candidate_idx": best_idx_conservative,
            "val_accuracy": conservative_val,
            "test_accuracy": best_test_conservative,
        },
        "total_metric_calls": result.total_metric_calls,
        "artifacts": {
            "metrics_history_csv": str(metrics_csv),
            "candidates_csv": str(run_dir / "candidates.csv"),
            "best_prompt_txt": str(run_dir / "best_prompt.txt"),
            "best_prompt_conservative_txt": str(run_dir / "best_prompt_conservative.txt"),
        },
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[gepa] best-by-val: idx={best_idx_val} val={best_val:.4f} test={best_test_val_pick:.4f}")
    print(
        f"[gepa] conservative: idx={best_idx_conservative} "
        f"val={conservative_val:.4f} test={best_test_conservative:.4f}"
    )
    print(f"[gepa] saved summary -> {run_dir / 'summary.json'}")


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="GEPA prompt optimization for campaign relevance")
    parser.add_argument("--model", required=True, help="OpenAI model id, e.g. gpt-4.1-nano")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split-dir", default=str(PO_ROOT / "splits"))
    parser.add_argument("--max-metric-calls", type=int, default=1500)
    parser.add_argument("--reflection-lm", default="openai/gpt-5.4")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-perfect-score",
        action="store_true",
        help="Skip reflection when minibatch is perfect (default: False, always reflect)",
    )
    parser.add_argument(
        "--val-margin",
        type=float,
        default=VAL_MARGIN_FOR_SELECTION,
        help="Val gap for conservative test-based selection among near-best candidates",
    )
    parser.add_argument("--prepare-splits-only", action="store_true")
    parser.add_argument("--force-resplit", action="store_true")
    parser.add_argument(
        "--seed-prompt-file",
        default=None,
        help="Warm-start GEPA from this system prompt instead of Langfuse seed",
    )
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    manifest = prepare_splits(split_dir, args.dataset, force=args.force_resplit)
    print(f"[gepa] splits: {manifest['counts']} (60/20/20 stratified)")

    if args.prepare_splits_only:
        return

    seed_prompt_path = None
    if args.seed_prompt_file:
        seed_prompt_path = Path(args.seed_prompt_file)
        if not seed_prompt_path.is_absolute():
            seed_prompt_path = (ROOT / seed_prompt_path).resolve()

    run_for_model(
        model=args.model,
        dataset_name=args.dataset,
        split_dir=split_dir,
        max_metric_calls=args.max_metric_calls,
        reflection_lm=args.reflection_lm,
        max_workers=args.max_workers,
        seed=args.seed,
        skip_perfect_score=args.skip_perfect_score,
        val_margin=args.val_margin,
        seed_prompt_path=seed_prompt_path,
    )


if __name__ == "__main__":
    main()
