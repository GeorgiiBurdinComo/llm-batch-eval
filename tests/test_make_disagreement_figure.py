import unittest

import pandas as pd

from make_disagreement_figure import order_rows, select_best_runs


class BestRunSelectionTests(unittest.TestCase):
    def test_prefers_maximum_coverage_then_earliest_run(self) -> None:
        measurements = pd.DataFrame(
            [
                {
                    "model": "model-a",
                    "run_id": "late-small",
                    "custom_id": f"id-{idx}",
                    "timestamp": pd.Timestamp("2026-05-02T00:00:00Z"),
                }
                for idx in range(4)
            ]
            + [
                {
                    "model": "model-a",
                    "run_id": "early-large",
                    "custom_id": f"id-{idx}",
                    "timestamp": pd.Timestamp("2026-05-01T00:00:00Z"),
                }
                for idx in range(5)
            ]
            + [
                {
                    "model": "model-a",
                    "run_id": "late-large",
                    "custom_id": f"id-{idx}",
                    "timestamp": pd.Timestamp("2026-05-03T00:00:00Z"),
                }
                for idx in range(5)
            ]
            + [
                {
                    "model": "model-b",
                    "run_id": "too-small",
                    "custom_id": f"id-{idx}",
                    "timestamp": pd.Timestamp("2026-05-01T00:00:00Z"),
                }
                for idx in range(3)
            ]
        )

        selected = select_best_runs(measurements, min_items=5)

        self.assertEqual(list(selected["model"]), ["model-a"])
        self.assertEqual(selected.iloc[0]["run_id"], "early-large")


class RowOrderingTests(unittest.TestCase):
    def test_places_mixed_rows_in_one_contiguous_band(self) -> None:
        labels = pd.Series(
            [False, False, True, True, False],
            index=["easy-blue", "band-low", "easy-red", "band-high", "missing"],
        )
        predictions = pd.DataFrame(
            {
                "m1": [False, False, True, True, pd.NA],
                "m2": [False, True, True, False, pd.NA],
                "m3": [False, True, True, True, pd.NA],
            },
            index=labels.index,
            dtype="boolean",
        )

        ordered_stats, _ = order_rows(labels, predictions)
        ordered_ids = ordered_stats.index.tolist()
        mixed_positions = [
            idx for idx, custom_id in enumerate(ordered_ids) if bool(ordered_stats.loc[custom_id, "mixed"])
        ]

        self.assertEqual(ordered_ids[0], "easy-blue")
        self.assertEqual(ordered_ids[-1], "missing")
        self.assertEqual(mixed_positions, list(range(mixed_positions[0], mixed_positions[-1] + 1)))
        self.assertEqual(ordered_ids[mixed_positions[0]:mixed_positions[-1] + 1], ["band-low", "band-high"])


if __name__ == "__main__":
    unittest.main()
