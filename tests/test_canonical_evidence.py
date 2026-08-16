import math
import unittest

import numpy as np
import pandas as pd

from canonical_evidence import (
    PANEL_N,
    exact_mcnemar_p,
    exact_sensitivity,
    exact_unconditional_power,
    holm_adjust,
    reconstruct_panel,
)


class ExactStatisticsTests(unittest.TestCase):
    def test_exact_mcnemar_reference_value(self) -> None:
        self.assertAlmostEqual(exact_mcnemar_p(9, 1), 11 / 1024)

    def test_holm_reference_values(self) -> None:
        actual = holm_adjust([0.01, 0.04, 0.03])
        np.testing.assert_allclose(actual, [0.03, 0.06, 0.06])

    def test_unconditional_power_matches_direct_multinomial_sum(self) -> None:
        n, rho, g, alpha = 8, 0.4, 0.2, 0.05
        p_b, p_c = (rho + g) / 2, (rho - g) / 2
        expected = 0.0
        for b in range(n + 1):
            for c in range(n - b + 1):
                concordant = n - b - c
                probability = (
                    math.factorial(n)
                    / (math.factorial(b) * math.factorial(c) * math.factorial(concordant))
                    * p_b**b
                    * p_c**c
                    * (1 - rho) ** concordant
                )
                if exact_mcnemar_p(b, c) <= alpha:
                    expected += probability
        self.assertAlmostEqual(exact_unconditional_power(n, rho, g, alpha), expected, places=14)

    def test_power_rejects_inadmissible_effect(self) -> None:
        with self.assertRaisesRegex(ValueError, "g <= rho"):
            exact_unconditional_power(300, 0.02, 0.03, 0.05)

    def test_sensitivity_never_exceeds_discordance(self) -> None:
        rho = 0.12
        sensitivity = exact_sensitivity(300, rho, 0.05)
        self.assertLessEqual(sensitivity, rho)
        self.assertGreaterEqual(exact_unconditional_power(300, rho, sensitivity, 0.05), 0.8)

    def test_null_size_is_bounded_by_nominal_alpha(self) -> None:
        self.assertLessEqual(exact_unconditional_power(300, 0.1, 0.0, 0.05), 0.05)


class PanelReconstructionTests(unittest.TestCase):
    def test_reconstructs_exact_scheduled_panel(self) -> None:
        ids = [f"id-{index:03d}" for index in range(PANEL_N)]
        rows = []
        for model, run_id, count in (
            ("model-a", "gha-1", 300),
            ("model-a", "gha-2", 250),
            ("model-a", "local-1", 300),
        ):
            for custom_id in ids[:count]:
                rows.append(
                    {
                        "model": model,
                        "run_id": run_id,
                        "custom_id": custom_id,
                        "timestamp": pd.Timestamp("2026-03-15T12:00:00Z"),
                    }
                )
        measurements = pd.DataFrame(rows)
        panel, eligible, panel_data = reconstruct_panel(measurements, set(ids))
        self.assertEqual(panel, ids)
        self.assertEqual(set(eligible["run_id"]), {"gha-1", "gha-2"})
        self.assertEqual(set(panel_data["run_id"]), {"gha-1", "gha-2"})

    def test_rejects_panel_not_in_benchmark(self) -> None:
        ids = [f"id-{index:03d}" for index in range(PANEL_N)]
        measurements = pd.DataFrame(
            {
                "model": "model-a",
                "run_id": "gha-1",
                "custom_id": ids,
                "timestamp": pd.Timestamp("2026-03-15T12:00:00Z"),
            }
        )
        with self.assertRaisesRegex(AssertionError, "not a subset"):
            reconstruct_panel(measurements, set(ids[:-1]))


if __name__ == "__main__":
    unittest.main()
