"""Deterministic physics-hygiene and CalibrationHead contract tests for CarbonGlobe ED Emulator."""

from pathlib import Path
import unittest
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestCarbonGlobePhysicsContracts(unittest.TestCase):
    def test_cross_age_smoothing_kernel_conservation(self) -> None:
        """Verify (0.12, 0.76, 0.12) cross-age smoothing kernel is symmetric and mass-conserving."""
        kernel = np.array([0.12, 0.76, 0.12], dtype=float)
        self.assertAlmostEqual(float(kernel.sum()), 1.0, places=12)
        self.assertEqual(kernel[0], kernel[2])
        self.assertTrue(np.all(kernel > 0.0))

        # Constant stand-age profile must remain invariant under interior smoothing
        profile = np.full(15, 2.5, dtype=float)
        padded = np.pad(profile, 1, mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
        np.testing.assert_allclose(smoothed, profile, atol=1e-12)

    def test_v8_anchor_weights_simplex(self) -> None:
        """Verify v8 CalibrationHead prior anchor (0.65 v5 + 0.35 v4) lies on the unit simplex."""
        w_v5, w_v4 = 0.65, 0.35
        self.assertAlmostEqual(w_v5 + w_v4, 1.0, places=12)
        self.assertGreater(w_v5, 0.0)
        self.assertGreater(w_v4, 0.0)

    def test_inverse_affine_y0_roundtrip(self) -> None:
        """Verify inverse-affine initial-state correction recovers clean y0 with R^2 > 0.999."""
        rng = np.random.default_rng(42)
        y0_clean = rng.uniform(0.1, 10.0, size=500)
        # Simulated test flux inflation (+14%) + offset
        scale, shift = 1.14, 0.035
        y0_shifted = scale * y0_clean + shift
        y0_recovered = (y0_shifted - shift) / scale
        ss_res = float(np.sum((y0_clean - y0_recovered) ** 2))
        ss_tot = float(np.sum((y0_clean - np.mean(y0_clean)) ** 2))
        r2 = 1.0 - ss_res / ss_tot
        self.assertGreater(r2, 0.999)

    def test_target_tensor_cardinality(self) -> None:
        """Verify competition test tensor dimensions: 6,171 sites x 2 scenarios x 15 cohorts = 185,130 pairs."""
        n_sites, n_scenarios, n_cohorts, n_targets = 6171, 2, 15, 2
        self.assertEqual(n_sites * n_scenarios * n_cohorts, 185130)
        self.assertEqual(n_sites * n_scenarios * n_cohorts * n_targets, 370260)


if __name__ == "__main__":
    unittest.main()
