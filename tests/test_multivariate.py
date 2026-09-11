import unittest
import warnings

import numpy as np
from scipy.optimize import nnls

from muaxis.processing.multivariate import (
    AnalysisCancelled,
    decompose_stack,
    linear_combination_map,
    mcr_als_reconstruct,
    nmf_reconstruct,
    nnls_reconstruct,
)


class MultivariateTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        energy = np.linspace(0, 1, 30)
        self.spectra = np.array([
            np.exp(-((energy - .2) / .13) ** 2),
            np.exp(-((energy - .65) / .18) ** 2),
            .05 + .8 * energy,
        ])
        self.scores = self.rng.random((120, 3))
        self.scores[:3] = np.eye(3)
        self.stack = (self.scores @ self.spectra).T.reshape(30, 10, 12)

    def test_nmf_clips_negative_od_and_preserves_factor_product(self):
        noisy = self.stack - .025
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            fitted, scores, spectra = nmf_reconstruct(noisy, 3, 600)
        self.assertTrue(np.isfinite(fitted).all())
        self.assertTrue((scores >= 0).all())
        self.assertTrue((spectra >= 0).all())
        self.assertTrue((spectra.max(axis=1) > .1).all())
        np.testing.assert_allclose((scores @ spectra).T.reshape(fitted.shape), fitted)
        target = np.maximum(noisy, 0)
        self.assertLess(np.linalg.norm(target - fitted) / np.linalg.norm(target), .04)
        rgb = linear_combination_map(scores, (10, 12), np.eye(3), 10)
        self.assertGreater(np.ptp(rgb[..., :3]), .5)

    def test_nmf_large_finite_scale_is_preserved_without_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            fitted, scores, spectra = nmf_reconstruct(self.stack * 1e150, 3, 40)
        self.assertTrue(np.isfinite(fitted).all())
        np.testing.assert_allclose((scores @ spectra).T.reshape(fitted.shape) / 1e150,
                                   fitted / 1e150)
        self.assertGreater(spectra.max(), 1e149)

    def test_nnls_agrees_with_scipy_and_returns_actual_scores(self):
        for basis in (self.spectra.T, self.rng.normal(size=(30, 4))):
            matrix = self.rng.normal(.2, .5, size=(120, 30))
            stack = matrix.T.reshape(30, 10, 12)
            fitted, scores = nnls_reconstruct(stack, basis, return_scores=True)
            expected = np.array([nnls(basis, row)[0] for row in matrix])
            np.testing.assert_allclose(scores, expected, atol=1e-9, rtol=1e-8)
            np.testing.assert_allclose(fitted.reshape(30, -1).T, scores @ basis.T)

    def test_nnls_rank_deficient_and_zero_spectra(self):
        basis = np.column_stack([self.spectra[0], self.spectra[0], self.spectra[1], np.zeros(30)])
        fitted, scores = nnls_reconstruct(self.stack, basis, return_scores=True)
        expected = np.array([basis @ nnls(basis, row)[0] for row in self.stack.reshape(30, -1).T])
        np.testing.assert_allclose(fitted.reshape(30, -1).T, expected, atol=1e-9)
        self.assertTrue((scores >= 0).all())
        zero, zscores = nnls_reconstruct(np.zeros_like(self.stack), basis, return_scores=True)
        self.assertEqual(np.count_nonzero(zero), 0)
        self.assertEqual(np.count_nonzero(zscores), 0)

    def test_mcr_is_nonnegative_als_and_fits_signed_observations(self):
        noisy = self.stack - .025
        fitted, scores, spectra, count = mcr_als_reconstruct(noisy, 3, 50)
        self.assertLessEqual(count, 50)
        self.assertTrue((scores >= 0).all())
        self.assertTrue((spectra >= 0).all())
        self.assertTrue((spectra.max(axis=1) > .1).all())
        np.testing.assert_allclose((scores @ spectra).T.reshape(fitted.shape), fitted)
        self.assertLess(np.linalg.norm(noisy - fitted) / np.linalg.norm(noisy), .035)
        # The final S update must solve NNLS against signed observations.
        matrix = noisy.reshape(30, -1).T
        expected = np.array([nnls(scores, matrix[:, e])[0] for e in range(30)]).T
        np.testing.assert_allclose(spectra, expected, rtol=1e-6, atol=1e-7)

    def test_mcr_closure(self):
        concentration = self.scores / self.scores.sum(axis=1, keepdims=True)
        stack = (concentration @ self.spectra).T.reshape(30, 10, 12)
        fitted, scores, spectra, _ = mcr_als_reconstruct(stack, 3, 50, closure=True)
        np.testing.assert_allclose(scores.sum(axis=1), 1, atol=1e-12)
        np.testing.assert_allclose(fitted.reshape(30, -1).T, scores @ spectra)

    def test_nan_border_is_excluded_and_infinity_rejected(self):
        stack = self.stack.copy()
        stack[0, 0, 0] = np.nan
        for solver in (nmf_reconstruct, mcr_als_reconstruct):
            fitted, scores, *_ = solver(stack, 3, 10)
            self.assertTrue(np.isnan(fitted[:, 0, 0]).all())
            self.assertEqual(np.count_nonzero(scores[0]), 0)
            self.assertTrue(np.isfinite(fitted[:, 1:, :]).all())
        fitted, scores = nnls_reconstruct(stack, self.spectra.T, return_scores=True)
        self.assertTrue(np.isnan(fitted[:, 0, 0]).all())
        self.assertEqual(np.count_nonzero(scores[0]), 0)
        for invalid in (np.full_like(stack, np.nan), np.full_like(stack, np.inf)):
            for solver in (nmf_reconstruct, mcr_als_reconstruct):
                with self.assertRaises(ValueError):
                    solver(invalid)

    def test_empty_signal_does_not_report_blank_success(self):
        for stack in (np.zeros_like(self.stack), -np.ones_like(self.stack)):
            for solver in (nmf_reconstruct, mcr_als_reconstruct):
                with self.assertRaisesRegex(ValueError, "positive OD"):
                    solver(stack)

    def test_cancellation_raises_without_partial_output(self):
        for mode in ("nmf", "mcr", "nnls", "map", "pca"):
            calls = 0

            def cancel():
                nonlocal calls
                calls += 1
                return calls >= 3

            with self.subTest(mode=mode), self.assertRaises(AnalysisCancelled):
                if mode == "nmf":
                    nmf_reconstruct(self.stack, cancel_check=cancel)
                elif mode == "mcr":
                    mcr_als_reconstruct(self.stack, cancel_check=cancel)
                elif mode == "nnls":
                    nnls_reconstruct(self.stack, self.spectra.T, cancel_check=cancel)
                elif mode == "map":
                    linear_combination_map(self.scores, (10, 12), np.eye(3), 10, cancel_check=cancel)
                else:
                    decompose_stack(self.stack, cancel_check=cancel)

    def test_pca_and_svd_full_rank_reconstruct_original(self):
        for method in ("PCA (centered)", "SVD (uncentered)"):
            result = decompose_stack(self.stack, 30, method)
            np.testing.assert_allclose(result.reconstructed, self.stack, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
