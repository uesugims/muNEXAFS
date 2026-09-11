"""Regression checks for completion, cancellation and recovery of the Qt worker."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtWidgets import QApplication

from muaxis.gui.multivariate_window import MultivariateWindow
from muaxis.processing.multivariate import AnalysisCancelled


class MultivariateWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def setUp(self):
        rng = np.random.default_rng(6)
        spectra = rng.uniform(0.1, 1.0, (3, 17))
        scores = rng.uniform(0.0, 2.0, (18 * 21, 3))
        self.stack = (scores @ spectra).T.reshape(17, 18, 21)
        self.stack[:, 0, 0] = -0.05
        self.stack[:, 0, 1] = np.nan
        self.window = MultivariateWindow(np.arange(17.0), self.stack)
        # The window now defaults to "PCA + clustering"; the RGB-composite and
        # NMF/NNLS/MCR modes below are exercised via the explicit method.
        self.window.method.setCurrentText("PCA (centered)")
        self.window.settings.components.setValue(3)
        self.destroyed = False
        self.window.destroyed.connect(self.mark_destroyed)
        self.window.settings.iterations.setValue(10)
        self.results = []
        self.window.resultReady.connect(self.results.append)

    def mark_destroyed(self):
        self.destroyed = True

    def wait_until(self, predicate, timeout=10):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.app.processEvents()
        self.assertTrue(predicate(), "Window did not reach the expected state")

    def tearDown(self):
        if self.destroyed:
            return
        self.window.stop_analysis()
        self.wait_until(lambda: self.window.thread is None)
        self.window.close()
        self.app.processEvents()

    def test_modes_render_profiles_and_map_then_allow_another_run(self):
        for mode in (None, "nmf_check", "nnls_check", "mcr_check"):
            with self.subTest(mode=mode):
                for name in ("nmf_check", "nnls_check", "mcr_check"):
                    getattr(self.window, name).setChecked(name == mode)
                before = len(self.results)
                self.window.run_analysis()
                self.wait_until(lambda: self.window.thread is None)
                self.assertEqual(len(self.results), before + 1)
                result = self.results[-1]
                self.assertTrue(np.isfinite(result.components).all())
                self.assertGreater(np.max(np.abs(result.components)), 0.01)
                self.assertGreater(np.ptp(result.rgb_map[..., :3]), 0.1)
                self.assertEqual(len(self.window.profile.listDataItems()), 3)
                np.testing.assert_array_equal(self.window.image.image, result.rgb_map)
                self.assertTrue(self.window.run.isEnabled())
                self.assertFalse(self.window.stop.isEnabled())
                if mode is not None:
                    valid = np.all(np.isfinite(self.stack), axis=0).ravel()
                    reconstruction = result.reconstructed.reshape(17, -1).T
                    np.testing.assert_allclose((result.scores @ result.components)[valid],
                                               reconstruction[valid], rtol=1e-8, atol=1e-10)

    def test_clustering_mode_produces_cluster_result(self):
        self.window.method.setCurrentText("PCA + clustering")
        self.window.clusters.setValue(4)
        before = len(self.results)
        self.window.run_analysis()
        self.wait_until(lambda: self.window.thread is None)
        self.assertEqual(len(self.results), before + 1)
        result = self.results[-1]
        self.assertEqual(result.n_clusters, 4)
        self.assertEqual(result.labels.shape, self.stack.shape[1:])
        self.assertEqual(result.rgb_map.shape, (*self.stack.shape[1:], 4))
        self.assertTrue((result.labels == -1).any())          # NaN pixel excluded
        self.assertEqual(result.mean_spectra.shape, (4, self.stack.shape[0]))
        np.testing.assert_array_equal(self.window.image.image, result.rgb_map)
        self.assertTrue(self.window.run.isEnabled())
        self.assertFalse(self.window.stop.isEnabled())

    def test_exception_releases_thread_and_next_run_succeeds(self):
        self.window.nnls_check.setChecked(True)
        with patch("muaxis.gui.multivariate_window.nnls_reconstruct", side_effect=ValueError("test solver error")):
            self.window.run_analysis()
            self.wait_until(lambda: self.window.thread is None)
        self.assertIn("test solver error", self.window.status.text())
        self.assertEqual(self.results, [])
        self.assertTrue(self.window.run.isEnabled())
        self.window.run_analysis()
        self.wait_until(lambda: self.window.thread is None)
        self.assertEqual(len(self.results), 1)

    def test_stop_discards_partial_result_and_preserves_previous_result(self):
        self.window.run_analysis()
        self.wait_until(lambda: self.window.thread is None)
        previous = self.window.result
        entered = threading.Event()

        def cancellable_solver(*args, cancel_check, **kwargs):
            entered.set()
            while not cancel_check():
                time.sleep(0.001)
            raise AnalysisCancelled("stopped")

        self.window.nnls_check.setChecked(True)
        with patch("muaxis.gui.multivariate_window.nnls_reconstruct", side_effect=cancellable_solver):
            self.window.run_analysis()
            self.wait_until(entered.is_set)
            started = time.monotonic()
            self.window.stop.click()
            self.wait_until(lambda: self.window.thread is None, timeout=2)
        self.assertLess(time.monotonic() - started, 2)
        self.assertIn("stopped", self.window.status.text())
        self.assertIs(self.window.result, previous)
        self.assertEqual(len(self.results), 1)
        self.window.run_analysis()
        self.wait_until(lambda: self.window.thread is None)
        self.assertEqual(len(self.results), 2)

    def test_close_during_calculation_defers_destruction_until_thread_finishes(self):
        entered = threading.Event()

        def cancellable_solver(*args, cancel_check, **kwargs):
            entered.set()
            while not cancel_check():
                time.sleep(0.001)
            raise AnalysisCancelled("stopped")

        self.window.nmf_check.setChecked(True)
        self.window.show()
        with patch("muaxis.gui.multivariate_window.nmf_reconstruct", side_effect=cancellable_solver):
            self.window.run_analysis()
            self.wait_until(entered.is_set)
            self.window.close()
            self.wait_until(lambda: self.destroyed, timeout=2)
        self.assertEqual(self.results, [])
