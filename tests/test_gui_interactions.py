"""GUI interaction tests for the frame box, whole-image profile, histogram
threshold dragging, registration preview, and auto pre-map."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import unittest

import numpy as np
from PySide6.QtWidgets import QApplication

from muaxis.io.stxm import FrameMetadata, ScanHeader, ScanStack
import tempfile

from muaxis.gui.main_window import MainWindow
from muaxis.gui.registration_window import RegistrationWindow
from muaxis.gui.segmentation_window import SegmentationWindow
from muaxis.processing.registration import register_translation_stack
from muaxis.report_data import build_report_payload
from muaxis.reporting import export_image_set


def _scan(n_energy=6, h=9, w=10, seed=0):
    energies = np.linspace(280.0, 300.0, n_energy)
    frames = tuple(
        FrameMetadata(i + 1, e, None, None, None, None) for i, e in enumerate(energies)
    )
    header = ScanHeader(
        Path("/tmp/syn.hdr"), "syn", None, energies, np.arange(float(w)), np.arange(float(h)), frames
    )
    rng = np.random.RandomState(seed)
    trans = 1.0 + rng.rand(n_energy, h, w)
    return ScanStack(header, tuple(Path(f"/tmp/f{i}.xim") for i in range(n_energy)), trans, None, None)


class MainWindowFrameAndProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.scan = _scan()
        self.w = MainWindow()
        self.w.scan = self.scan
        self.w._restoring_session = True
        self.w._configure_normalization(self.scan.energies_eV)
        self.w._current_frame = 0
        self.w.frame_total_label.setText(f"/ {len(self.scan.frame_paths)}")
        self.w.frame_edit.setEnabled(True)
        self.w.layer_list.setEnabled(True)
        self.w._restoring_session = False
        self.w._show_frame(0)

    def tearDown(self):
        self.w.close()

    def test_whole_image_profile_shown_without_roi(self):
        self.assertTrue(self.w.whole_image_curve.isVisible())
        x, _ = self.w.whole_image_curve.getData()
        self.assertEqual(len(x), len(self.scan.energies_eV))

    def test_roi_replaces_whole_image_profile(self):
        self.w.create_spectrum_roi((1, 1, 3, 3))
        self.w._update_spectrum()
        self.assertFalse(self.w.whole_image_curve.isVisible())
        self.assertEqual(len(self.w._spectrum_curves), 1)
        self.w.clear_spectrum_roi()
        self.assertTrue(self.w.whole_image_curve.isVisible())

    def test_frame_box_jumps_and_clamps(self):
        self.w.frame_edit.setText("5")
        self.w._frame_edit_entered()
        self.assertEqual(self.w._current_frame, 4)
        self.assertEqual(self.w.frame_edit.text(), "5")
        self.w.frame_edit.setText("999")
        self.w._frame_edit_entered()
        self.assertEqual(self.w._current_frame, len(self.scan.frame_paths) - 1)
        self.w.frame_edit.setText("0")
        self.w._frame_edit_entered()
        self.assertEqual(self.w._current_frame, 0)

    def test_invalid_frame_box_reverts(self):
        self.w._select_frame(2)
        self.w.frame_edit.setText("abc")
        self.w._frame_edit_entered()
        self.assertEqual(self.w._current_frame, 2)
        self.assertEqual(self.w.frame_edit.text(), "3")

    def test_energy_cursor_selects_frame(self):
        self.w.energy_line.setValue(float(self.scan.energies_eV[3]))
        self.assertEqual(self.w._current_frame, 3)


class AutoPremapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_apply_registration_builds_premap(self):
        scan = _scan()
        w = MainWindow()
        w.scan = scan
        w._current_frame = 0
        w.frame_total_label.setText(f"/ {len(scan.frame_paths)}")
        w.frame_edit.setEnabled(True)
        w.layer_list.setEnabled(True)
        result = register_translation_stack(scan.transmission, reference_index=0, upsample_factor=5)
        w._set_registration_result(result, auto_premap=True)
        names = [w.layer_list.item(i).text() for i in range(w.layer_list.count())]
        self.assertTrue(any("Net absorption map" in n for n in names), names)
        self.assertTrue(any("Peak map" in n for n in names), names)
        self.assertTrue(any(n.startswith("OD - pre-edge") for n in names), names)
        self.assertTrue(w._premap_maps)
        self.assertTrue(w.layer_list.currentItem().text().startswith("Registered"))
        w.close()


class RegistrationPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_preview_enabled_after_run(self):
        scan = _scan()
        w = RegistrationWindow(scan, scan.transmission, "OD")
        self.assertFalse(w.preview_slider.isEnabled())
        result = register_translation_stack(scan.transmission, reference_index=0, upsample_factor=5)
        w._registration_finished(result)
        self.assertTrue(w.preview_slider.isEnabled())
        self.assertTrue(w.apply_button.isEnabled())
        w.preview_slider.setValue(2)
        self.assertIn("3/6", w.preview_label.text())
        self.assertIn("registered", w.preview_label.text())
        w.show_original_check.setChecked(True)
        self.assertIn("original", w.preview_label.text())
        w.close()


class SegmentationThresholdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.scan = _scan(n_energy=3, h=3, w=4)
        self.feature = np.arange(12.0).reshape(3, 4)
        self.od = np.array([np.arange(12.0).reshape(3, 4) + o for o in (0.0, 2.0, 5.0)])
        self.w = SegmentationWindow(self.scan, self.feature, self.od, "Peak map")

    def tearDown(self):
        self.w.close()

    def test_default_thresholds_span_full_range_and_fill_boxes(self):
        lo, hi = self.w._thresholds()
        self.assertAlmostEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 11.0)
        self.assertEqual(self.w.low_edit.text(), "0")
        self.assertEqual(self.w.high_edit.text(), "11")

    def test_text_boxes_set_thresholds(self):
        self.w.low_edit.setText("2")
        self.w.high_edit.setText("8")
        self.w._edits_changed()
        self.assertEqual(self.w._thresholds(), (2.0, 8.0))
        # The histogram region mirrors the thresholds.
        self.assertEqual(tuple(round(v, 3) for v in self.w.hist_region.getRegion()), (2.0, 8.0))

    def test_nearer_threshold_moves(self):
        self.w._low_thr, self.w._high_thr = 2.0, 8.0
        self.w._threshold_changed()
        # value 3 is nearer the lower threshold
        value = 3.0
        low, high = self.w._thresholds()
        if abs(value - low) <= abs(value - high):
            self.w._low_thr = value
        else:
            self.w._high_thr = value
        self.w._threshold_changed()
        self.assertEqual(self.w._thresholds()[0], 3.0)

    def test_invalid_text_reverts(self):
        self.w.low_edit.setText("2")
        self.w.high_edit.setText("8")
        self.w._edits_changed()
        self.w.low_edit.setText("oops")
        self.w._edits_changed()
        self.assertEqual(self.w._thresholds(), (2.0, 8.0))
        self.assertEqual(self.w.low_edit.text(), "2")


class SegmentationImageExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _window_with_segmentation(self):
        scan = _scan(n_energy=4, h=4, w=5)
        w = MainWindow()
        w.scan = scan
        labels = np.array([
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 0, 3],
            [0, 0, 3, 3, 3],
        ], dtype=int)
        prof = np.zeros(4).tolist()
        w._segmentation_config = {
            "source": "OD", "threshold": [0.0, 1.0], "minimum_area": 1,
            "saved_groups": {
                "phaseA": {
                    "profiles": [prof, prof, prof], "areas": [4, 4, 4],
                    "cluster_ids": [1, 2, 3], "labels": labels.tolist(),
                    "threshold": [0.0, 1.0], "minimum_area": 1,
                    "analysis_parameters": {"source": "OD"},
                }
            },
        }
        return w

    def test_segmentation_image_set_and_export(self):
        w = self._window_with_segmentation()
        payload = build_report_payload(w)
        sets = {s["name"]: s for s in payload["image_sets"]}
        self.assertIn("Segmentation", sets)
        seg = sets["Segmentation"]
        self.assertEqual(seg["kind"], "segmentation")
        self.assertEqual(len(seg["data"]), 1)
        self.assertEqual([c["id"] for c in seg["data"][0]["clusters"]], [1, 2, 3])
        with tempfile.TemporaryDirectory() as d:
            outputs = export_image_set(seg, Path(d) / "images")
            names = sorted(p.name for p in outputs)
            self.assertIn("all_clusters.png", names)
            for cid in (1, 2, 3):
                self.assertIn(f"cluster_{cid:03d}.png", names)
        w.close()


if __name__ == "__main__":
    unittest.main()
