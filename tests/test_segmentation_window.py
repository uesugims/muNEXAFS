import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import unittest

import numpy as np
from PySide6.QtWidgets import QApplication

from muaxis.gui.segmentation_window import SegmentationWindow
from muaxis.io.stxm import FrameMetadata, ScanHeader, ScanStack


class SegmentationWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        energies = np.array([280.0, 281.0, 282.0])
        frames = tuple(
            FrameMetadata(index + 1, energy, None, None, None, None)
            for index, energy in enumerate(energies)
        )
        header = ScanHeader(
            Path("/tmp/synthetic.hdr"), None, None, energies,
            np.arange(4.0), np.arange(3.0), frames,
        )
        self.scan = ScanStack(header, (), np.ones((3, 3, 4)), None, None)
        self.feature = np.arange(12.0).reshape(3, 4)
        self.od = np.array([
            np.arange(12.0).reshape(3, 4) + offset for offset in (0.0, 2.0, 5.0)
        ])
        self.window = SegmentationWindow(
            self.scan, self.feature, self.od, "Peak map",
        )

    def tearDown(self):
        self.window.close()

    def test_histogram_log_y_checkbox_changes_only_the_count_axis(self):
        plot = self.window.histogram.getPlotItem()
        thresholds = self.window._thresholds()
        self.assertFalse(self.window.histogram_log_y.isChecked())
        self.assertFalse(plot.getAxis("left").logMode)

        self.window.histogram_log_y.setChecked(True)

        self.assertTrue(plot.getAxis("left").logMode)
        self.assertFalse(plot.getAxis("bottom").logMode)
        self.assertEqual(self.window._thresholds(), thresholds)

    def test_two_energy_normalization_enables_energy_fields(self):
        self.assertFalse(self.window.norm_e1.isEnabled())
        self.assertFalse(self.window.norm_e2.isEnabled())
        self.window.norm_combo.setCurrentText("Two energy values")

        self.assertTrue(self.window.norm_e1.isEnabled())
        self.assertTrue(self.window.norm_e2.isEnabled())
        self.window.norm_combo.setCurrentText("None")
        self.assertFalse(self.window.norm_e1.isEnabled())
        self.assertFalse(self.window.norm_e2.isEnabled())

    def test_run_and_two_energy_normalization_update_the_profile(self):
        self.window.run_segmentation()
        self.assertEqual(self.window.table.rowCount(), 1)
        self.window.norm_combo.setCurrentText("Two energy values")
        self.window.norm_e1.setValue(280.0)
        self.window.norm_e2.setValue(282.0)

        _, values = self.window.profile_plot.listDataItems()[0].getData()

        np.testing.assert_allclose(values, [0.0, 0.4, 1.0])

    def test_saved_label_restores_parameters_and_reviews_all_profiles(self):
        saved = {
            "Review A": {
                "profiles": [[0.1, 0.4, 0.9], [0.8, 0.5, 0.2]],
                "areas": [4, 3],
                "cluster_ids": [2, 5],
                "labels": [[0, 2, 2, 0], [0, 0, 0, 0], [5, 5, 5, 0]],
                "threshold": [2.0, 8.0],
                "minimum_area": 3,
                "analysis_parameters": {"source": "Peak map", "threshold": [2.0, 8.0]},
            }
        }
        self.window.close()
        self.window = SegmentationWindow(
            self.scan, self.feature, self.od, "Peak map",
            {"Peak map": self.feature, "OD": self.od}, saved_groups=saved,
        )

        self.window.saved_label_combo.setCurrentText("Review A")

        self.assertEqual(self.window.label_edit.text(), "Review A")
        self.assertEqual(self.window.min_area_spin.value(), 3)
        self.assertAlmostEqual(self.window._thresholds()[0], 2.0, places=2)
        self.assertAlmostEqual(self.window._thresholds()[1], 8.0, places=2)
        self.assertEqual(
            {int(self.window.table.item(row, 0).text()) for row in range(self.window.table.rowCount())},
            {2, 5},
        )
        self.assertEqual(len(self.window.profile_plot.listDataItems()), 2)
        self.assertFalse(self.window.save_button.isEnabled())

        selected_row = next(
            row for row in range(self.window.table.rowCount())
            if self.window.table.item(row, 0).text() == "5"
        )
        self.window.table.selectRow(selected_row)
        pens = [curve.opts["pen"].widthF() for curve in self.window.profile_plot.listDataItems()]
        self.assertEqual(len(pens), 2)
        self.assertIn(4.0, pens)
        self.assertIn(1.0, pens)


if __name__ == "__main__":
    unittest.main()
