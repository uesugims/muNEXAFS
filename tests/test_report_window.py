"""Exercise the real report preview and selected exports through Qt workers."""
import csv
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtWidgets import QApplication

from muaxis.gui.report_window import ReportWindow
from muaxis.reporting import export_pdf


class ReportWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="muaxis-report-gui-test-")
        self.root = Path(self.temporary.name)
        energies = np.array([280.0, 281.5, 284.0, 290.0])
        raw = np.array([1.0, 2.0, 3.0, 5.0])
        used = np.array([0.0, 0.25, 0.5, 1.0])
        self.payload = {
            "metadata": {
                "title": "Report regression",
                "scan": str(self.root / "synthetic.hdr"),
                "measurement_parameters": {"Dwell time (ms)": 3.0, "Energy frames": 4},
                "analysis_parameters": {"OD": {"I0 ROI": [0, 0, 2, 2]}},
            },
            "energies": energies,
            "layers": [{
                "name": "OD",
                "image": np.arange(20.0).reshape(4, 5),
                "source": "Synthetic transmission",
                "condition": "Optical density from direct beam ROI",
                "parameters": {"I0 ROI": [0, 0, 2, 2], "Energy (eV)": 284.0},
                "energies": energies,
                "profile_ylabel": "OD",
                "profiles": [
                    {"name": "Raw reference", "values": raw, "color": "red"},
                    {"name": "Used endmember", "values": used, "color": "blue"},
                ],
            }],
            "spectra": {"raw_reference": raw, "used_endmember": used, "unused_component": raw[::-1]},
            "mapping_spectra": {"PTEE_endmembers": {"used_endmember": used}},
            "warnings": [],
        }
        self.window = None
        self.destroyed = False
        self.release = threading.Event()

    def create_window(self):
        self.window = ReportWindow(self.payload)
        self.window.destroyed.connect(lambda: setattr(self, "destroyed", True))
        return self.window

    def wait_until(self, predicate, timeout=30):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.app.processEvents()
        self.assertTrue(predicate(), "Report window did not reach the expected state")

    def await_preview(self):
        self.wait_until(lambda: self.window.document.pageCount() > 0 and self.window.thread is None)
        self.assertIn("Preview ready", self.window.status.text())

    def tearDown(self):
        self.release.set()
        if self.window is not None and not self.destroyed:
            self.wait_until(lambda: self.window.thread is None)
            self.window.close()
            self.app.processEvents()
        self.temporary.cleanup()

    def test_preview_contains_parameters_and_step_then_navigates(self):
        window = self.create_window()
        self.await_preview()
        self.assertGreaterEqual(window.document.pageCount(), 2)
        first_page = window.document.getAllText(0).text()
        all_text = "\n".join(window.document.getAllText(i).text()
                             for i in range(window.document.pageCount()))
        self.assertIn("Dwell time", first_page)
        self.assertIn("I0 ROI", all_text)
        self.assertIn("OD", all_text)
        self.assertIn("Raw reference", all_text)
        self.assertIn("Used endmember", all_text)
        window.next.click()
        self.assertEqual(window.preview.pageNavigator().currentPage(), 1)
        self.assertTrue(window.previous.isEnabled())
        self.assertTrue(window.export_button.isEnabled())
        # Preview files belong to an internal temporary folder, not the scan.
        self.assertEqual(list(self.root.iterdir()), [])

    def test_csv_only_exports_all_and_exact_mapping_subset(self):
        window = self.create_window()
        self.await_preview()
        for name, checkbox in window._choices.items():
            checkbox.setChecked(name == "CSV")
        window.export_button.click()
        self.wait_until(lambda: window.thread is None)
        self.assertNotIn("Failed", window.status.text())
        paths = list(self.root.rglob("*"))
        files = [path for path in paths if path.is_file()]
        csv_files = [path for path in files if path.suffix == ".csv"]
        self.assertEqual(len(csv_files), 2)
        self.assertEqual(files, csv_files)
        self.assertEqual(set(window.exported_paths), set(csv_files))
        parsed = []
        for path in csv_files:
            with path.open(newline="", encoding="utf-8-sig") as handle:
                parsed.append(list(csv.reader(handle)))
        aggregate = next(rows for rows in parsed if "raw_reference" in rows[0])
        selected = next(rows for rows in parsed if "raw_reference" not in rows[0])
        self.assertEqual(set(aggregate[0]), {"energy_eV", "raw_reference", "used_endmember", "unused_component"})
        self.assertEqual(selected[0], ["energy_eV", "used_endmember"])
        self.assertEqual(len(selected), len(self.payload["energies"]) + 1)
        self.assertEqual(float(selected[-1][0]), 290.0)
        self.assertEqual(float(selected[-1][1]), 1.0)

    def test_no_selected_format_does_not_create_output(self):
        window = self.create_window()
        self.await_preview()
        for checkbox in window._choices.values():
            checkbox.setChecked(False)
        window.export_button.click()
        self.assertIsNone(window.thread)
        self.assertIn("Select at least one", window.status.text())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_preview_failure_releases_worker_and_csv_remains_available(self):
        with patch("muaxis.gui.report_window.export_pdf", side_effect=RuntimeError("test preview unavailable")):
            window = self.create_window()
            self.wait_until(lambda: "test preview unavailable" in window.status.text() and window.thread is None)
        self.assertTrue(window.export_button.isEnabled())
        self.assertEqual(window.document.pageCount(), 0)
        for name, checkbox in window._choices.items():
            checkbox.setChecked(name == "CSV")
        window.export_button.click()
        self.wait_until(lambda: window.thread is None)
        self.assertEqual(len(window.exported_paths), 2)
        self.assertTrue(all(path.suffix == ".csv" for path in window.exported_paths))
        self.assertNotIn("Failed", window.status.text())

    def test_close_during_preview_waits_for_worker_and_removes_temporary_files(self):
        entered = threading.Event()

        def delayed_pdf(*args, **kwargs):
            entered.set()
            if not self.release.wait(10):
                raise RuntimeError("Preview test release timed out")
            return export_pdf(*args, **kwargs)

        with patch("muaxis.gui.report_window.export_pdf", side_effect=delayed_pdf):
            window = self.create_window()
            preview_directory = Path(window._temporary.name)
            window.show()
            self.wait_until(entered.is_set)
            window.close()
            self.assertFalse(self.destroyed)
            self.assertTrue(window._close_pending)
            self.release.set()
            self.wait_until(lambda: self.destroyed)
        self.assertFalse(preview_directory.exists())
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
