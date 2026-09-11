import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
import unittest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from muaxis.gui.main_window import MainWindow


class RecentScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="muaxis-recent-test-")
        self.root = Path(self.temporary.name)
        self.window = MainWindow()
        self.window._recent_path = self.root / "recent.json"
        self.window.recent_list.clear()

    def tearDown(self):
        self.window.close()
        self.temporary.cleanup()

    def test_reopening_scan_keeps_selected_filename_sort(self):
        a = self.root / "a_scan.hdr"
        z = self.root / "z_scan.hdr"
        self.window._insert_recent_item(z)
        self.window._insert_recent_item(a)
        self.window.recent_sort_combo.setCurrentText("Filename")
        self.window._sort_recent_scans()

        self.window._add_recent_scan(z)

        paths = [
            self.window.recent_list.item(row).data(Qt.ItemDataRole.UserRole)
            for row in range(self.window.recent_list.count())
        ]
        self.assertEqual(paths, [str(a), str(z)])
        self.assertEqual(self.window.recent_list.currentItem().data(Qt.ItemDataRole.UserRole), str(z))
        self.assertEqual(self.window.recent_list.currentRow(), 1)


if __name__ == "__main__":
    unittest.main()
