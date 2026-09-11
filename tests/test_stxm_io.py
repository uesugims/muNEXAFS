from pathlib import Path
import tempfile
import unittest

import numpy as np

from muaxis.io.stxm import StxmFormatError, read_stxm_scan


def _write_scan(directory: Path) -> Path:
    header = directory / "demo.hdr"
    header.write_text(
        '''ScanDefinition = { Label = "demo.hdr"; Type = "NEXAFS Image Scan";\n'''
        '''PAxis = { Points = (2, 0.0, 1.0); };\n'''
        '''QAxis = { Points = (2, 0.0, 1.0); };\n'''
        '''StackAxis = { Points = (2, 280.0, 281.0); };\n'''
        '''Image000_0 = {StorageRingCurrent = 0.00; Energy = 280.00; Time = "2020 Jan 01";};\n'''
        '''Image001_0 = {StorageRingCurrent = 300.00; Energy = 281.00; Time = "2020 Jan 01";};\n''',
        encoding="utf-8",
    )
    (directory / "demo_a000.xim").write_text("1\t2\t\n3\t4\t\n", encoding="utf-8")
    np.savetxt(directory / "demo_a001.xim", [[5, 6], [7, 8]], delimiter="\t", fmt="%d")
    (directory / "i0.txt").write_text("10,0\n20,0\n", encoding="utf-8")
    (directory / "drift.txt").write_text("0.0,1.0\n2.0,3.0\n", encoding="utf-8")
    return header


class StxmIoTests(unittest.TestCase):
    def test_reads_metadata_images_and_companions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scan = read_stxm_scan(_write_scan(Path(directory)))

        self.assertEqual(scan.shape, (2, 2, 2))
        self.assertEqual(scan.energies_eV.tolist(), [280.0, 281.0])
        self.assertEqual(scan.header.x_um.tolist(), [0.0, 1.0])
        assert scan.transmission is not None
        self.assertEqual(scan.transmission[1, 1, 1], 8)
        assert scan.i0 is not None
        self.assertEqual(scan.i0.tolist(), [10.0, 20.0])
        assert scan.shifts_xy is not None
        self.assertEqual(scan.shifts_xy.tolist(), [[0.0, 1.0], [2.0, 3.0]])

    def test_reports_missing_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            header = _write_scan(path)
            (path / "demo_a001.xim").unlink()

            with self.assertRaisesRegex(StxmFormatError, "Missing 1 frame"):
                read_stxm_scan(header)
