import tempfile
import unittest
from pathlib import Path
import numpy as np

from muaxis.reporting import export_layer_images, export_pdf, export_spectra_csv


class ReportingTests(unittest.TestCase):
    def test_csv_pdf_and_layer_images_are_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layers = [{"name": "Optical density (OD)", "image": np.arange(12).reshape(3, 4)}]
            images = export_layer_images(layers, root / "layers")
            csv_path = export_spectra_csv({"component_1": np.arange(3.)}, np.arange(3.), root / "spectra.csv")
            pdf_path = export_pdf(layers, {"title": "test"}, root, "report.pdf")
            self.assertTrue(images[0].exists())
            self.assertTrue(csv_path.exists())
            self.assertTrue(pdf_path.exists())
            self.assertGreater(pdf_path.stat().st_size, 100)
            self.assertIn("energy_eV,component_1", csv_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
