import unittest

import numpy as np

from muaxis.processing.absorbance import compute_optical_density


class AbsorbanceTests(unittest.TestCase):
    def test_od_broadcasts_per_energy_i0(self) -> None:
        transmission = np.asarray([[[5.0, 10.0]], [[2.0, 4.0]]])
        result = compute_optical_density(transmission, np.asarray([10.0, 8.0]))

        np.testing.assert_allclose(
            result.optical_density,
            np.log(np.asarray([[[2.0, 1.0]], [[4.0, 2.0]]])),
        )
        self.assertTrue(result.valid_mask.all())

    def test_invalid_input_becomes_nan(self) -> None:
        result = compute_optical_density(np.asarray([[[0.0]]]), np.asarray([1.0]))

        self.assertTrue(np.isnan(result.optical_density[0, 0, 0]))
        self.assertFalse(result.valid_mask[0, 0, 0])
