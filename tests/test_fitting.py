import unittest
import numpy as np

from muaxis.processing.fitting import spectral_r2_map


class SinglePhaseAssignmentTests(unittest.TestCase):
    def setUp(self):
        self.E = np.linspace(280.0, 300.0, 40)
        peak = lambda c: np.exp(-0.5 * ((self.E - c) / 0.5) ** 2)
        # Three references map to the red, green and blue channels.
        self.refs = np.array([peak(285.0), peak(288.6), peak(290.4)])
        # Columns 0/1/2: pixels dominated by ref 0/1/2.  Column 3: an equal blend
        # of ref 0 and ref 1.  Column 4: noise (no phase).
        stack = np.zeros((40, 1, 5))
        stack[:, 0, 0] = peak(285.0)
        stack[:, 0, 1] = peak(288.6)
        stack[:, 0, 2] = peak(290.4)
        stack[:, 0, 3] = peak(285.0) + peak(288.6)
        stack[:, 0, 4] = np.random.RandomState(0).randn(40) * 0.01
        self.stack = stack

    @staticmethod
    def _active_phase_channels(result):
        # With three references there is no yellow channel, so the red/green/blue
        # channels of the map count how many phases a pixel is assigned to.
        return (result.rgb_y[..., :3] > 0).sum(axis=-1)

    def test_single_phase_assigns_one_reference_per_pixel(self):
        single = spectral_r2_map(self.stack, self.E, self.refs, r2_floor=0.0, single_phase=True)
        self.assertLessEqual(int(self._active_phase_channels(single).max()), 1)

    def test_blend_mode_keeps_more_channels_than_single_on_a_mixed_pixel(self):
        blend = spectral_r2_map(self.stack, self.E, self.refs, r2_floor=0.0, single_phase=False)
        single = spectral_r2_map(self.stack, self.E, self.refs, r2_floor=0.0, single_phase=True)
        # The blended pixel (column 3) matches two references in mixture mode but
        # exactly one in single-phase mode.
        self.assertGreater(int(self._active_phase_channels(blend)[0, 3]), 1)
        self.assertEqual(int(self._active_phase_channels(single)[0, 3]), 1)

    def test_below_floor_pixel_stays_unassigned(self):
        single = spectral_r2_map(self.stack, self.E, self.refs, r2_floor=0.9, single_phase=True)
        # Column 4 is noise; its best match is below the floor, so it is blank.
        self.assertEqual(float(single.rgb_y[0, 4, 3]), 0.0)

    def test_pure_pixel_is_a_pure_colour_in_single_phase(self):
        single = spectral_r2_map(self.stack, self.E, self.refs, r2_floor=0.9, single_phase=True)
        red = single.rgb_y[0, 0]  # column 0 -> ref 0 -> red only
        self.assertGreater(red[0], 0.9)
        self.assertEqual(red[1], 0.0)
        self.assertEqual(red[2], 0.0)


if __name__ == "__main__":
    unittest.main()
