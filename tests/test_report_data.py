import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from dataclasses import replace
import numpy as np

from muaxis.processing.fitting import spectral_r2_map
from muaxis.processing.multivariate import MultivariateResult
from muaxis.report_data import build_report_payload, result_from_record, result_to_record


class ReportDataTests(unittest.TestCase):
    def window(self):
        e = np.array([280., 281., 282., 283., 284.])
        stack = np.arange(30., dtype=float).reshape(5, 2, 3) + 1
        scan = NS(energies_eV=e, transmission=stack, shape=stack.shape,
                  header=NS(path=Path("/test/scan.hdr"), label="sample", scan_type="Image stack",
                            frames=[NS(acquired_at="2026-09-10")], x_um=np.arange(3.), y_um=np.arange(2.), dwell_time_ms=10))
        return NS(scan=scan, _current_frame=2, _spectrum_rois=[],
                  _od_result=NS(optical_density=stack / 10, i0=np.ones(5) * 50),
                  _od_roi_bounds=[0, 0, 1, 1], _registered_stack=stack / 20,
                  _registration_config={"reference_index": 1}, _premap_maps={}, _premap_config=None,
                  _segmentation_config=None, _multivariate_result=None, _fitting_result=None)

    def test_registered_image_profiles_and_measurement_are_actual_values(self):
        window = self.window()
        payload = build_report_payload(window)
        registered = next(layer for layer in payload["layers"] if layer["name"] == "Registered OD")
        np.testing.assert_array_equal(registered["image"], window._registered_stack[2])
        np.testing.assert_allclose(registered["profiles"][0]["values"], window._registered_stack.mean(axis=(1, 2)))
        self.assertIn("I0 — direct beam", payload["spectra"])
        self.assertEqual(payload["metadata"]["measurement_parameters"]["Image size (x, y pixels)"], [3, 2])

    def test_mapping_export_uses_normalized_actual_active_endmember_after_restart(self):
        window = self.window()
        refs = np.array([[2., 4., 8., 6., 3.], [np.nan]*5, [np.nan]*5, [np.nan]*5])
        result = spectral_r2_map(window._od_result.optical_density, window.scan.energies_eV, refs)
        result = replace(result, raw_references=refs.copy(), analysis_parameters={
            "profile_normalization": "Min–max", "channels": {"R": {"label": "Label A", "cluster_id": "2", "active": True}}})
        record = json.loads(json.dumps(result_to_record(result, "fitting")))
        self.assertNotIn("r2", record)
        window._fitting_result = result_from_record(record, "fitting")
        payload = build_report_payload(window)
        used = payload["mapping_spectra"]["PTEE_endmembers"]
        self.assertEqual(len(used), 1)
        name, values = next(iter(used.items()))
        self.assertIn("Label A / cluster 2", name)
        np.testing.assert_allclose(values, (refs[0] - 2) / 6)
        np.testing.assert_equal(payload["spectra"][name.replace("mapping endmember", "raw reference")], refs[0])

    def test_multivariate_all_components_and_selected_mapping_components_survive(self):
        window = self.window()
        components = np.eye(5)
        result = MultivariateResult("PCA (centered)", components, np.ones((6, 5)),
                                    np.ones((5, 2, 3)), np.ones(5), np.ones(5)/5,
                                    np.ones((2, 3, 4)), {"rgb_coefficients": [[1, 0, 0, 0, 0], [0, 0, 0, 2, 0], [0, 0, 0, 0, 0]], "linear_combination_iterations": 10})
        record = json.loads(json.dumps(result_to_record(result, "multivariate")))
        self.assertNotIn("scores", record)
        self.assertNotIn("reconstructed", record)
        window._multivariate_result = result_from_record(record, "multivariate")
        payload = build_report_payload(window)
        used = payload["mapping_spectra"]["Multivariate_mapping_components"]
        self.assertEqual(set(used), {"PCA (centered) — component 1", "PCA (centered) — component 4"})
        self.assertEqual(len([key for key in payload["spectra"] if "— component" in key]), 5)

    def test_old_maps_do_not_crash_or_invent_spectra(self):
        window = self.window()
        window._fitting_result = result_from_record({"rgb_y": np.ones((2, 3, 4)).tolist()}, "fitting")
        window._multivariate_result = result_from_record({"rgb_map": np.ones((2, 3, 4)).tolist()}, "multivariate")
        window._segmentation_config = {"saved_groups": {"Old label": {"profiles": [[1, 2, 3, 4, 5]], "cluster_ids": [9], "areas": [20]}}}
        payload = build_report_payload(window)
        self.assertFalse(payload["mapping_spectra"])
        self.assertGreaterEqual(len(payload["warnings"]), 3)
        segmentation = next(layer for layer in payload["layers"] if "Segmentation" in layer["name"])
        self.assertIsNone(segmentation["image"])
        self.assertEqual(segmentation["profiles"][0]["name"], "Old label — cluster 9")

    def test_condition_tables_exclude_stacks_and_saved_label_image_is_used(self):
        window = self.window()
        window._analysis_stack = window._od_result.optical_density / 2
        window._premap_config = {"denoised_stack": window._analysis_stack.tolist(), "source": "OD", "multivariate": {"method": "PCA", "components": 2, "clip_negative": True}}
        window._premap_maps = {"Peak map (OD)": np.ones((2, 3))}
        window._segmentation_config = {"saved_groups": {"Saved": {"profiles": [[1, 2, 3, 4, 5]], "cluster_ids": [2], "areas": [2], "labels": [[0, 2, 0], [0, 2, 0]], "analysis_parameters": {"source": "Peak map (OD)", "threshold": [281, 282]}}}}
        payload = build_report_payload(window)
        parameters = json.dumps(payload["metadata"]["analysis_parameters"])
        self.assertNotIn("denoised_stack", parameters)
        segmentation = next(layer for layer in payload["layers"] if "Segmentation" in layer["name"])
        self.assertEqual(segmentation["image"].shape, (2, 3, 3))
        self.assertTrue(np.all(segmentation["image"][:, 1] != 0))
        self.assertEqual(segmentation["parameters"]["source"], "Peak map (OD)")


if __name__ == "__main__":
    unittest.main()
