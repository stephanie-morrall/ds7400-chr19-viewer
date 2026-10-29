"""Checks quality thresholds, rank interpolation and unscaled distance inputs."""

import json
import unittest

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import torch

from training.cleaning import _fill_trace, clean_spots, fit_cleaning
from training.features import features_for_cells
from training.prepare import feature_tensor


def spot_table(traces):
    """Creates measured trace rows whose coordinates depend on genomic rank."""
    rows = []
    for cell, trace, loci in traces:
        rows.extend([cell, trace, locus, float(locus), 2.0 * locus, -float(locus)]
                    for locus in loci)
    return pd.DataFrame(rows, columns=["Cell_ID", "Trace_ID", "Chrom_Start", "X", "Y", "Z"])


def full_protocol(loci):
    """Fits a complete training example for tests of later quality decisions."""
    complete = spot_table([(1, 1, loci)])
    return fit_cleaning({1: complete, 2: complete})


class CleaningTests(unittest.TestCase):
    def test_filter_a_keeps_exact_half_and_drops_over_half_missing(self):
        first = spot_table([(1, 1, [100, 200, 300]), (2, 1, [100, 200])])
        second = spot_table([(3, 1, [100, 200, 300]), (4, 1, [100, 200])])
        protocol = fit_cleaning({1: first, 2: second})
        self.assertEqual(protocol["retained_loci"], [100, 200, 300])
        # One measured trace out of four gives 75% missingness in replicate 2.
        second = pd.concat([second, spot_table([(5, 1, [100, 200]), (6, 1, [100, 200])])])
        protocol = fit_cleaning({1: first, 2: second})
        self.assertEqual(protocol["retained_loci"], [100, 200])
        self.assertEqual(protocol["training_locus_missingness"]["2"]["300"], 0.75)
        json.dumps(protocol)

    def test_filter_a_absent_locus_is_completely_missing(self):
        first = spot_table([(1, 1, [100, 200, 300])])
        second = spot_table([(2, 1, [100, 200])])
        protocol = fit_cleaning({1: first, 2: second})
        self.assertEqual(protocol["retained_loci"], [100, 200])
        self.assertEqual(protocol["training_locus_missingness"]["2"]["300"], 1.0)

    def test_filter_b_removes_entire_five_trace_cell(self):
        protocol = full_protocol(list(range(5)))
        traces = [(1, trace, list(range(5))) for trace in range(5)]
        traces += [(2, trace, list(range(5))) for trace in range(4)]
        filled, counts = clean_spots(spot_table(traces), protocol)
        self.assertEqual(filled["Cell_ID"].unique().tolist(), [2])
        self.assertEqual(counts["raw"]["traces"], 9)
        self.assertEqual(counts["after_filter_b"]["traces"], 4)

    def test_filter_c_uses_measured_coverage_before_interpolation(self):
        protocol = full_protocol(list(range(5)))
        spots = spot_table([(1, 1, [0, 1, 3, 4]), (2, 1, [0, 2, 4])])
        filled, counts = clean_spots(spots, protocol)
        self.assertEqual(counts["minimum_observed_loci"], 4)
        self.assertEqual(counts["after_filter_c"]["cells"], 1)
        self.assertEqual(filled["Cell_ID"].unique().tolist(), [1])
        self.assertEqual(len(filled), 5)
        self.assertEqual(filled["is_imputed"].tolist(), [False, False, True, False, False])

    def test_full_raw_axis_anchors_survive_until_interpolation_finishes(self):
        protocol = full_protocol(list(range(6)))
        protocol["retained_loci"] = [0, 1, 3, 4, 5]
        spots = spot_table([(1, 1, [0, 1, 2, 4, 5])])
        spots.loc[spots["Chrom_Start"].eq(2), "X"] = 100.0
        spots.loc[spots["Chrom_Start"].eq(4), "X"] = 0.0
        filled, _ = clean_spots(spots, protocol)
        self.assertNotIn(2, filled["Chrom_Start"].tolist())
        self.assertEqual(float(filled.set_index("Chrom_Start").loc[3, "X"]), 50.0)
        measured = filled.loc[~filled["is_imputed"]].set_index("Chrom_Start")
        original = spots.set_index("Chrom_Start").loc[measured.index]
        np.testing.assert_array_equal(measured[["X", "Y", "Z"]], original[["X", "Y", "Z"]])

    def test_both_endpoints_extrapolate_on_rank_and_match_research_operation(self):
        # Uneven physical locus labels still receive equal-spaced integer ranks.
        loci = [1, 10, 20, 30, 40, 50, 60, 70, 80, 1000]
        protocol = full_protocol(loci)
        spots = spot_table([(1, 1, loci[1:-1])])
        ranks = np.arange(1, 9)
        spots[["X", "Y", "Z"]] = np.column_stack([2 * ranks, 3 * ranks, -ranks])
        filled, _ = clean_spots(spots, protocol)
        expected = np.column_stack([2 * np.arange(10), 3 * np.arange(10), -np.arange(10)])
        np.testing.assert_allclose(filled[["X", "Y", "Z"]], expected)
        xyz = np.full((10, 3), np.nan)
        xyz[1:9] = spots[["X", "Y", "Z"]]
        research = interp1d(ranks, xyz[1:9], axis=0, fill_value="extrapolate")
        np.testing.assert_allclose(_fill_trace(xyz), research(np.arange(10)))

    def test_translation_and_rotation_preserve_mean_distance_features(self):
        protocol = full_protocol([0, 1, 2, 3, 4])
        spots = spot_table([(1, 1, [0, 1, 3, 4]), (1, 2, [0, 1, 2, 3, 4])])
        features, _ = features_for_cells(spots, protocol)
        transformed = spots.copy()
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        transformed[["X", "Y", "Z"]] = spots[["X", "Y", "Z"]].to_numpy() @ rotation
        transformed[["X", "Y", "Z"]] += [10, -20, 4]
        same, _ = features_for_cells(transformed, protocol)
        np.testing.assert_allclose(features, same)
        self.assertEqual(features.shape, (1, 10))

    def test_feature_tensor_preserves_physical_distance_scale(self):
        features = pd.DataFrame({"pair_a": [1.0, 100.0], "pair_b": [5.0, 500.0]})
        values = feature_tensor(features)
        self.assertEqual(values.dtype, torch.float32)
        np.testing.assert_array_equal(values.numpy(), features.to_numpy())
        with self.assertRaisesRegex(ValueError, "finite"):
            feature_tensor(pd.DataFrame({"distance": [np.nan]}))

    def test_empty_quality_result_keeps_feature_schema(self):
        protocol = full_protocol(list(range(5)))
        features, counts = features_for_cells(spot_table([(1, 1, [0, 1])]), protocol)
        self.assertTrue(features.empty)
        self.assertEqual(len(features.columns), 10)
        self.assertEqual(counts["after_filter_c"]["traces"], 0)

    def test_unknown_loci_and_duplicate_measurements_are_rejected(self):
        protocol = full_protocol([0, 1, 2])
        spots = spot_table([(1, 1, [0, 1, 2])])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            clean_spots(pd.concat([spots, spots.iloc[:1]]), protocol)
        with self.assertRaisesRegex(ValueError, "outside"):
            clean_spots(spot_table([(1, 1, [0, 1, 999])]), protocol)


if __name__ == "__main__":
    unittest.main()
