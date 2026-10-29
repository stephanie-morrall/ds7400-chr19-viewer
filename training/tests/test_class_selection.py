"""Checks that class eligibility comes only from training cells."""

import unittest

import pandas as pd

from training.class_selection import apply_class_selection, fit_class_selection


class ClassSelectionTests(unittest.TestCase):
    def test_other_meets_same_threshold_as_other_non_unknown_classes(self):
        names = {code: f"class {code}" for code in range(1, 9)}
        below = pd.Series([1] * 299 + [3] * 300 + [6] * 300 + [8] * 500)
        at_threshold = pd.concat([below, pd.Series([1])], ignore_index=True)
        below_selection = fit_class_selection(below, names)
        threshold_selection = fit_class_selection(at_threshold, names)
        self.assertEqual(below_selection["paper_code_to_target"], {"3": 0, "6": 1})
        self.assertEqual(threshold_selection["paper_code_to_target"],
                         {"1": 0, "3": 1, "6": 2})
        self.assertEqual(threshold_selection["excluded_paper_codes"], [8])
        features = pd.DataFrame({"distance": range(len(below))})
        _, _, counts = apply_class_selection(features, below, below_selection)
        self.assertEqual(counts["excluded_small_class"], 299)
        self.assertEqual(counts["excluded_unknown"], 500)

    def test_threshold_unknown_and_validation_mapping(self):
        names = {code: f"class {code}" for code in range(1, 9)}
        # The boundary values and a large Unknown group exercise both exclusion reasons.
        training = pd.Series([2] * 299 + [3] * 300 + [6] * 301 + [8] * 500 + [1] * 3)
        selection = fit_class_selection(training, names)
        self.assertEqual(selection["paper_code_to_target"], {"3": 0, "6": 1})
        self.assertEqual(selection["class_names"], ["class 3", "class 6"])
        validation = pd.Series([2] * 500 + [3, 6, 8, 1])
        features = pd.DataFrame({"distance": range(len(validation))})
        kept, mapped, counts = apply_class_selection(features, validation, selection)
        self.assertEqual(kept.index.tolist(), [500, 501])
        self.assertEqual(mapped.tolist(), [0, 1])
        self.assertEqual(counts["excluded_small_class"], 501)
        self.assertEqual(counts["excluded_unknown"], 1)
        # Applying validation data cannot alter the fitted mapping.
        self.assertEqual(selection["paper_code_to_target"], {"3": 0, "6": 1})


if __name__ == "__main__":
    unittest.main()
