import unittest

import compare_statsmodels


class StatsModelsEvidenceTests(unittest.TestCase):
    def test_regression_data_shapes_match_case_names(self) -> None:
        x, y, weights = compare_statsmodels.regression_data(2000)
        self.assertEqual(x.shape, (2000, 3))
        self.assertEqual(y.shape, (2000,))
        self.assertEqual(weights.shape, (2000,))

    def test_logit_data_has_both_classes(self) -> None:
        x, y = compare_statsmodels.logit_data(800)
        self.assertEqual(x.shape, (800, 2))
        self.assertGreater(y.sum(), 0)
        self.assertLess(y.sum(), len(y))


if __name__ == "__main__":
    unittest.main()
