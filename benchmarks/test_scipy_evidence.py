import unittest

import compare_scipy


class SciPyEvidenceTests(unittest.TestCase):
    def test_integration_data_shapes_match_case_names(self) -> None:
        x, y, dx = compare_scipy.integration_data(10_001)
        self.assertEqual(x.shape, (10_001,))
        self.assertEqual(y.shape, (10_001,))
        self.assertGreater(dx, 0.0)

    def test_benchmark_functions_have_expected_signs(self) -> None:
        self.assertLess(compare_scipy.root_function(1.0), 0.0)
        self.assertGreater(compare_scipy.root_function(2.0), 0.0)
        self.assertLess(compare_scipy.scipy_asv_f2(0.5), 0.0)
        self.assertGreater(compare_scipy.scipy_asv_f2(3.0**0.5), 0.0)
        self.assertLess(compare_scipy.objective(1.2345), compare_scipy.objective(0.0))


if __name__ == "__main__":
    unittest.main()
