"""Tests for the softmax implementation."""

import unittest

import numpy as np

from softmax import softmax


class SoftmaxTests(unittest.TestCase):
    def test_weights_sum_to_one(self) -> None:
        weights = softmax([1.0, 2.0, 3.0])

        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertTrue(np.all(weights >= 0.0))
        self.assertTrue(np.all(weights <= 1.0))

    def test_matches_known_distribution(self) -> None:
        weights = softmax([1.0, 2.0, 3.0])

        np.testing.assert_allclose(
            weights,
            [0.09003057, 0.24472847, 0.66524096],
            rtol=1e-7,
        )

    def test_weights_produce_expected_weighted_average(self) -> None:
        values = np.array([10.0, 20.0, 30.0])
        weights = softmax([1.0, 2.0, 3.0])

        self.assertAlmostEqual(float(np.sum(weights * values)), 25.7521, places=4)

    def test_large_logits_are_numerically_stable(self) -> None:
        weights = softmax([10_000.0, 10_001.0, 10_002.0])

        np.testing.assert_allclose(
            weights,
            [0.09003057, 0.24472847, 0.66524096],
            rtol=1e-7,
        )
        self.assertTrue(np.all(np.isfinite(weights)))

    def test_normalizes_each_row_by_default(self) -> None:
        weights = softmax([[1.0, 2.0], [3.0, 4.0]])

        np.testing.assert_allclose(weights.sum(axis=1), [1.0, 1.0])

    def test_can_normalize_a_different_axis(self) -> None:
        weights = softmax([[1.0, 2.0], [3.0, 4.0]], axis=0)

        np.testing.assert_allclose(weights.sum(axis=0), [1.0, 1.0])

    def test_rejects_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            softmax([])

    def test_rejects_scalar_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one dimension"):
            softmax(1.0)

    def test_rejects_non_finite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "only finite values"):
            softmax([1.0, np.inf])


if __name__ == "__main__":
    unittest.main()
