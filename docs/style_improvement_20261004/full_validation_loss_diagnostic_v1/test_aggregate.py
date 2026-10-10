"""Pure scalar callbacks: coverage and original batch-mean aggregation only."""
import unittest

from evaluate_full_validation import aggregate, ranges


class AggregationTests(unittest.TestCase):
    def test_tail_weighting_matches_complete_population_for_any_batch(self):
        values = [float(index) for index in range(11)]
        for batch in (1, 4, 8, 32):
            rows = [{"start": start, "count": count,
                     "loss": sum(values[start:start + count]) / count}
                    for start, count in ranges(len(values), batch)]
            result = aggregate(len(values), rows)
            self.assertEqual(result["windows"], len(values))
            self.assertEqual(result["weighted_loss"], sum(values) / len(values))
        self.assertEqual(ranges(10566, 32)[-1], (10560, 6))
        self.assertEqual(ranges(19386, 32)[-1], (19360, 26))

    def test_missing_overlap_nonfinite_and_bool_counts_rejected(self):
        for rows in ([{"start": 0, "count": 4, "loss": 1.}],
                     [{"start": 0, "count": 4, "loss": 1.}, {"start": 3, "count": 2, "loss": 1.}],
                     [{"start": 0, "count": 6, "loss": float("nan")}],
                     [{"start": 0, "count": True, "loss": 1.}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                aggregate(6, rows)


if __name__ == "__main__":
    unittest.main()
