"""Controls for threshold calibration and paired reviewer randomness."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
import signal_thresholds as signals
import loop


class MethodControls(unittest.TestCase):
    def test_precision_target_must_hold_for_entire_tied_group(self):
        scores = np.array([0.9, 0.9, 0.2])
        y = np.array([1, 0, 1])
        threshold = signals.pick(scores, y, prec=0.75)
        self.assertTrue(np.isinf(threshold), "no realizable threshold reaches 75% precision")

    def test_feasible_precision_target_keeps_maximum_coverage(self):
        scores = np.array([0.9, 0.9, 0.2, 0.1])
        y = np.array([1, 1, 0, 0])
        threshold = signals.pick(scores, y, prec=0.6)
        self.assertEqual(threshold, 0.2)
        self.assertGreaterEqual(signals.score_at(scores, y, threshold)["precision"], 0.6)

    def test_reviewer_noise_is_paired_by_item_not_selection_order(self):
        labels = np.arange(200) % 2
        state = {"ypool": labels}
        hard = np.linspace(0, 1, 200)
        draws = loop.reviewer_draws(104, 200)
        left = loop.review(np.array([3, 8, 19]), state, hard, draws, "adjudicate", 1)
        right = loop.review(np.array([19, 3, 8]), state, hard, draws, "adjudicate", 1)
        for a, b in zip(left, right):
            np.testing.assert_equal(a, b[[1, 2, 0]])

    def test_weekly_arrivals_exhaust_pool_without_reuse(self):
        arrivals = [set(range((w-1)*loop.WEEKLY_ARRIVALS, w*loop.WEEKLY_ARRIVALS))
                    for w in range(1, loop.WEEKS+1)]
        self.assertEqual(sum(map(len, arrivals)), loop.N_POOL)
        self.assertEqual(len(set.union(*arrivals)), loop.N_POOL)


if __name__ == "__main__":
    unittest.main()
