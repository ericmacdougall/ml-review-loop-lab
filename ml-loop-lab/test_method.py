"""Controls for threshold calibration and paired reviewer randomness."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
import signal_thresholds as signals
import loop
from freeze import verify_freeze, validate_runs, sha
import tempfile
import json


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

    def test_freeze_refuses_data_code_and_hyperparameter_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"data.bin").write_bytes(b"fixed data")
            (root/"code.py").write_bytes(b"fixed code")
            manifest = {"settings": {"C_HEAD": 100}, "data": {"data.bin": sha(root/"data.bin")},
                        "implementation": {"code.py": sha(root/"code.py")}}
            (root/"frozen_inputs.json").write_text(json.dumps(manifest))
            verify_freeze(root, 100, root)
            with self.assertRaises(SystemExit):
                verify_freeze(root, 1, root)
            for file in ("data.bin", "code.py"):
                original = (root/file).read_bytes()
                (root/file).write_bytes(b"changed")
                with self.assertRaises(SystemExit):
                    verify_freeze(root, 100, root)
                (root/file).write_bytes(original)

    def test_analysis_refuses_incomplete_or_duplicate_seeds(self):
        with self.assertRaisesRegex(ValueError, "matrix"):
            validate_runs([])
        cfg = loop.configs("main", [100], True)[0]
        with self.assertRaisesRegex(ValueError, "matrix"):
            validate_runs([{"cfg": cfg}, {"cfg": cfg}])


if __name__ == "__main__":
    unittest.main()
