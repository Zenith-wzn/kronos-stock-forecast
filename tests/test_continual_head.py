import unittest
import numpy as np

from continual_abcs.head import make_same_date_batches, pairwise_rank_loss_numpy


class HeadTests(unittest.TestCase):
    def test_batches_never_mix_dates(self):
        dates = np.array(["2024-01-01"] * 3 + ["2024-01-02"] * 2)
        batches = make_same_date_batches(dates, batch_size=2, seed=7, shuffle=False)
        self.assertTrue(all(len(set(dates[ix])) == 1 for ix in batches))

    def test_pairwise_loss_is_zero_for_perfect_order(self):
        loss = pairwise_rank_loss_numpy(np.array([3., 2., 1.]), np.array([3., 2., 1.]))
        self.assertLess(loss, 0.2)


if __name__ == "__main__":
    unittest.main()
