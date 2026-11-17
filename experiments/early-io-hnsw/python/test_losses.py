"""Numerical and gradient checks for the node-ID-aware delay loss."""
import math
import unittest

import torch

from losses import weighted_bce_with_tp_delay


class DelayLossTests(unittest.TestCase):
    def example(self):
        probabilities = torch.tensor([[[.3, .8], [.9, .4], [.7, .6], [.8, .2]]], dtype=torch.double)
        ids = torch.tensor([[[10, 90], [90, 10], [10, 90], [90, 10]]])
        labels = (ids == 10).double()
        progress = torch.tensor([[.3, .9, 1.2, 1.6]], dtype=torch.double)
        return probabilities.logit().requires_grad_(), labels, progress, ids, torch.tensor([[10, 20]]), torch.tensor([4])

    def test_rank_changes_running_max_and_stored_progress(self):
        args = self.example()
        total, bce, delay = weighted_bce_with_tp_delay(*args, delay_weight=2)
        self.assertAlmostEqual(delay.item(), (.4 + .2) / 8)
        p = args[0].sigmoid(); y = args[1]; sp = args[2].unsqueeze(-1)
        expected = (-y*p.log() - (1+sp)*(1-y)*(1-p).log()).mean()
        self.assertTrue(torch.allclose(bce, expected))
        self.assertTrue(torch.allclose(total, bce + 2*delay))
        gradient = torch.autograd.grad(delay, args[0])[0]
        self.assertTrue(torch.all(gradient[args[3] == 90] == 0))
        self.assertLess(gradient[0, 0, 0].item(), 0)
        self.assertEqual(gradient[0, 3, 1].item(), 0)

    def test_first_entry_padding_and_equal_query_average(self):
        ids = torch.tensor([[[90, -1], [10, 90], [90, 10]], [[10, -1], [10, 90], [10, 90]]])
        p = torch.tensor([[[.8, .99], [.2, .8], [.8, .4]], [[.3, .99], [.99, .9], [.99, .9]]], dtype=torch.double)
        labels = (ids == 10).double()
        args = (p.logit(), labels, torch.zeros(2, 3, dtype=torch.double), ids,
                torch.tensor([[10, 20], [10, 20]]), torch.tensor([3, 1]))
        total, bce, delay = weighted_bce_with_tp_delay(*args)
        self.assertAlmostEqual(delay.item(), ((.6+.2)/6 + .4/2)/2)
        individual = []
        for b in range(2):
            individual.append(weighted_bce_with_tp_delay(*(x[b:b+1] for x in args))[0])
        self.assertTrue(torch.allclose(total, torch.stack(individual).mean()))
        # Invalid -1 slots and padded timesteps do not change either component.
        changed = list(args)
        changed[0] = args[0].clone()
        changed[0][ids < 0] = -20
        changed[0][1, 1:] = -20
        after = weighted_bce_with_tp_delay(*changed)
        for a, b in zip((total, bce, delay), after):
            self.assertTrue(torch.allclose(a, b))

    def test_autograd(self):
        args = self.example()
        self.assertTrue(torch.autograd.gradcheck(
            lambda x: weighted_bce_with_tp_delay(x, *args[1:])[0], (args[0],)))

    def test_weight_zero_and_invalid_settings(self):
        args = self.example()
        total, bce, _ = weighted_bce_with_tp_delay(*args, delay_weight=0)
        self.assertTrue(torch.equal(total, bce))
        for theta in (0, -1, 2, math.nan):
            with self.assertRaises(ValueError):
                weighted_bce_with_tp_delay(*args, threshold=theta)
        with self.assertRaises(ValueError):
            weighted_bce_with_tp_delay(*args, delay_weight=-1)


if __name__ == "__main__":
    unittest.main()
