import unittest

import torch
from torch import nn

from rangerlite.rangerlite import RangerLite


class RangerTest(unittest.TestCase):
    @staticmethod
    def _step(optimizer, x):
        optimizer.zero_grad()
        loss = torch.sum(x**2)
        loss.backward()
        optimizer.step()

    def test_optimizer_step(self):
        x = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        x_init = x.clone()
        betas = (0.95, 0.999)
        optimizer = RangerLite(
            params=[x],
            lr=0.5,
            betas=betas,
            lookahead_alpha=0.5,
            lookahead_steps=2,
        )
        self._step(optimizer, x)
        self.assertTrue(
            expr=torch.allclose(x, torch.zeros_like(x), atol=1e-5),
            msg="First step should behave like a vanilla SGD step",
        )
        self.assertAlmostEqual(
            first=optimizer.state[x]["step"].item(),
            second=1.0,
            msg="Lookahead step should be 1 after first step",
        )
        self.assertTrue(
            expr=torch.allclose(optimizer.state[x]["cached_params"], x_init, atol=1e-5),
            msg="Cached parameters should match initial parameters",
        )
        self._step(optimizer, x)
        inter_result = -x_init * betas[0] / (1.0 + betas[0])  # expected result of RAdam
        expectation = 0.5 * (x_init + inter_result)  # expected result of Lookahead
        self.assertTrue(
            expr=torch.allclose(x, expectation, atol=1e-5),
            msg=f"Lookahead step hailed. got: {x.tolist()}, expected: {expectation.tolist()}",
        )
        self.assertAlmostEqual(
            first=optimizer.state[x]["step"].item(),
            second=2.0,
            msg="Lookahead step should be 2 after second step",
        )
        self._step(optimizer, x)
        self.assertAlmostEqual(
            first=optimizer.state[x]["step"].item(),
            second=3.0,
            msg="Lookahead step should be 3 after third step",
        )
        self.assertIs(optimizer.state[x]["cached_params"].grad_fn, None)

    def test_state_dict(self):
        x = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        optimizer = RangerLite([x], lr=0.5, lookahead_alpha=0.5, lookahead_steps=2)
        self._step(optimizer, x)
        state_dict = optimizer.state_dict()
        optimizer2 = RangerLite([x])
        optimizer2.load_state_dict(state_dict)
        for state_key, state_value in optimizer.state[x].items():
            loaded_value = optimizer2.state[x][state_key]
            self.assertEqual(
                first=state_value.shape,
                second=loaded_value.shape,
                msg=f"State shape mismatch for {state_key}",
            )
            self.assertEqual(
                first=state_value.dtype,
                second=loaded_value.dtype,
                msg=f"State dtype mismatch for {state_key}",
            )
            self.assertTrue(
                torch.allclose(state_value, loaded_value),
                msg=f"State mismatch for {state_key}",
            )
        for param_key, param_value in optimizer2.param_groups[0].items():
            loaded_value = optimizer2.param_groups[0][param_key]
            if isinstance(param_value, torch.Tensor):
                self.assertTrue(
                    torch.allclose(param_value, loaded_value),
                    msg=f"Param group state mismatch for {param_key}",
                )
            else:
                self.assertEqual(
                    first=type(param_value),
                    second=type(loaded_value),
                    msg=f"Param group state mismatch for {param_key}",
                )
                self.assertEqual(
                    first=param_value,
                    second=loaded_value,
                    msg=f"Param group state mismatch for {param_key}",
                )


if __name__ == "__main__":
    unittest.main()
