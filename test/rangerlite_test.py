import torch
from torch import nn

from rangerlite import RangerLite


def _step(optimizer, x):
    optimizer.zero_grad()
    loss = torch.sum(x**2)
    loss.backward()
    optimizer.step()


def test_optimizer_step():
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
    _step(optimizer, x)
    assert torch.allclose(x, torch.zeros_like(x), atol=1e-5), (
        "First step should behave like a vanilla SGD step"
    )
    assert abs(optimizer.state[x]["step"].item() - 1.0) < 1e-7, (
        "Lookahead step should be 1 after first step"
    )
    assert torch.allclose(optimizer.state[x]["cached_params"], x_init, atol=1e-5), (
        "Cached parameters should match initial parameters"
    )
    _step(optimizer, x)
    inter_result = -x_init * betas[0] / (1.0 + betas[0])  # expected result of RAdam
    expectation = 0.5 * (x_init + inter_result)  # expected result of Lookahead
    assert torch.allclose(x, expectation, atol=1e-5), (
        f"Lookahead step hailed. got: {x.tolist()}, expected: {expectation.tolist()}"
    )
    assert abs(optimizer.state[x]["step"].item() - 2.0) < 1e-7, (
        "Lookahead step should be 2 after second step"
    )
    _step(optimizer, x)
    assert abs(optimizer.state[x]["step"].item() - 3.0) < 1e-7, (
        "Lookahead step should be 3 after third step"
    )
    assert optimizer.state[x]["cached_params"].grad_fn is None


def test_state_dict():
    x = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    optimizer = RangerLite([x], lr=0.5, lookahead_alpha=0.5, lookahead_steps=2)
    _step(optimizer, x)
    state_dict = optimizer.state_dict()
    optimizer2 = RangerLite([x])
    optimizer2.load_state_dict(state_dict)
    for state_key, state_value in optimizer.state[x].items():
        loaded_value = optimizer2.state[x][state_key]
        assert state_value.shape == loaded_value.shape, (
            f"State shape mismatch for {state_key}"
        )
        assert state_value.dtype == loaded_value.dtype, (
            f"State dtype mismatch for {state_key}"
        )
        assert torch.allclose(state_value, loaded_value), (
            f"State mismatch for {state_key}"
        )
    for param_key, param_value in optimizer.param_groups[0].items():
        loaded_value = optimizer2.param_groups[0][param_key]
        if isinstance(param_value, torch.Tensor):
            assert torch.allclose(param_value, loaded_value), (
                f"Param group state mismatch for {param_key}"
            )
        else:
            assert type(param_value) is type(loaded_value), (
                f"Param group state mismatch for {param_key}"
            )
            assert param_value == loaded_value, (
                f"Param group state mismatch for {param_key}"
            )
