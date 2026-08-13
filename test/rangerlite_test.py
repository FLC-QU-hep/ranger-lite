import pytest
import torch
from torch.nn import Parameter
from torch.optim import Optimizer

from rangerlite import RangerLite


def _step(optimizer: Optimizer, x: Parameter) -> None:
    optimizer.zero_grad()
    loss = torch.sum(x**2)
    loss.backward()
    optimizer.step()


def test_optimizer_step() -> None:
    x = Parameter(torch.tensor([1.0, 2.0, 3.0]))
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


def test_lookahead_steps_one() -> None:
    x = Parameter(torch.tensor([1.0, 2.0, 3.0]))
    x_init = x.clone()
    optimizer = RangerLite(params=[x], lr=0.5, lookahead_alpha=0.5, lookahead_steps=1)
    _step(optimizer, x)
    expectation = (
        0.5 * x_init
    )  # blend of RAdam's zero first step and the cached initial params
    assert torch.allclose(x, expectation, atol=1e-5), (
        f"got: {x.tolist()}, expected: {expectation.tolist()}"
    )
    assert torch.allclose(optimizer.state[x]["cached_params"], x, atol=1e-5), (
        "Cached parameters should be updated to the blended result every step"
    )
    _step(optimizer, x)
    assert torch.allclose(optimizer.state[x]["cached_params"], x, atol=1e-5), (
        "Cached parameters should keep tracking the parameter every step"
    )


def test_multiple_param_groups_use_independent_lookahead_settings() -> None:
    x1 = Parameter(torch.tensor([1.0, 2.0, 3.0]))
    x2 = Parameter(torch.tensor([1.0, 2.0, 3.0]))
    x1_init = x1.clone()
    x2_init = x2.clone()
    optimizer = RangerLite(
        params=[
            {"params": [x1], "lookahead_steps": 1},
            {"params": [x2], "lookahead_steps": 3},
        ],
        lr=0.5,
        lookahead_alpha=0.5,
    )
    optimizer.zero_grad()
    loss = torch.sum(x1**2) + torch.sum(x2**2)
    loss.backward()
    optimizer.step()
    assert torch.allclose(x1, 0.5 * x1_init, atol=1e-5), (
        "Group with lookahead_steps=1 should blend on the first step"
    )
    assert torch.allclose(x2, torch.zeros_like(x2), atol=1e-5), (
        "Group with lookahead_steps=3 should not blend on the first step"
    )
    assert torch.allclose(optimizer.state[x2]["cached_params"], x2_init, atol=1e-5), (
        "Unblended group should keep its initial cached parameters"
    )


def test_weight_decay_shrinks_parameters_more() -> None:
    def run(weight_decay: float) -> Parameter:
        x = Parameter(torch.tensor([1.0, 2.0, 3.0]))
        optimizer = RangerLite(
            [x], lr=0.1, lookahead_steps=100, weight_decay=weight_decay
        )
        for _ in range(5):
            _step(optimizer, x)
        return x

    x_no_decay = run(0.0)
    x_with_decay = run(0.1)
    assert x_with_decay.norm() < x_no_decay.norm(), (
        "Weight decay should shrink the parameters more than no weight decay"
    )


@pytest.mark.parametrize("lookahead_steps", [0, -1, 2.5])
def test_invalid_lookahead_steps_raises_value_error(lookahead_steps: int) -> None:
    x = Parameter(torch.tensor([1.0]))
    with pytest.raises(ValueError, match="lookahead_steps"):
        RangerLite([x], lookahead_steps=lookahead_steps)


@pytest.mark.parametrize("lookahead_alpha", [0.0, 1.0, -0.1, 1.1])
def test_invalid_lookahead_alpha_raises_value_error(lookahead_alpha: float) -> None:
    x = Parameter(torch.tensor([1.0]))
    with pytest.raises(ValueError, match="lookahead_alpha"):
        RangerLite([x], lookahead_alpha=lookahead_alpha)


def test_state_dict() -> None:
    x = Parameter(torch.tensor([1.0, 2.0, 3.0]))
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
