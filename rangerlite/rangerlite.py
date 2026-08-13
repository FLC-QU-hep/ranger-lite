from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import RAdam

# need to define our own ParamsT to support older versions of pytorch
ParamsT = (
    Iterable[torch.Tensor]
    | Iterable[dict[str, Any]]
    | Iterable[tuple[str, torch.Tensor]]
)


class RangerLite(RAdam):
    """Lookahead optimizer using RAdam as its inner optimizer, inspired by the Ranger optimizer.

    References:
        1. Lookahead: https://arxiv.org/abs/1907.08610v1
        2. RAdam: https://arxiv.org/abs/1908.03265
        3. Ranger: https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = 1e-3,
        betas: tuple[float, float] = (0.95, 0.999),
        eps: float = 1e-5,
        weight_decay: float = 0.0,
        lookahead_steps: int = 6,
        lookahead_alpha: float = 0.5,
        *,
        foreach: bool | None = None,
        maximize: bool = False,
        capturable: bool = False,
    ) -> None:
        if not isinstance(lookahead_steps, int) or lookahead_steps < 1:
            raise ValueError(
                f"Invalid lookahead_steps value: {lookahead_steps}. "
                "Must be a positive integer."
            )
        if not isinstance(lookahead_alpha, float) or not (0.0 < lookahead_alpha < 1.0):
            raise ValueError(
                f"Invalid lookahead_alpha value: {lookahead_alpha}. "
                "Must be a float between 0 and 1."
            )
        defaults: dict[str, Any] = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "maximize": maximize,
            "foreach": foreach,
            "capturable": capturable,
            "decoupled_weight_decay": True,
            "differentiable": False,
            "lookahead_steps": lookahead_steps,
            "lookahead_alpha": lookahead_alpha,
        }
        super(RAdam, self).__init__(params, defaults)

    def _init_group(
        self,
        group: dict[str, Any],
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        exp_avgs: list[Tensor],
        exp_avg_sqs: list[Tensor],
        state_steps: list[Tensor],
    ) -> bool:
        has_complex = super()._init_group(
            group, params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps
        )
        for parameter in group["params"]:
            if isinstance(parameter, Tensor) and parameter.grad is not None:
                state = self.state[parameter]
                if "cached_params" not in state:
                    state["cached_params"] = torch.clone(parameter.data.detach())
        return has_complex

    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        loss = super().step(closure)
        for group in self.param_groups:
            for parameter in group["params"]:
                if not isinstance(parameter, Tensor) or parameter.grad is None:
                    continue
                state = self.state[parameter]
                if round(state["step"].item()) % group["lookahead_steps"] == 0:
                    parameter.data.mul_(group["lookahead_alpha"]).add_(
                        state["cached_params"],
                        alpha=1.0 - group["lookahead_alpha"],
                    )
                    state["cached_params"].copy_(parameter.data.detach())
        return loss
