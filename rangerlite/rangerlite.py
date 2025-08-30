from collections.abc import Callable
from typing import Any, override

import torch
from torch import Tensor
from torch.optim import RAdam
from torch.optim.optimizer import ParamsT


class RangerLite(RAdam):
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
        defaults = {
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

    @override
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

    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = super().step(closure)
        for group in self.param_groups:
            for parameter in group["params"]:
                if not isinstance(parameter, Tensor) or parameter.grad is None:
                    continue
                state = self.state[parameter]
                if int(state["step"].item() + 0.1) % group["lookahead_steps"] == 0:
                    parameter.data.mul_(group["lookahead_alpha"]).add_(
                        state["cached_params"],
                        alpha=1.0 - group["lookahead_alpha"],
                    )
                    state["cached_params"].copy_(parameter.data.detach())
        return loss
