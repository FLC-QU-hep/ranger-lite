# RangerLite

A pytorch implementation of the Lookahead optimizer with RAdam as inner optimizer.

## Installation
The simplest way to install ranger-lite is to use pypi:
```bash
pip install ranger-lite
```

For development, first clone the repository and than install it as editable package including the development dependencies (consider using a virtual environment):
```bash
git clone https://github.com/ThorstenBuss/ranger-lite.git
cd ranger-lite
pip install -e ".[dev]"
```

## Usage
You can use RangerLite as a drop in pytorch optimizer:
```python
import torch
from rangerlite import RangerLite

model = ...  # your model
optimizer = RangerLite(model.parameters())
```

A simple example of using RangerLite to find the minimum of the quadratic function (20 steps are not sufficient to get close to the minimum, but you can see the loss decreasing):
```python
import torch
from rangerlite import RangerLite

x = torch.tensor(2.0, requires_grad=True)
optimizer = RangerLite(
    params=[x],
    lr=2e-1,
    lookahead_steps=4,
)

for step in range(20):
    loss = x**2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Step {step+1:2d}: x = {x:.4f}, loss = {loss:.4f}, grad = {x.grad:.4f}")
```

## Relation to other Implementations
The RangerLite optimizer is inspired by the original Ranger[1] optimizer, which combines RAdam[2], Lookahead[3] and gradient centralization. RangerLite drops the gradient centralization and provides a lightweight alternative that is easier to use and integrate into existing PyTorch workflows.

### Why not use the original Lookahead implementation?
The original Lookahead implementation[4] uses composition of optimizers, which can lead to unexpected behavior when setting hyper-parameters for individual parameter groups or using it in combination with frameworks like pytorch-lightning. Saving and loading the parameter dict is also not guaranteed to result in the same state.

## References
1. Ranger: [https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)
2. RAdam: [https://arxiv.org/abs/1908.03265](https://arxiv.org/abs/1908.03265)
3. Lookahead: [https://arxiv.org/abs/1907.08610](https://arxiv.org/abs/1907.08610v1) 
4. Original Lookahead implementation: [https://github.com/michaelrzhang/lookahead](https://github.com/michaelrzhang/lookahead)
