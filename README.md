# RangerLite
A pytorch implementation of the Lookahead optimizer with RAdam as inner optimizer.

### Table of Contents
- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Relation to other Implementations](#relation-to-other-implementations)
- [Development](#development)
- [References](#references)


## Installation
The simplest way to install RangerLite is to use pip:
```bash
pip install git+https://github.com/ThorstenBuss/ranger-lite.git
```

In your `requirements.txt` you can add:
```
rangerlite @ git+https://github.com/ThorstenBuss/ranger-lite.git@v0.1.0
```

## Requirements
RangerLite requires Python 3.10 or later and PyTorch 2.0 or later.

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
The RangerLite optimizer is inspired by the original Ranger[1] optimizer, which combines RAdam[2], Lookahead[3] and gradient centralization[4]. RangerLite drops the gradient centralization and provides a lightweight alternative inheriting from PyTorch's RAdam implementation. It can easily be used and integrated into existing PyTorch workflows and provides type hinting for better developer experience.

### Why not use the original Lookahead implementation?
The original Lookahead implementation[5] uses composition of optimizers, which can lead to unexpected behavior when setting hyper-parameters for individual parameter groups or frameworks like pytorch-lightning. Saving and loading the state dict is also not guaranteed to result in the same state leading to potential issues with checkpointing.

## Development
For development, first clone the repository and than install it as editable package including the development dependencies:
```bash
# clone the repository
git clone https://github.com/ThorstenBuss/ranger-lite.git
cd ranger-lite

# create a virtual environment (you can also use a different tool)
python3 -m venv .venv
source .venv/bin/activate

# install the package and its development dependencies as editable package
pip install -e .
pip install --group dev
pip install --group test
```

This repository uses [pre-commit](https://pre-commit.com/) hooks to ensure consistent code style. To install the pre-commit hooks, run:
```bash
pre-commit install
```

The unit tests can be run with:
```bash
pytest
```

## References
1. Ranger: [https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)
2. RAdam: [https://arxiv.org/abs/1908.03265](https://arxiv.org/abs/1908.03265)
3. Lookahead: [https://arxiv.org/abs/1907.08610](https://arxiv.org/abs/1907.08610v1)
4. Gradient Centralization: [https://arxiv.org/abs/2004.01461](https://arxiv.org/abs/2004.01461)
5. Original Lookahead implementation: [https://github.com/michaelrzhang/lookahead](https://github.com/michaelrzhang/lookahead)
