<table>
  <tr>
    <td>
      <svg height="200" width="200" xmlns="http://www.w3.org/2000/svg">
        <circle cx="100" cy="100" r="95" fill="#ef4b28"/>
        <rect x="35" y="70" rx="10" ry="10" width="60" height="60" fill="#006AC9"/>
        <path fill="#2F4858" d="M166 130V70L35 130z"/>
        <circle cx="135" cy="100" r="29" fill="#2F9A00"/>
      </svg>
    </td>
    <td style="font-size: 30px; padding-left: 20px;">
      <strong>CMIR - Cross Modality Image Registration</strong>
    </td>
  </tr>
</table>


[![PyPI version](https://badge.fury.io/py/cmir.svg)](https://badge.fury.io/py/cmir)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cmir)](https://pypi.org/project/cmir/)
![PyPI - License](https://img.shields.io/pypi/l/cmir) 
[![PyPI - Downloads](https://img.shields.io/pypi/dm/cmir)](https://pypi.org/project/cmir/)

- [cmir](#cmir)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Metrics](#registration)

## Introduction

Cross Modality Image Registration (CMIR) is a PyTorch-based package for tackling multimodal registration problems that involve sets of images or spatial stacks of images. CMIR provides a set of tools which are focused on stack registration on the GPU using new algorithms. Other packages such as [MONAI](https://monai.io/) or [NITorch](https://github.com/balbasty/nitorch) are more suitable for non serial sectioning registration problems.

CMIR is designed to be:

- **Easy to install**: PyTorch sole-dependency.
- **Easy to use**: Pipeline based workflows.
- **Easy to contribute**: Make your own torch.nn.Module to either compute metrics, deform pixel coordinates, or augment images.
- **Fast**: GPU accelerated registration algorithms.
- **Flexible**: PyTorch has increasing support for AMD GPUs and Apple Silicon. Using TorchScript for just-in-time (jit) compiled modules.

## Installation

The package can be installed using pip:

```bash
pip install cmir
```

## Usage

### CCA Loss Metrics for Multimodal Data

The package provides a set of evaluation metrics implemented in the `cmir.metrics` module:

```python

import torch
from cmir.metrics import PatchCCADense

# Create two stacks of images (B, C, H, W)
# Notice that channel dimensions are different
stack_reference = torch.rand(10, 4, 32, 32)
stack_moving = torch.rand(10, 3, 32, 32)

# CCA in patches with radius 2 and 5 will be computed
metric = PatchCCADense([2, 5])

# inter-image CCA between all 32**2 = 1024 patches
metric(stack_reference, stack_moving) # (10, 1024, 3)
