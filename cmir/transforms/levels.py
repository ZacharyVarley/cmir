"""
Module for breaking image channel dimension into indicators as 
is often done in the calculation of binnned mutual information.
"""

from typing import Optional
import torch
import torch.nn as nn


@torch.jit.script
def _hard_indicators(x: torch.Tensor, n_levels: int) -> torch.Tensor:
    """Helper function for HardLevels."""
    B, C, H, W = x.shape
    output = (x[:, :, None, :, :] * n_levels).byte() == torch.arange(
        n_levels, dtype=torch.uint8, device=x.device
    )[None, None, :, None, None]
    return output.reshape(B, int(C * n_levels), H, W)


class HardLevels(nn.Module):
    """Break image channel dimension into indicators with a hard binning.
    This is not differentiable. Use SoftLevels for a differentiable version.
    Each channel is broken into n_levels indicators and each sub-channel
    indicator can only take on values 0 or 1. This is useful if you want to
    use a gradient-less optimizer like CMA-ES and you want to have faster
    downstream calculations with uint8 tensors.

    Args:
        n_levels (int): Number of levels to break each channel into.
        output_dtype (torch.dtype): dtype of output tensor.

    """

    def __init__(self, n_levels: int, out_dtype: torch.dtype = torch.uint8):
        super().__init__()

        self.n_levels = n_levels
        self.out_dtype = out_dtype

        if n_levels > 255 or n_levels < 2:
            raise ValueError("n_levels must be a positive integer from 2 to 255.")

        bin_centers = torch.arange(n_levels, dtype=out_dtype)
        self.register_buffer("bin_centers", bin_centers)

    def forward(self, x):
        # input sanitization
        if x.ndim != 4:
            raise ValueError("Input must be 4D tensor.")
        return _hard_indicators(x, self.n_levels).to(self.out_dtype)


@torch.jit.script
def _linear_bin_contributions(
    x: torch.Tensor, bin_centers: torch.Tensor
) -> torch.Tensor:
    """Helper function for LinearLevels."""
    B, C, H, W = x.shape
    n_levels = bin_centers.shape[0]
    diff = (x[:, :, None, :, :] - bin_centers[None, None, :, None, None]).abs()
    center_scores = 1.0 - (diff * n_levels)
    center_scores[center_scores < 0.0] = 0.0
    return torch.nn.functional.normalize(center_scores, p=1.0, dim=2).reshape(
        B, int(C * n_levels), H, W
    )


class LinearLevels(nn.Module):
    """Break image channel dimension into indicators with a linear binning.
    This is differentiable. Each channel is broken into n_levels indicators
    and each sub-channel indicators are normalize to a sum of 1.

    Args:
        n_levels (int): Number of levels to break each channel into.

    """

    def __init__(self, n_levels: int):
        super().__init__()

        bin_centers = torch.linspace(0.5 / n_levels, 1.0 - (0.5 / n_levels), n_levels)
        self.register_buffer("bin_centers", bin_centers)

    def forward(self, x):
        # input sanitization
        if x.ndim != 4:
            raise ValueError("Input must be 4D tensor.")
        return _linear_bin_contributions(x, self.bin_centers)


class GaussianLevels(nn.Module):
    """Break image channel dimension into indicators with a Gaussian probability
    as is done with Kernel Density Estimation with the Parzen-Rosenblatt window method

    https://en.wikipedia.org/wiki/Kernel_density_estimation

    Args:
        n_levels (int): Number of levels to break each channel into.
        sigma (float): Standard deviation of Gaussian distribution. By default it is
            Silverman's rule of thumb for Gaussian KDE, given h = 1.06 * sigma_hat * n^(-1/5)
            where n is n_levels, and sigma_hat is arbitrarily set to 0.4 here so it is not
            recalcuated every time the module is called.

    """

    def __init__(
        self,
        n_levels: int,
        parzen_h: Optional[float] = None,
    ):
        super().__init__()

        self.n_levels = n_levels

        bin_centers = torch.linspace(0.5 / n_levels, 1.0 - (0.5 / n_levels), n_levels)
        self.register_buffer("bin_centers", bin_centers)

        if parzen_h is None:
            self.h = 0.4 * (n_levels ** (-0.2))
        else:
            self.h = parzen_h

    def forward(self, x):
        # input sanitization
        if x.ndim != 4:
            raise ValueError("Input must be 4D tensor.")
        B, C, H, W = x.shape
        diff = x[:, :, None, :, :] - self.bin_centers[None, None, :, None, None]
        gaussian_ = torch.exp(-0.5 * (diff / self.h) ** 2)
        normed_gaussian = torch.nn.functional.normalize(gaussian_, p=1.0, dim=2)
        return normed_gaussian.reshape(B, int(C * self.n_levels), H, W)


class GumbelLevels(nn.Module):
    """Break image channel dimension into indicators using Gumbel-Softmax.
    Each channel is broken into n_levels indicators that have probabilities
    that sum to 1. If you need a one-hot output and a gradient for backprop,
    this will work with argument hard=True.

    Args:
        n_levels (int): Number of levels to break each channel into.

    """

    def __init__(
        self,
        n_levels: int,
        temperature: float = 1.0,
        hard: bool = True,
    ):
        super().__init__()

        self.n_levels = n_levels

        bin_centers = torch.linspace(0.5 / n_levels, 1.0 - (0.5 / n_levels), n_levels)
        self.register_buffer("bin_centers", bin_centers)

        # if temperature is below 0.0 raise an error
        if temperature < 0.0:
            raise ValueError("Temperature must be greater than 0.0.")

        self.temperature = temperature
        self.hard = hard

    def forward(self, x):
        # input sanitization
        if x.ndim != 4:
            raise ValueError("Input must be 4D tensor.")
        B, C, H, W = x.shape
        diff = (x[:, :, None, :, :] - self.bin_centers[None, None, :, None, None]).abs()
        # shape of diff is (B, C, n_levels, H, W) so softmax over n_levels
        # use torch.nn.functional.gumbel_softmax to get a differentiable one-hot encoding
        probabilties = torch.nn.functional.gumbel_softmax(
            diff, tau=self.temperature, hard=self.hard, dim=2
        )
        return probabilties.reshape(B, int(C * self.n_levels), H, W)
