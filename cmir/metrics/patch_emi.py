"""

This module contains the implementation of the Patch EMI metric. EMI is is
Edgeworth Mutual Information, a measure of the mutual information between two
variables which converges quickly under subsampling. Patch EMI is a novel metric
that computes the mutual information between patches of two images. It is
symmetric, inherently differentiable (unlike binning-based mutual information),
and can be computed with fewer operations.

Algorithm Steps for Input Images x and y:

1) Store a sliding window calculation of E[x], E[y], σ(x), σ(y) used later to 
zero mean and unit variance the pixels within each patch. 

2) Compute the cumulants from the central moments of the patches. If only a 3rd
order EMI approximation is used, then the cumulants are the expected powers. 4th
order cumulants (and above) are not equal to 4th order central moments.

3) Estimate the EMI using the cumulants and the Edgeworth expansion.

For 3rd order EMI, standardization is generally not desirable, and we actually
calculate the cumulants from non-central (raw) moments, removing step 1, and
modify the underlying assumptions in Mathieu's derivation. For 4th order EMI,
both standardization and lack thereof are reasonable, and we use the
standardization approach.

References:

The statistics and derivation are directly taken from the PhD dissertation of
Mathieu Rubeaux, available in French here:

https://theses.hal.science/tel-00632128 

Rubeaux, Mathieu. "Approximation de l'Information Mutuelle basée sur le
développement d'Edgeworth: application au recalage d'images médicales." PhD
diss., Université Rennes 1, 2011.

For tabulation of the bivariate cumulants and moments, see the following paper:

Cook, M. B. "Bi-variate k-statistics and cumulants of their joint sampling
distribution." Biometrika 38, no. 1/2 (1951): 179-195.

"""

import torch
import math
from torch import Tensor


@torch.jit.script
def spatially_separated_2D_avg_pool(
    x: Tensor,
    patch_radius: int,
) -> Tensor:
    """
    A spatially separated 2D average pooling operation for much larger patches.

    Args:
        x (Tensor): A 2D tensor of shape (B, C, H, W).
        patch_size (int): The size of the patches to average.

    Returns:
        Tensor: A 2D tensor of shape (B, C, H, W)
    """

    # need to get number of groups so that each layer is averaged separately
    B, C, H, W = x.shape

    # pad the input tensor to ensure that the output tensor has the same shape.
    x_padded = torch.nn.functional.pad(
        x,
        (patch_radius, patch_radius, patch_radius, patch_radius),
    )

    # 1D convolutional kernel to average the patches for use with torch.nn.functional.conv1d
    kernel = torch.ones(patch_radius * 2 + 1, dtype=x.dtype, device=x.device) / (
        patch_radius * 2 + 1
    )

    # reshape and repeat to (C, 1, 2*patch_radius+1, 1)
    kernel = kernel.view(1, 1, -1, 1).repeat(C, 1, 1, 1)

    # Average the patches along the rows
    avg = torch.nn.functional.conv2d(
        x_padded, kernel, stride=(1, 1), groups=C, padding=(0, 0)
    )

    # Average the patches along the columns
    avg = torch.nn.functional.conv2d(
        avg, kernel.transpose(2, 3), stride=(1, 1), groups=C, padding=(0, 0)
    )
    return avg


@torch.jit.script
def box_avg_per_channel(
    x: Tensor,
    patch_radius: int,
    padding_mode: str = "reflect",
) -> Tensor:
    """
    This function either uses separable 2D average pooling or a 2D convolution
    depending on the patch size. Above 11x11, the separable 2D average pooling
    is faster on a Nvidia P4 GPU, so that is the hardcoded threshold (r=5).

    Args:
        x (Tensor): A 2D tensor of shape (B, C, H, W).
        patch_size (int): The size of the patches to average.

    Returns:
        Tensor: A 2D tensor of shape (B, C, H, W)
    """

    # pad with reflection padding
    x = torch.nn.functional.pad(x, (patch_radius,) * 4, mode=padding_mode)

    # if the patch size is larger than 5, use separable 2D average pooling
    if patch_radius > 5:
        return spatially_separated_2D_avg_pool(x, patch_radius)
    else:
        return torch.nn.functional.avg_pool2d(
            x, patch_radius * 2 + 1, stride=1, padding=patch_radius
        )


@torch.jit.script
def _norm_mi(
    ent_joint: Tensor,
    ent_src: Tensor,
    ent_dst: Tensor,
    norm_mode: str = "none",
) -> Tensor:
    mi = ent_src + ent_dst - ent_joint
    # if the normalization mode is "none" then return the distances
    if norm_mode == "none":
        nmi = mi
    elif norm_mode == "Kvalseth Sum":  # divide by marginal arithmetic mean
        nmi = 2.0 * mi / (ent_src + ent_dst)
    elif norm_mode == "Kvalseth Min":  # divide by max marginal
        nmi = mi / torch.min(ent_src, ent_dst)
    elif norm_mode == "Kvalseth Max":  # divide by min marginal
        nmi = mi / torch.max(ent_src, ent_dst)
    elif norm_mode == "Yao":  # divide by joint
        nmi = mi / ent_joint
    elif norm_mode == "Strehl-Ghosh":  # divide by marginal geometric mean
        nmi = mi / torch.sqrt(ent_src * ent_dst)
    elif norm_mode == "Logarithmic":  # divide by marginal log mean
        # handle case if ent_src and ent_dst are the same
        nmi = mi / torch.where(
            (ent_src - ent_dst).abs() < 1e-6,
            ent_dst,
            (ent_src - ent_dst) / (torch.log(ent_src) - torch.log(ent_dst)),
        )
    else:
        raise ValueError(f"Unknown normalization mode: {norm_mode}")
    return nmi


@torch.jit.script
def patch_emi_3(
    x: Tensor,
    y: Tensor,
    x_ind: Tensor,
    y_ind: Tensor,
    patch_radius: int,
) -> Tensor:
    """
    This function computes the Patch EMI between two images x and y.

    Args:
        x (Tensor): A 2D tensor of shape (B, 1, H, W).
        y (Tensor): A 2D tensor of shape (B, 1, H, W).
        patch_radius (int): The radius of the patches to use.

    Returns:
        Tensor: A 2D tensor of shape (B,)
    """

    # # compute the cumulants from the central moments of the patches
    # m_10 = box_avg_per_channel(x, patch_radius)
    # m_01 = box_avg_per_channel(y, patch_radius)

    # m_20 = box_avg_per_channel(x**2, patch_radius)
    # m_11 = box_avg_per_channel(x * y, patch_radius)
    # m_02 = box_avg_per_channel(y**2, patch_radius)

    # m_30 = box_avg_per_channel(x**3, patch_radius)
    # m_21 = box_avg_per_channel(x**2 * y, patch_radius)
    # m_12 = box_avg_per_channel(x * y**2, patch_radius)
    # m_03 = box_avg_per_channel(y**3, patch_radius)

    # stack up x, y, x**2, ... for a single box_avg_per_channel call
    val_10 = x
    val_01 = y
    val_20 = x**2
    val_11 = x * y
    val_02 = y**2
    val_30 = x**3
    val_21 = x**2 * y
    val_12 = x * y**2
    val_03 = y**3

    # stack up the values
    vals = torch.cat(
        [val_10, val_01, val_20, val_11, val_02, val_30, val_21, val_12, val_03], dim=1
    )

    vals_avg = box_avg_per_channel(vals, patch_radius)

    # unstack the values
    m_10, m_01, m_20, m_11, m_02, m_30, m_21, m_12, m_03 = (
        vals_avg[:, 0, :, :],
        vals_avg[:, 1, :, :],
        vals_avg[:, 2, :, :],
        vals_avg[:, 3, :, :],
        vals_avg[:, 4, :, :],
        vals_avg[:, 5, :, :],
        vals_avg[:, 6, :, :],
        vals_avg[:, 7, :, :],
        vals_avg[:, 8, :, :],
    )

    # compute the cumulants from raw moments
    k_20 = m_20 - m_10**2
    k_02 = m_02 - m_01**2
    k_11 = m_11 - m_10 * m_01
    k_30 = m_30 - 3 * m_20 * m_10 + 2 * m_10**3
    k_21 = m_21 - m_20 * m_01 - 2 * m_11 * m_10 + 2 * m_10**2 * m_01
    k_12 = m_12 - m_02 * m_10 - 2 * m_11 * m_01 + 2 * m_01**2 * m_10
    k_03 = m_03 - 3 * m_02 * m_01 + 2 * m_01**3

    # standardize the 3rd order and lower cumulants
    k_11 = k_11 / torch.sqrt(k_20 * k_02)
    k_30 = k_30 / (k_20**1.5)
    k_21 = k_21 / (k_20 * torch.sqrt(k_02))
    k_12 = k_12 / (k_02 * torch.sqrt(k_20))
    k_03 = k_03 / (k_02**1.5)

    # estimate the joint histogram entropy
    k_20_2nd = k_20**2
    k_20_3rd = k_20**3

    k_11_2nd = k_11**2
    k_11_3rd = k_11**3

    k_02_2nd = k_02**2
    k_02_3rd = k_02**3

    k_30_2nd = k_30**2
    k_21_2nd = k_21**2
    k_12_2nd = k_12**2
    k_03_2nd = k_03**2

    # terms for 3rd order approximation
    emi_3e_term_A = (
        k_03_2nd * k_20_3rd
        + 3 * k_12_2nd * k_20_2nd * k_02
        + 3 * k_21_2nd * k_20 * k_02_2nd
        + k_30_2nd * k_02_3rd
    )
    emi_3e_term_B = -1.0 * k_11_3rd * (2 * k_03 * k_30 + 6 * k_12 * k_21)
    emi_3e_term_C = (
        6.0
        * k_11_2nd
        * (k_03 * k_21 * k_20 + k_12_2nd * k_20 + k_12 * k_30 * k_02 + k_21_2nd * k_02)
    )
    emi_3e_term_D = (
        -6.0
        * k_11
        * (
            k_03 * k_12 * k_20_2nd
            + 2 * k_12 * k_21 * k_20 * k_02
            + k_21 * k_30 * k_02_2nd
        )
    )

    det_cov = k_20 * k_02 - k_11_2nd

    # calculate the joint entropy
    ent_joint = (
        1.0
        + math.log(2 * torch.pi)
        + 0.5 * torch.log(det_cov)
        - (1.0 / (12.0 * (det_cov**3)))
        * (emi_3e_term_A + emi_3e_term_B + emi_3e_term_C + emi_3e_term_D)
    )

    # compute the EMI using the cumulants and the Edgeworth expansion
    ent_src = 0.5 * torch.log(2 * torch.pi * torch.e * k_20) - (k_30_2nd / 12.0)
    ent_dst = 0.5 * torch.log(2 * torch.pi * torch.e * k_02) - (k_03_2nd / 12.0)

    # remove negative entropy values as they are not possible
    ent_joint[ent_joint < 0.0] = 0.0
    ent_src[ent_src < 0.0] = 0.0
    ent_dst[ent_dst < 0.0] = 0.0

    # normalize the mutual information
    nmi = _norm_mi(ent_joint, ent_src, ent_dst, "none")

    # replace any NaNs or Infs or Negatives with zeros
    nmi[torch.isnan(nmi) | torch.isinf(nmi) | (nmi < 0.0)] = 0.0

    # average over the image plane
    nmi = nmi.mean(dim=(1, 2, 3))

    return nmi


# test runtime against torch.nn.functional.avg_pool2d
from torch.profiler import ProfilerActivity, profile
import torch.nn.functional as F
import time

x = torch.rand(1, 128, 512, 512, device="cuda", dtype=torch.float16)
patch_radius = 5
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
) as prof:
    for _ in range(100):
        spatially_separated_2D_avg_pool(x, patch_radius)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
) as prof:
    for _ in range(100):
        F.avg_pool2d(x, patch_radius * 2 + 1, stride=1, padding=patch_radius)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
