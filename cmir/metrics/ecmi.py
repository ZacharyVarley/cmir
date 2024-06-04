"""

This file contains the 3rd and 4th order Edgeworth expansions for the mutual
information between two images. Mutual information costs 5 cross correlations
or 10 if both images are masked because noncentral moments have to be computed
before cumulants are calculated. For reference, 16 bin histogram mutual info
costs 256 cross correlations.

The main idea here is that a series approximation can be used to estimate the
entropy of both the joint and marginal distributions. This is done by using the
cumulants of the distributions. The cumulants are calculated from the moments of
the distributions, in turn calculated from the images. 

The statistics and derivation are directly taken from the PhD dissertation of
Mathieu Rubeaux, available in French here:

https://theses.hal.science/tel-00632128 

Rubeaux, Mathieu. "Approximation de l'Information Mutuelle basée sur le
développement d'Edgeworth: application au recalage d'images médicales." PhD
diss., Université Rennes 1, 2011.

---

My contribution is solely to take these formulae and densely calculate them over
discrete shifts of the images, as is done in cross correlation.

In his thesis, Mathieu observed certain terms exploding with his implementation
of the 4th order approximation. I was not able to reproduce these numerical
issues, and I think as he thought, that it was possibly an implementation error.
I did however observe numerical instability in calculating cumulants from the
raw moments, as would be expected. The best way to mitigate this seems to be to
normalize the images to zero mean and unit variance, even if it is redone in the
dense calculation.

For tabulation of the bivariate cumulants and moments, see the following paper:

Cook, M. B. "Bi-variate k-statistics and cumulants of their joint sampling
distribution." Biometrika 38, no. 1/2 (1951): 179-195.

"""

import torch
import math
from torch import Tensor
from typing import Optional
from torch.fft import rfft2, irfft2, fftshift


@torch.jit.script
def _norm_via_validity_mask(images: Tensor, masks: Tensor) -> Tensor:
    """

    This function normalizes a stack of images by subtracting the mean and
    dividing by the standard deviation. The mean and standard deviation are
    calculated only within the validity mask on a per image basis.

    Args:
        img_stack: a stack of images of shape (B, 1, H, W)
        mask: a validity mask of shape (B, 1, H, W)

    Returns:
        Tensor: the normalized image stack

    """
    # check that the image and mask are 4D tensors
    if images.ndim != 4 or masks.ndim != 4:
        raise ValueError(
            f"Image and mask must be 4D tensors, but got {images.ndim} and {masks.ndim}"
        )

    # check that the image and mask are the same size
    if images.shape != masks.shape:
        raise ValueError(
            f"Image and mask must be the same shape, but got {images.shape} and {masks.shape}"
        )

    # apply the mask to the image stack
    mask_sums = masks.sum(dim=(-2, -1), keepdim=True)
    imgs_normed = (images * masks).sum(dim=(-2, -1), keepdim=True) / mask_sums

    # subtract the mean from the images. We just set E[X] = 0 for the valid pixels
    imgs_normed = -1.0 * (imgs_normed - images)

    # calculate the standard deviation via VAR(X) = E[X^2] - E[X]^2 = E[X^2] - (0)^2
    imgs_normed /= (
        (imgs_normed**2 * masks).sum(dim=(-2, -1), keepdim=True) / mask_sums
    ) ** 0.5

    # set the invalid pixels to zero
    imgs_normed *= masks

    return imgs_normed


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
def enmi_3(
    images_dst: Tensor,
    images_src: Tensor,
    dst_ind: Optional[Tensor] = None,
    src_ind: Optional[Tensor] = None,
    norm_mode: str = "none",
) -> Tensor:
    """
    ---
    WARNING:
    Use the 4th order approximation provided under the name "enmi_4".
    This is provided for completeness and clarity.
    ---

    Calculate 3rd Order Edgeworth Normalized Mutual Information densely
    over shifts, making a "map" of the mutual information as the src image
    is shifted over the dst image. The zero point is the center of the map.

    Calculate a "map" of the the Edgeworth expansion based estimation of the
    mutual information between two images. This is done densely via real valued
    2D FFTs.

    Args:
        images_src: The source images of shape (B, 1, H, W)
        images_dst: The destination images of shape (B, 1, H, W)
        src_ind: Indicator masks for source images of shape (B, 1, H, W)
        dst_ind: Indicator masks for destination images of shape (B, 1, H, W)
        normalization_mode: The normalization mode to use. One of:
            "none": No normalization is applied
            "Kvalseth Sum": The arithmetic mean of the marginal entropies
            "Kvalseth Min": The minimum of the marginal entropies
            "Kvalseth Max": The maximum of the marginal entropies
            "Yao": The joint entropy
            "Strehl-Ghosh": The geometric mean of the marginal entropies
            "Logarithmic": The logarithmic mean of the marginal entropies

    Returns:
        Tensor: The "estimated" mutual information map: (B, 1, H, W)

    """
    if images_src.ndim != 4 or images_dst.ndim != 4:
        raise ValueError(
            f"Images must be 4D tensors, but got {images_src.ndim} and {images_dst.ndim}"
        )

    b1, c1, h1, w1 = images_src.shape
    b2, c2, h2, w2 = images_dst.shape

    # check that these are 4D tensors and they are grayscale
    if c1 != 1 or c2 != 1:
        raise ValueError(f"Images must be grayscale, but got {c1} and {c2} channels")

    # check that the images are the same size
    if h1 != h2 or w1 != w2:
        raise ValueError(f"Images different size: src {h1}x{w1} and dst {h2}x{w2}")

    # check that the image batch sizes are the same size
    if b1 != b2 and b1 != 1 and b2 != 1:
        raise ValueError(f"Incompatible batch sizes: src {b1} and dst {b2}")

    # we need to zero mean and unit norm the images (only within the mask)
    if src_ind is None:
        src_normed = images_src - images_src.mean(dim=(2, 3), keepdim=True)
        src_normed = src_normed / src_normed.std(dim=(2, 3), keepdim=True)
    else:
        src_normed = _norm_via_validity_mask(images_src, src_ind)

    if dst_ind is None:
        dst_normed = images_dst - images_dst.mean(dim=(2, 3), keepdim=True)
        dst_normed = dst_normed / dst_normed.std(dim=(2, 3), keepdim=True)
    else:
        dst_normed = _norm_via_validity_mask(images_dst, dst_ind)

    # make a stack of the normed images raised to 1st, 2nd, 3rd, and 4th powers
    # padding for FFTs. This is not just for "partial overlap". It is required.
    src_normed_2nd = src_normed**2
    src_normed_3rd = src_normed**3

    dst_normed_2nd = dst_normed**2
    dst_normed_3rd = dst_normed**3

    # do the overlap first
    # if both indicators are None, then use the same ones tensor for both
    if src_ind is None and dst_ind is None:
        # if there is no masking set the indicators to the scalar 1
        src_ind = torch.tensor(1.0, device=src_normed.device, dtype=src_normed.dtype)
        dst_ind = src_ind
        overlap = torch.tensor(
            src_normed.shape[-2] * src_normed.shape[-1],
            device=src_normed.device,
            dtype=src_normed.dtype,
        )
    else:
        if src_ind is None:
            src_ind = torch.tensor(
                1.0, device=src_normed.device, dtype=src_normed.dtype
            )
        if dst_ind is None:
            dst_ind = torch.tensor(
                1.0, device=dst_normed.device, dtype=dst_normed.dtype
            )
        overlap = (src_ind * dst_ind).sum(dim=(-2, -1))

    # second order
    k_11 = (src_normed * dst_normed).sum(dim=(-2, -1)) / overlap
    # third order
    k_30 = (src_normed_3rd * dst_ind).sum(dim=(-2, -1)) / overlap
    k_21 = (src_normed_2nd * dst_normed).sum(dim=(-2, -1)) / overlap
    k_12 = (src_normed * dst_normed_2nd).sum(dim=(-2, -1)) / overlap
    k_03 = (src_ind * dst_normed_3rd).sum(dim=(-2, -1)) / overlap

    # estimate the joint histogram entropy
    k_11_sq = k_11**2
    k_30_sq = k_30**2
    k_03_sq = k_03**2
    k_12_sq = k_12**2
    k_21_sq = k_21**2

    term_A = k_30_sq + k_03_sq + 3 * k_12_sq + 3 * k_21_sq
    term_B = -6.0 * k_11 * (k_30 * k_21 + k_03 * k_12 + 2 * k_21 * k_12)
    term_C = 6.0 * k_11_sq * (k_21_sq + k_12_sq + k_30 * k_12 + k_03 * k_21)
    term_D = -2.0 * k_11**3 * (k_30 * k_03 + 3 * k_21 * k_12)

    # calculate the joint entropy
    ent_joint = (
        1.0
        + math.log(2 * torch.pi)
        + 0.5 * torch.log(1 - k_11_sq)
        - (1.0 / (12.0 * (1 - k_11_sq) ** 3)) * (term_A + term_B + term_C + term_D)
    )

    # calculate the marginal entropies
    ent_src = 0.5 * math.log(2 * torch.pi * torch.e) - (k_30_sq / 12.0)
    ent_dst = 0.5 * math.log(2 * torch.pi * torch.e) - (k_03_sq / 12.0)

    # calculate the mutual information
    mi = ent_src + ent_dst - ent_joint

    # replace any NaNs or Infs or Negatives with zeros
    mi[torch.isnan(mi) | torch.isinf(mi)] = 0.0

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
def enmi_4(
    images_dst: Tensor,
    images_src: Tensor,
    dst_ind: Optional[Tensor] = None,
    src_ind: Optional[Tensor] = None,
    norm_mode: str = "none",
) -> Tensor:
    """

    Calculate 4th Order Edgeworth Normalized Mutual Information densely
    over shifts, making a "map" of the mutual information as the src image
    is shifted over the dst image. The zero point is the center of the map.

    Args:
        images_src: a stack of source images of shape (B, 1, H, W)
        images_dst: a stack of destination images of shape (B, 1, H, W)
        src_ind: indicator mask of the source images' foregrounds. shape (B, 1,
            H, W). If None, the entire image is considered foreground. Defaults
            to None.
        dst_ind: indicator mask of the destination images' foregrounds. shape
            (B, 1, H, W). If None, the entire image is considered foreground.
            Defaults to None.
        normalization_mode: the mode for normalizing the mutual information.
            Options are "none", "Kvalseth Sum", "Kvalseth Min", "Kvalseth Max",
            "Yao", "Strehl-Ghosh", "Logarithmic". Defaults to "none".
        zero_mean_approximation: whether to use the zero mean approximation.
            Defaults to False. As the image shifts off the edge of the canvas,
            the zero mean immediately becomes invalid. We can ignore this and
            calculate cumulants as if the center of the pixel distribution is
            still 0.

    Returns:
        Tensor: The "estimated" normalized mutual information values: (B,)

    """
    if images_src.ndim != 4 or images_dst.ndim != 4:
        raise ValueError(
            f"Images must be 4D tensors, but got {images_src.ndim} and {images_dst.ndim}"
        )

    b1, c1, h1, w1 = images_src.shape
    b2, c2, h2, w2 = images_dst.shape

    # check that these are 4D tensors and they are grayscale
    if c1 != 1 or c2 != 1:
        raise ValueError(f"Images must be grayscale, but got {c1} and {c2} channels")

    # check that the images are the same size
    if h1 != h2 or w1 != w2:
        raise ValueError(f"Images different size: src {h1}x{w1} and dst {h2}x{w2}")

    # check that the image batch sizes are the same size
    if b1 != b2 and b1 != 1 and b2 != 1:
        raise ValueError(f"Incompatible batch sizes: src {b1} and dst {b2}")

    # we need to zero mean and unit norm the images (only within the mask)
    if src_ind is None:
        src_normed = images_src - images_src.mean(dim=(2, 3), keepdim=True)
        src_normed = src_normed / src_normed.std(dim=(2, 3), keepdim=True)
    else:
        src_normed = _norm_via_validity_mask(images_src, src_ind)

    if dst_ind is None:
        dst_normed = images_dst - images_dst.mean(dim=(2, 3), keepdim=True)
        dst_normed = dst_normed / dst_normed.std(dim=(2, 3), keepdim=True)
    else:
        dst_normed = _norm_via_validity_mask(images_dst, dst_ind)

    # make a stack of the normed images raised to 1st, 2nd, 3rd, and 4th powers
    # padding for FFTs. This is not just for "partial overlap". It is required.
    src_normed_2nd = src_normed**2
    src_normed_3rd = src_normed**3
    src_normed_4th = src_normed**4

    dst_normed_2nd = dst_normed**2
    dst_normed_3rd = dst_normed**3
    dst_normed_4th = dst_normed**4

    # do the overlap first
    # if both indicators are None, then use the same ones tensor for both
    if src_ind is None and dst_ind is None:
        # if there is no masking set the indicators to the scalar 1
        src_ind = torch.tensor(1.0, device=src_normed.device, dtype=src_normed.dtype)
        dst_ind = src_ind
        overlap = torch.tensor(
            src_normed.shape[-2] * src_normed.shape[-1],
            device=src_normed.device,
            dtype=src_normed.dtype,
        )
    else:
        if src_ind is None:
            src_ind = torch.tensor(
                1.0, device=src_normed.device, dtype=src_normed.dtype
            )
        if dst_ind is None:
            dst_ind = torch.tensor(
                1.0, device=dst_normed.device, dtype=dst_normed.dtype
            )
        overlap = (src_ind * dst_ind).sum(dim=(-2, -1))

    # second order
    k_11 = (src_normed * dst_normed).sum(dim=(-2, -1)) / overlap
    # third order
    k_30 = (src_normed_3rd * dst_ind).sum(dim=(-2, -1)) / overlap
    k_21 = (src_normed_2nd * dst_normed).sum(dim=(-2, -1)) / overlap
    k_12 = (src_normed * dst_normed_2nd).sum(dim=(-2, -1)) / overlap
    k_03 = (src_ind * dst_normed_3rd).sum(dim=(-2, -1)) / overlap
    # fourth order
    k_40 = (src_normed_4th * dst_ind).sum(dim=(-2, -1)) / overlap
    k_31 = (src_normed_3rd * dst_normed).sum(dim=(-2, -1)) / overlap
    k_22 = (src_normed_2nd * dst_normed_2nd).sum(dim=(-2, -1)) / overlap
    k_13 = (src_normed * dst_normed_3rd).sum(dim=(-2, -1)) / overlap
    k_04 = (src_ind * dst_normed_4th).sum(dim=(-2, -1)) / overlap

    # estimate the joint histogram entropy
    k_11_2nd = k_11**2
    k_11_3rd = k_11**3
    k_11_4th = k_11**4
    k_11_5th = k_11**5
    k_11_6th = k_11**6

    k_30_2nd = k_30**2
    k_21_2nd = k_21**2
    k_12_2nd = k_12**2
    k_03_2nd = k_03**2

    k_30_3rd = k_30**3
    k_03_3rd = k_03**3

    k_30_4th = k_30**4
    k_03_4th = k_03**4

    k_40_2nd = k_40**2
    k_31_2nd = k_31**2
    k_22_2nd = k_22**2
    k_13_2nd = k_13**2
    k_04_2nd = k_04**2

    # terms for 3rd order approximation
    emi_3e_term_A = k_30_2nd + k_03_2nd + 3 * k_12_2nd + 3 * k_21_2nd
    emi_3e_term_B = -6.0 * k_11 * (k_30 * k_21 + k_03 * k_12 + 2 * k_21 * k_12)
    emi_3e_term_C = 6.0 * k_11_2nd * (k_21_2nd + k_12_2nd + k_30 * k_12 + k_03 * k_21)
    emi_3e_term_D = -2.0 * k_11_3rd * (k_30 * k_03 + 3 * k_21 * k_12)

    # calculate the joint entropy
    emi_joint_3rd_order = (
        1.0
        + math.log(2 * torch.pi)
        + 0.5 * torch.log(1 - k_11_2nd)
        - (1.0 / (12.0 * (1 - k_11_2nd) ** 3))
        * (emi_3e_term_A + emi_3e_term_B + emi_3e_term_C + emi_3e_term_D)
    )

    emi_4e_term_1A = k_40_2nd**2 + 4 * k_31_2nd + 6 * k_22_2nd + 4 * k_13_2nd + k_04_2nd

    emi_4e_term_1B = (
        -8.0 * k_11 * (k_40 * k_31 + 3 * k_31 * k_22 + 3 * k_22 * k_13 + k_13 * k_04)
    )
    emi_4e_term_1C = (
        12.0
        * k_11_2nd
        * (
            k_31_2nd
            + 2.0 * k_22_2nd
            + k_13_2nd
            + k_40 * k_22
            + 2.0 * k_31 * k_13
            + k_22 * k_04
        )
    )
    emi_4e_term_1D = (
        -8.0
        * k_11_3rd
        * (k_40 * k_13 + 3 * k_31 * k_22 + k_31 * k_04 + 3 * k_22 * k_13)
    )
    emi_4e_term_1E = 2.0 * k_11_4th * (3 * k_22_2nd + k_40 * k_04 + 4 * k_31 * k_13)

    emi_4e_term_2A = (
        k_30_4th
        + 6 * k_30_2nd * k_12_2nd
        + 15 * k_30_2nd * k_12_2nd
        + 20 * k_30_2nd * k_03_2nd
        + 15 * k_21_2nd * k_03_2nd
        + 6 * k_12_2nd * k_03_2nd
        + k_03_4th
    )

    emi_4e_term_2B = (
        -12.0
        * k_11
        * (
            k_30_3rd * k_21
            + 10 * k_30_2nd * k_12 * k_03
            + 5 * k_30_2nd * k_21 * k_12
            + 10 * k_30 * k_21 * k_03_2nd
            + 5 * k_21 * k_12 * k_03_2nd
            + k_12 * k_03_3rd
        )
    )

    emi_4e_term_2C = (
        30.0
        * k_11_2nd
        * (
            k_30_2nd * k_21_2nd
            + 4 * k_30_2nd * k_12_2nd
            + 6 * k_30_2nd * k_03_2nd
            + 4 * k_21_2nd * k_03_2nd
            + k_12_2nd * k_03_2nd
            + k_30_3rd * k_12
            + 4 * k_30_2nd * k_21 * k_03
            + 6 * k_30 * k_21 * k_12 * k_03
            + 4 * k_30 * k_12 * k_03_2nd
            + k_21 * k_03_3rd
        )
    )

    emi_4e_term_2D = (
        -40.0
        * k_11_3rd
        * (
            k_30_3rd * k_03
            + 9 * k_30_2nd * k_12 * k_03
            + 3 * k_30_2nd * k_21 * k_12
            + 3 * k_30 * k_21_2nd * k_03
            + 9 * k_30 * k_21 * k_03_2nd
            + 3 * k_30 * k_12_2nd * k_03
            + 3 * k_21 * k_12 * k_03_2nd
            + k_30 * k_03_3rd
        )
    )

    emi_4e_term_2E = (
        30.0
        * k_11_4th
        * (
            3 * k_30_2nd * k_12_2nd
            + 6 * k_30_2nd * k_03_2nd
            + 3 * k_21_2nd * k_03_2nd
            + 5 * k_30_2nd * k_21 * k_03
            + 10 * k_30 * k_21 * k_12 * k_03
            + 5 * k_30 * k_12 * k_03_2nd
        )
    )

    emi_4e_term_2F = (
        -12.0
        * k_11_5th
        * (
            5 * k_30 * k_21_2nd * k_03
            + 11 * k_30 * k_21 * k_03_2nd
            + 11 * k_30_2nd * k_12 * k_03
            + 5 * k_30 * k_12_2nd * k_03
        )
    )

    emi_4e_term_2G = (
        2.0 * k_11_6th * (11 * k_30_2nd * k_03_2nd + 21 * k_30 * k_21 * k_12 * k_03)
    )

    emi_4e_term_1_prefactor = -(1.0 / (48.0 * (1 - k_11_2nd) ** 4))
    emi_4e_term_2_prefactor = -(10.0 / (144.0 * (1 - k_11_2nd) ** 6))

    ent_joint = (
        emi_joint_3rd_order
        + emi_4e_term_1_prefactor
        * (
            emi_4e_term_1A
            + emi_4e_term_1B
            + emi_4e_term_1C
            + emi_4e_term_1D
            + emi_4e_term_1E
        )
        + emi_4e_term_2_prefactor
        * (
            emi_4e_term_2A
            + emi_4e_term_2B
            + emi_4e_term_2C
            + emi_4e_term_2D
            + emi_4e_term_2E
            + emi_4e_term_2F
            + emi_4e_term_2G
        )
    )

    # calculate the marginal entropies
    ent_src = (
        0.5 * math.log(2 * torch.pi * torch.e)
        - (k_30_2nd / 12.0)
        - (k_40_2nd / 48.0)
        - (5.0 * k_30_2nd**2 / 72.0)
    )
    ent_dst = (
        0.5 * math.log(2 * torch.pi * torch.e)
        - (k_03_2nd / 12.0)
        - (k_04_2nd / 48.0)
        - (5.0 * k_03_2nd**2 / 72.0)
    )

    # calculate the mutual information
    mi = ent_src + ent_dst - ent_joint

    # replace any NaNs or Infs or Negatives with zeros
    mi[torch.isnan(mi) | torch.isinf(mi)] = 0.0

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


# @torch.jit.script
def enmi_3_map_2D(
    images_dst: Tensor,
    images_src: Tensor,
    dst_ind: Optional[Tensor] = None,
    src_ind: Optional[Tensor] = None,
    norm_mode: str = "none",
    overlap_threshold: float = 0.5,
) -> Tensor:
    """

    Calculate 3rd Order Edgeworth Normalized Mutual Information densely
    over shifts, making a "map" of the mutual information as the src image
    is shifted over the dst image. The zero point is the center of the map.

    If dst_ind or src_ind are not provided, then the entire image is used
    to zero mean and unit norm the images. This can completely ruin the
    mutual information calculation if unintended.

    Args:
        images_src: a stack of source images of shape (B, 1, H, W)
        images_dst: a stack of destination images of shape (B, 1, H, W)
        src_ind: indicator mask of the source images' foregrounds. shape (B, 1,
            H, W). If None, the entire image is considered foreground. Defaults
            to None.
        dst_ind: indicator mask of the destination images' foregrounds. shape
            (B, 1, H, W). If None, the entire image is considered foreground.
            Defaults to None.
        normalization_mode: the mode for normalizing the mutual information.
            Options are "none", "Kvalseth Sum", "Kvalseth Min", "Kvalseth Max",
            "Yao", "Strehl-Ghosh", "Logarithmic". Defaults to "none".
        pad_contributes: if True, then the padding pixels contribute to the
            mutual information. If False, then the padding pixels are ignored.
            Defaults to False.

    Returns:
        Tensor: The "estimated" mutual information map: (B, 1, H, W)

    """
    if images_src.ndim != 4 or images_dst.ndim != 4:
        raise ValueError(
            f"Images must be 4D tensors, but got {images_src.ndim} and {images_dst.ndim}"
        )

    b1, c1, h1, w1 = images_dst.shape
    b2, c2, h2, w2 = images_src.shape

    # check that these are 4D tensors and they are grayscale
    if c1 != 1 or c2 != 1:
        raise ValueError(f"Images must be grayscale, but got {c1} and {c2} channels")

    # check that the images are the same size
    if h1 != h2 or w1 != w2:
        raise ValueError(f"Images different size: src {h1}x{w1} and dst {h2}x{w2}")

    # check that the image batch sizes are the same size
    if b1 != b2 and b1 != 1 and b2 != 1:
        raise ValueError(f"Incompatible batch sizes: src {b1} and dst {b2}")

    # we need to zero mean and unit norm the images (only within the mask)
    if src_ind is None:
        src_normed = images_src - images_src.mean(dim=(2, 3), keepdim=True)
        src_normed = src_normed / src_normed.std(dim=(2, 3), keepdim=True)
        src_ind = torch.ones_like(src_normed)
    else:
        src_normed = _norm_via_validity_mask(images_src, src_ind)

    if dst_ind is None:
        dst_normed = images_dst - images_dst.mean(dim=(2, 3), keepdim=True)
        dst_normed = dst_normed / dst_normed.std(dim=(2, 3), keepdim=True)
        dst_ind = torch.ones_like(dst_normed)
    else:
        dst_normed = _norm_via_validity_mask(images_dst, dst_ind)

    # find the padding needed for the FFT
    src_p_H = int(h1 * 0.25) + 1
    src_p_W = int(w1 * 0.25) + 1

    # pad quarter height on top and bot, and quarter width on left and right
    padding = (
        src_p_W,
        src_p_W,
        src_p_H,
        src_p_H,
    )

    # take powers of the images
    src_1st = src_normed
    src_2nd = src_normed**2
    src_3rd = src_normed**3

    dst_1st = dst_normed
    dst_2nd = dst_normed**2
    dst_3rd = dst_normed**3

    # padding for FFTs. This is not just for "partial overlap". It is required.
    src_0th_padded = torch.nn.functional.pad(src_ind, padding)
    src_1st_padded = torch.nn.functional.pad(src_1st, padding)
    src_2nd_padded = torch.nn.functional.pad(src_2nd, padding)
    src_3rd_padded = torch.nn.functional.pad(src_3rd, padding)

    dst_0th_padded = torch.nn.functional.pad(dst_ind, padding)
    dst_1st_padded = torch.nn.functional.pad(dst_1st, padding)
    dst_2nd_padded = torch.nn.functional.pad(dst_2nd, padding)
    dst_3rd_padded = torch.nn.functional.pad(dst_3rd, padding)

    # calculate the 2D FFT of the indicator images, and conjugate the source
    src_0th_f = rfft2(src_0th_padded).conj()
    src_1st_f = rfft2(src_1st_padded).conj()
    src_2nd_f = rfft2(src_2nd_padded).conj()
    src_3rd_f = rfft2(src_3rd_padded).conj()

    dst_0th_f = rfft2(dst_0th_padded)
    dst_1st_f = rfft2(dst_1st_padded)
    dst_2nd_f = rfft2(dst_2nd_padded)
    dst_3rd_f = rfft2(dst_3rd_padded)

    # calculate the convolution for the cross terms
    olap_f = src_0th_f * dst_0th_f

    m_01_f = src_1st_f * dst_0th_f
    m_10_f = src_0th_f * dst_1st_f

    m_20_f = src_2nd_f * dst_0th_f
    m_11_f = src_1st_f * dst_1st_f
    m_02_f = src_0th_f * dst_2nd_f

    m_30_f = src_3rd_f * dst_0th_f
    m_21_f = src_2nd_f * dst_1st_f
    m_12_f = src_1st_f * dst_2nd_f
    m_03_f = src_0th_f * dst_3rd_f

    # calculate the inverse FFTs for the cross terms
    olap = fftshift(irfft2(olap_f), dim=(-2, -1)).real

    m_01 = fftshift(irfft2(m_01_f), dim=(-2, -1)).real
    m_10 = fftshift(irfft2(m_10_f), dim=(-2, -1)).real

    m_20 = fftshift(irfft2(m_20_f), dim=(-2, -1)).real
    m_11 = fftshift(irfft2(m_11_f), dim=(-2, -1)).real
    m_02 = fftshift(irfft2(m_02_f), dim=(-2, -1)).real

    m_30 = fftshift(irfft2(m_30_f), dim=(-2, -1)).real
    m_21 = fftshift(irfft2(m_21_f), dim=(-2, -1)).real
    m_12 = fftshift(irfft2(m_12_f), dim=(-2, -1)).real
    m_03 = fftshift(irfft2(m_03_f), dim=(-2, -1)).real

    # crop off the padding
    # crop the images to the original size
    olap = olap[:, :, src_p_H : (src_p_H + h1), src_p_W : (src_p_W + w1)]
    # olap_frac = olap / torch.min(
    #     torch.sum(src_ind, dim=(-2, -1), keepdim=True),
    #     torch.sum(dst_ind, dim=(-2, -1), keepdim=True),
    # )
    olap_frac = olap / (h1 * w1)
    olapr = torch.where(
        olap_frac > overlap_threshold,
        1.0 / olap,
        torch.zeros_like(olap),
    )

    m_01 = m_01[:, :, src_p_H : (src_p_H + h1), src_p_W : (src_p_W + w1)] * olapr
    m_10 = m_10[:, :, src_p_H : (src_p_H + h1), src_p_W : (src_p_W + w1)] * olapr

    m_20 = m_20[:, :, src_p_H : (src_p_H + h1), src_p_W : (src_p_W + w1)] * olapr
    m_11 = m_11[:, :, src_p_H : (src_p_H + h1), src_p_W : (src_p_W + w1)] * olapr
    m_02 = m_02[:, :, src_p_H : (src_p_H + h1), src_p_W : (src_p_W + w1)] * olapr

    m_30 = m_30[:, :, src_p_H : (src_p_H + h1), src_p_W : (src_p_W + w1)] * olapr
    m_12 = m_12[:, :, src_p_H : (src_p_H + h1), src_p_W : (src_p_W + w1)] * olapr
    m_21 = m_21[:, :, src_p_H : (src_p_H + h1), src_p_W : (src_p_W + w1)] * olapr
    m_03 = m_03[:, :, src_p_H : (src_p_H + h1), src_p_W : (src_p_W + w1)] * olapr

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

    """
    The 3rd order estimate of the entropy of a univariate changes when the
    variance is not zero. The entropy estimate is structured the same as page
    180 of https://theses.hal.science/tel-00632128 and plug in a general
    distribution with:
     
    Wolfram Alpha query for H_3(x)^2: definite integral -inf to inf of
    (1/sqrt(2*pi*v)) * e^(-(1/(2*v)) * u^2) * (u^3 - 3u)^2

    Result: 3 v (3 + v (-6 + 5 v)) 
    
    Expand: 9 v - 18 v^2 + 15 v^3

    Wolfram Alpha query for H_4(x)^2: definite integral -inf to inf of
    (1/sqrt(2*pi*v)) * e^(-(1/(2*v)) * u^2) * (u^4 - 6u^2 + 3)^2

    Result: 9 + 3 v (-12 + v (42 + 5 v (-12 + 7 v)))
    
    Expand: 9 - 36 v + 126 v^2 - 180 v^3 + 105 v^4

    Wolfram Alpha query for H_6(x)^2: definite integral -inf to inf of
    (1/sqrt(2*pi*v)) * e^(-(1/(2*v)) * u^2) * (u^6 - 15 * u^4 + 45 * u^2 - 15)^2

    Result: 45 (5 + v (-30 + v (165 + v (-460 + 21 v (35 + v (-30 + 11 v))))))

    Expand: 225 - 1350v + 7425v^2 - 20700v^3 + 33075v^4 - 28350v^5 + 10395v^6

    Putting it together

    H_3rd(x) = 0.5 * log(2 * pi * e * k_20) 
    
    - 0.5 * k_30**2 * integral_H3_2nd / (3!)^2

    H_3rd(x) = 0.5 * log(2 * pi * e * k_20) 
    
    - 0.5 * k_30**2 * (9 * k_20 - 18 * k_20**2 + 15 * k_20**3) / 36

    """
    # calculate the marginal entropies
    ent_src = 0.5 * torch.log(2 * torch.pi * torch.e * k_20) - (k_30_2nd / 12.0)
    ent_dst = 0.5 * torch.log(2 * torch.pi * torch.e * k_02) - (k_03_2nd / 12.0)

    # # loop and make a collage plot of all of the terms for debugging
    # # make the figure
    # for termname, term in zip(
    #     [
    #         "src_normed",
    #         "dst_normed",
    #         "ent_dst",
    #         "ent_src",
    #         "ent_joint",
    #         "emi_3e_term_A",
    #         "emi_3e_term_B",
    #         "emi_3e_term_C",
    #         "emi_3e_term_D",
    #         "det_cov",
    #         "k_20",
    #         "k_02",
    #         "k_11",
    #         "k_30",
    #         "k_21",
    #         "k_12",
    #         "k_03",
    #         "k_20*k_02",
    #         "k_11**2",
    #         "k_20 * k_02 - k_11**2",
    #         "(k_20 * k_02 - k_11**2)**-3",
    #     ],
    #     [
    #         src_normed,
    #         dst_normed,
    #         ent_dst,
    #         ent_src,
    #         ent_joint,
    #         emi_3e_term_A,
    #         emi_3e_term_B,
    #         emi_3e_term_C,
    #         emi_3e_term_D,
    #         det_cov,
    #         k_20,
    #         k_02,
    #         k_11,
    #         k_30,
    #         k_21,
    #         k_12,
    #         k_03,
    #         k_20 * k_02,
    #         k_11**2,
    #         k_20 * k_02 - k_11**2,
    #     ],
    # ):
    #     # mask out only where overlap is greater than 0.25
    #     termv = term[[0]][olap_frac[[0]] > 0.25]
    #     # print name and minmax
    #     print(
    #         f"abs min {termv.abs().min().item():.9f}, min {termv.min().item():.9f}, max {termv.max().item():.9f} for {termname}"
    #     )

    # remove negative entropy values as they are not possible
    ent_joint[ent_joint < 0.0] = 0.0
    ent_src[ent_src < 0.0] = 0.0
    ent_dst[ent_dst < 0.0] = 0.0

    # normalize the mutual information
    nmi = _norm_mi(ent_joint, ent_src, ent_dst, norm_mode)

    # replace any NaNs or Infs or Negatives with zeros
    nmi[torch.isnan(nmi) | torch.isinf(nmi) | (nmi < 0.0)] = 0.0

    return nmi


# @torch.compile()
def enmi_4_map_2D(
    images_dst: Tensor,
    images_src: Tensor,
    dst_ind: Optional[Tensor] = None,
    src_ind: Optional[Tensor] = None,
    overlap_threshold: float = 0.5,
    norm_mode: str = "none",
) -> Tensor:
    """

    Calculate 3rd Order Edgeworth Normalized Mutual Information densely
    over shifts, making a "map" of the mutual information as the src image
    is shifted over the dst image. The zero point is the center of the map.

    If dst_ind or src_ind are not provided, then the entire image is used
    to zero mean and unit norm the images. This can completely ruin the
    mutual information calculation if unintended.

    Args:
        images_src: a stack of source images of shape (B, 1, H, W)
        images_dst: a stack of destination images of shape (B, 1, H, W)
        src_ind: indicator mask of the source images' foregrounds. shape (B, 1,
            H, W). If None, the entire image is considered foreground. Defaults
            to None.
        dst_ind: indicator mask of the destination images' foregrounds. shape
            (B, 1, H, W). If None, the entire image is considered foreground.
            Defaults to None.
        normalization_mode: the mode for normalizing the mutual information.
            Options are "none", "Kvalseth Sum", "Kvalseth Min", "Kvalseth Max",
            "Yao", "Strehl-Ghosh", "Logarithmic". Defaults to "none".
        pad_contributes: if True, then the padding pixels contribute to the
            mutual information. If False, then the padding pixels are ignored.
            Defaults to False.

    Returns:
        Tensor: The "estimated" mutual information map: (B, 1, H, W)

    """
    if images_src.ndim != 4 or images_dst.ndim != 4:
        raise ValueError(
            f"Images must be 4D tensors, but got {images_src.ndim} and {images_dst.ndim}"
        )

    b1, c1, h1, w1 = images_dst.shape
    b2, c2, h2, w2 = images_src.shape

    # check that these are 4D tensors and they are grayscale
    if c1 != 1 or c2 != 1:
        raise ValueError(f"Images must be grayscale, but got {c1} and {c2} channels")

    # check that the images are the same size
    if h1 != h2 or w1 != w2:
        raise ValueError(f"Images different size: src {h1}x{w1} and dst {h2}x{w2}")

    # check that the image batch sizes are the same size
    if b1 != b2 and b1 != 1 and b2 != 1:
        raise ValueError(f"Incompatible batch sizes: src {b1} and dst {b2}")

    # we need to zero mean and unit norm the images (only within the mask)
    if src_ind is None:
        src_normed = images_src - images_src.mean(dim=(-1, -2), keepdim=True)
        src_normed = src_normed / src_normed.std(dim=(-1, -2), keepdim=True)
        src_ind = torch.ones_like(src_normed)
    else:
        src_normed = _norm_via_validity_mask(images_src, src_ind)

    if dst_ind is None:
        dst_normed = images_dst - images_dst.mean(dim=(-1, -2), keepdim=True)
        dst_normed = dst_normed / dst_normed.std(dim=(-1, -2), keepdim=True)
        dst_ind = torch.ones_like(dst_normed)
    else:
        dst_normed = _norm_via_validity_mask(images_dst, dst_ind)

    # find the padding needed for the FFT
    src_p_H = int(h1 * 0.25) + 1
    src_p_W = int(w1 * 0.25) + 1

    # pad quarter height on top and bot, and quarter width on left and right
    padding = (
        src_p_W,
        src_p_W,
        src_p_H,
        src_p_H,
    )

    # take powers of the images
    src_1st = src_normed
    src_2nd = src_normed**2
    src_3rd = src_normed**3
    src_4th = src_normed**4

    dst_1st = dst_normed
    dst_2nd = dst_normed**2
    dst_3rd = dst_normed**3
    dst_4th = dst_normed**4

    # padding for FFTs. This is not just for "partial overlap". It is required.
    src_0th_padded = torch.nn.functional.pad(src_ind, padding)
    src_1st_padded = torch.nn.functional.pad(src_1st, padding)
    src_2nd_padded = torch.nn.functional.pad(src_2nd, padding)
    src_3rd_padded = torch.nn.functional.pad(src_3rd, padding)
    src_4th_padded = torch.nn.functional.pad(src_4th, padding)

    dst_0th_padded = torch.nn.functional.pad(dst_ind, padding)
    dst_1st_padded = torch.nn.functional.pad(dst_1st, padding)
    dst_2nd_padded = torch.nn.functional.pad(dst_2nd, padding)
    dst_3rd_padded = torch.nn.functional.pad(dst_3rd, padding)
    dst_4th_padded = torch.nn.functional.pad(dst_4th, padding)

    # calculate the 2D FFT of the indicator images, and conjugate the source
    src_0th_f = rfft2(src_0th_padded).conj()
    src_1st_f = rfft2(src_1st_padded).conj()
    src_2nd_f = rfft2(src_2nd_padded).conj()
    src_3rd_f = rfft2(src_3rd_padded).conj()
    src_4th_f = rfft2(src_4th_padded).conj()

    dst_0th_f = rfft2(dst_0th_padded)
    dst_1st_f = rfft2(dst_1st_padded)
    dst_2nd_f = rfft2(dst_2nd_padded)
    dst_3rd_f = rfft2(dst_3rd_padded)
    dst_4th_f = rfft2(dst_4th_padded)

    # calculate the convolution for the cross terms
    olap_f = src_0th_f * dst_0th_f

    m_01_f = src_1st_f * dst_0th_f
    m_10_f = src_0th_f * dst_1st_f

    m_20_f = src_2nd_f * dst_0th_f
    m_11_f = src_1st_f * dst_1st_f
    m_02_f = src_0th_f * dst_2nd_f

    m_30_f = src_3rd_f * dst_0th_f
    m_21_f = src_2nd_f * dst_1st_f
    m_12_f = src_1st_f * dst_2nd_f
    m_03_f = src_0th_f * dst_3rd_f

    m_40_f = src_4th_f * dst_0th_f
    m_31_f = src_3rd_f * dst_1st_f
    m_22_f = src_2nd_f * dst_2nd_f
    m_13_f = src_1st_f * dst_3rd_f
    m_04_f = src_0th_f * dst_4th_f

    # calculate the inverse FFTs for the cross terms
    olap = fftshift(irfft2(olap_f), dim=(-2, -1)).real

    m_01 = fftshift(irfft2(m_01_f), dim=(-2, -1)).real
    m_10 = fftshift(irfft2(m_10_f), dim=(-2, -1)).real

    m_20 = fftshift(irfft2(m_20_f), dim=(-2, -1)).real
    m_11 = fftshift(irfft2(m_11_f), dim=(-2, -1)).real
    m_02 = fftshift(irfft2(m_02_f), dim=(-2, -1)).real

    m_30 = fftshift(irfft2(m_30_f), dim=(-2, -1)).real
    m_21 = fftshift(irfft2(m_21_f), dim=(-2, -1)).real
    m_12 = fftshift(irfft2(m_12_f), dim=(-2, -1)).real
    m_03 = fftshift(irfft2(m_03_f), dim=(-2, -1)).real

    m_40 = fftshift(irfft2(m_40_f), dim=(-2, -1)).real
    m_31 = fftshift(irfft2(m_31_f), dim=(-2, -1)).real
    m_22 = fftshift(irfft2(m_22_f), dim=(-2, -1)).real
    m_13 = fftshift(irfft2(m_13_f), dim=(-2, -1)).real
    m_04 = fftshift(irfft2(m_04_f), dim=(-2, -1)).real

    # crop off the padding
    # crop the images to the original size
    olap = olap[:, :, src_p_H : (src_p_H + h1), src_p_W : (src_p_W + w1)]
    # olap_frac = olap / torch.min(
    #     torch.sum(src_ind, dim=(-2, -1), keepdim=True),
    #     torch.sum(dst_ind, dim=(-2, -1), keepdim=True),
    # )
    olap_frac = olap / (h1 * w1)
    olapr = torch.where(
        olap_frac > overlap_threshold,
        torch.reciprocal(olap),
        torch.zeros_like(olap),
    )

    m_01 = m_01[:, :, src_p_H : (src_p_H + h1), src_p_W : (src_p_W + w1)] * olapr
    m_10 = m_10[:, :, src_p_H : (src_p_H + h1), src_p_W : (src_p_W + w1)] * olapr

    m_20 = m_20[:, :, src_p_H : (src_p_H + h1), src_p_W : (src_p_W + w1)] * olapr
    m_11 = m_11[:, :, src_p_H : (src_p_H + h1), src_p_W : (src_p_W + w1)] * olapr
    m_02 = m_02[:, :, src_p_H : (src_p_H + h1), src_p_W : (src_p_W + w1)] * olapr

    m_30 = m_30[:, :, src_p_H : (src_p_H + h1), src_p_W : (src_p_W + w1)] * olapr
    m_12 = m_12[:, :, src_p_H : (src_p_H + h1), src_p_W : (src_p_W + w1)] * olapr
    m_21 = m_21[:, :, src_p_H : (src_p_H + h1), src_p_W : (src_p_W + w1)] * olapr
    m_03 = m_03[:, :, src_p_H : (src_p_H + h1), src_p_W : (src_p_W + w1)] * olapr

    m_40 = m_40[:, :, src_p_H : (src_p_H + h1), src_p_W : (src_p_W + w1)] * olapr
    m_31 = m_31[:, :, src_p_H : (src_p_H + h1), src_p_W : (src_p_W + w1)] * olapr
    m_22 = m_22[:, :, src_p_H : (src_p_H + h1), src_p_W : (src_p_W + w1)] * olapr
    m_13 = m_13[:, :, src_p_H : (src_p_H + h1), src_p_W : (src_p_W + w1)] * olapr
    m_04 = m_04[:, :, src_p_H : (src_p_H + h1), src_p_W : (src_p_W + w1)] * olapr

    # compute the cumulants from raw moments
    k_20 = m_20 - m_10**2
    k_11 = m_11 - m_10 * m_01
    k_02 = m_02 - m_01**2

    k_30 = m_30 - 3 * m_20 * m_10 + 2 * m_10**3
    k_21 = m_21 - m_20 * m_01 - 2 * m_11 * m_10 + 2 * m_10**2 * m_01
    k_12 = m_12 - m_02 * m_10 - 2 * m_11 * m_01 + 2 * m_01**2 * m_10
    k_03 = m_03 - 3 * m_02 * m_01 + 2 * m_01**3

    # fourth order
    k_40 = m_40 - 4 * m_30 * m_10 - 3 * m_20**2 + 12 * m_20 * m_10**2 - 6 * m_10**4
    k_31 = (
        m_31
        - m_30 * m_01
        - 3 * m_21 * m_10
        - 3 * m_20 * m_11
        + 6 * m_20 * m_10 * m_01
        + 6 * m_11 * m_10**2
        - 6 * m_10**3 * m_01
    )
    k_22 = (
        m_22
        - 2 * m_21 * m_01
        + 2 * m_20 * m_01**2
        - m_20 * m_02
        - 2 * m_12 * m_10
        - 2 * m_11**2
        + 8 * m_11 * m_10 * m_01
        - 6 * m_10**2 * m_01**2
        + 2 * m_10**2 * m_02
    )
    k_13 = (
        m_13
        - m_03 * m_10
        - 3 * m_12 * m_01
        - 3 * m_20 * m_11
        + 6 * m_11 * m_01**2
        + 6 * m_11 * m_01**2
        - 6 * m_01**3 * m_10
    )
    k_04 = m_04 - 4 * m_03 * m_01 - 3 * m_02**2 + 12 * m_02 * m_01**2 - 6 * m_01**4

    # standardize the 3rd order and lower cumulants
    k_11 = k_11 / torch.sqrt(k_20 * k_02)
    k_30 = k_30 / (k_20**1.5)
    k_21 = k_21 / (k_20 * torch.sqrt(k_02))
    k_12 = k_12 / (k_02 * torch.sqrt(k_20))
    k_03 = k_03 / (k_02**1.5)

    # 4th order central moments are not equal to their central cumulants
    k_40 = k_40 / k_20**2
    k_31 = k_31 / (k_20**1.5 * torch.sqrt(k_02))
    k_22 = k_22 / (k_20 * k_02)
    k_13 = k_13 / (k_02**1.5 * torch.sqrt(k_20))
    k_04 = k_04 / k_02**2

    # # 4th order central moments are not equal to their central cumulants
    # # however, the below corrections are not valid for some reason
    # # I think this means that standardized cumulants are found from
    # # nonstandard cumulants , not from the central nonstandard moments
    # k_40 = (k_40 + 3 * k_20**2) / k_20**2 - 3 * k_20**2
    # k_31 = (
    #     (k_31 + 3 * k_20 * k_11) / (k_20**1.5 * torch.sqrt(k_02))
    # ) - 3 * k_20 * k_11
    # k_22 = (
    #     ((k_22 + 2 * k_11**2 + 2 * k_20 * k_02) / (k_20 * k_02))
    #     - 2 * k_11**2
    #     - 2 * k_20 * k_02
    # )
    # k_13 = (
    #     (k_13 + 3 * k_02 * k_12) / (k_02**1.5 * torch.sqrt(k_20))
    # ) - 3 * k_02 * k_12
    # k_04 = (k_04 + 3 * k_02**2) / k_02**2 - 3 * k_02**2

    # estimate the joint histogram entropy
    k_11_2nd = k_11**2
    k_11_3rd = k_11**3
    k_11_4th = k_11**4
    k_11_5th = k_11**5
    k_11_6th = k_11**6

    k_30_2nd = k_30**2
    k_21_2nd = k_21**2
    k_12_2nd = k_12**2
    k_03_2nd = k_03**2

    k_30_3rd = k_30**3
    k_03_3rd = k_03**3

    k_30_4th = k_30**4
    k_03_4th = k_03**4

    k_40_2nd = k_40**2
    k_31_2nd = k_31**2
    k_22_2nd = k_22**2
    k_13_2nd = k_13**2
    k_04_2nd = k_04**2

    # terms for 3rd order approximation
    emi_3e_term_A = k_30_2nd + k_03_2nd + 3 * k_12_2nd + 3 * k_21_2nd
    emi_3e_term_B = -6.0 * k_11 * (k_30 * k_21 + k_03 * k_12 + 2 * k_21 * k_12)
    emi_3e_term_C = 6.0 * k_11_2nd * (k_21_2nd + k_12_2nd + k_30 * k_12 + k_03 * k_21)
    emi_3e_term_D = -2.0 * k_11_3rd * (k_30 * k_03 + 3 * k_21 * k_12)

    # calculate the joint entropy
    ent_joint_3rd_order = (
        1.0
        + math.log(2 * torch.pi)
        + 0.5 * torch.log(1 - k_11_2nd)
        - (1.0 / (12.0 * (1 - k_11_2nd) ** 3))
        * (emi_3e_term_A + emi_3e_term_B + emi_3e_term_C + emi_3e_term_D)
    )

    # calculate the marginal entropies
    ent_src = (
        0.5 * math.log(2 * torch.pi * torch.e)
        - (k_30_2nd / 12.0)
        - (k_40_2nd / 48.0)
        - (5.0 * k_30_4th / 72.0)
    )
    ent_dst = (
        0.5 * math.log(2 * torch.pi * torch.e)
        - (k_03_2nd / 12.0)
        - (k_04_2nd / 48.0)
        - (5.0 * k_03_4th / 72.0)
    )

    emi_4e_term_1_prefactor = -(1.0 / (48.0 * (1 - k_11_2nd) ** 4))
    emi_4e_term_2_prefactor = -(10.0 / (144.0 * (1 - k_11_2nd) ** 6))

    emi_4e_term_1A = k_40_2nd**2 + 4 * k_31_2nd + 6 * k_22_2nd + 4 * k_13_2nd + k_04_2nd

    emi_4e_term_1B = (
        -8.0 * k_11 * (k_40 * k_31 + 3 * k_31 * k_22 + 3 * k_22 * k_13 + k_13 * k_04)
    )
    emi_4e_term_1C = (
        12.0
        * k_11_2nd
        * (
            k_31_2nd
            + 2.0 * k_22_2nd
            + k_13_2nd
            + k_40 * k_22
            + 2.0 * k_31 * k_13
            + k_22 * k_04
        )
    )
    emi_4e_term_1D = (
        -8.0
        * k_11_3rd
        * (k_40 * k_13 + 3 * k_31 * k_22 + k_31 * k_04 + 3 * k_22 * k_13)
    )
    emi_4e_term_1E = 2.0 * k_11_4th * (3 * k_22_2nd + k_40 * k_04 + 4 * k_31 * k_13)

    emi_4e_term_2A = (
        k_30_4th
        + 6 * k_30_2nd * k_21_2nd
        + 15 * k_30_2nd * k_12_2nd
        + 20 * k_30_2nd * k_03_2nd
        + 15 * k_21_2nd * k_03_2nd
        + 6 * k_12_2nd * k_03_2nd
        + k_03_4th
    )

    emi_4e_term_2B = (
        -12.0
        * k_11
        * (
            k_30_3rd * k_21
            + 10 * k_30_2nd * k_12 * k_03
            + 5 * k_30_2nd * k_21 * k_12
            + 10 * k_30 * k_21 * k_03_2nd
            + 5 * k_21 * k_12 * k_03_2nd
            + k_12 * k_03_3rd
        )
    )

    emi_4e_term_2C = (
        30.0
        * k_11_2nd
        * (
            k_30_2nd * k_21_2nd
            + 4 * k_30_2nd * k_12_2nd
            + 6 * k_30_2nd * k_03_2nd
            + 4 * k_21_2nd * k_03_2nd
            + k_12_2nd * k_03_2nd
            + k_30_3rd * k_12
            + 4 * k_30_2nd * k_21 * k_03
            + 6 * k_30 * k_21 * k_12 * k_03
            + 4 * k_30 * k_12 * k_03_2nd
            + k_21 * k_03_3rd
        )
    )

    emi_4e_term_2D = (
        -40.0
        * k_11_3rd
        * (
            k_30_3rd * k_03
            + 9 * k_30_2nd * k_12 * k_03
            + 3 * k_30_2nd * k_21 * k_12
            + 3 * k_30 * k_21_2nd * k_03
            + 9 * k_30 * k_21 * k_03_2nd
            + 3 * k_30 * k_12_2nd * k_03
            + 3 * k_21 * k_12 * k_03_2nd
            + k_30 * k_03_3rd
        )
    )

    emi_4e_term_2E = (
        30.0
        * k_11_4th
        * (
            3 * k_30_2nd * k_12_2nd
            + 6 * k_30_2nd * k_03_2nd
            + 3 * k_21_2nd * k_03_2nd
            + 5 * k_30_2nd * k_21 * k_03
            + 10 * k_30 * k_21 * k_12 * k_03
            + 5 * k_30 * k_12 * k_03_2nd
        )
    )

    emi_4e_term_2F = (
        -12.0
        * k_11_5th
        * (
            5 * k_30 * k_21_2nd * k_03
            + 11 * k_30 * k_21 * k_03_2nd
            + 11 * k_30_2nd * k_12 * k_03
            + 5 * k_30 * k_12_2nd * k_03
        )
    )

    emi_4e_term_2G = (
        2.0 * k_11_6th * (11 * k_30_2nd * k_03_2nd + 21 * k_30 * k_21 * k_12 * k_03)
    )

    ent_joint = (
        ent_joint_3rd_order
        + emi_4e_term_1_prefactor
        * (
            emi_4e_term_1A
            + emi_4e_term_1B
            + emi_4e_term_1C
            + emi_4e_term_1D
            + emi_4e_term_1E
        )
        + emi_4e_term_2_prefactor
        * (
            emi_4e_term_2A
            + emi_4e_term_2B
            + emi_4e_term_2C
            + emi_4e_term_2D
            + emi_4e_term_2E
            + emi_4e_term_2F
            + emi_4e_term_2G
        )
    )

    # # loop and make a collage plot of all of the terms for debugging
    # # make the figure
    # for termname, term in zip(
    #     [
    #         "ent_dst",
    #         "ent_src",
    #         "ent_joint",
    #         "emi_3e_term_A",
    #         "emi_3e_term_B",
    #         "emi_3e_term_C",
    #         "emi_3e_term_D",
    #         "emi_4e_term_1A",
    #         "emi_4e_term_1B",
    #         "emi_4e_term_1C",
    #         "emi_4e_term_1D",
    #         "emi_4e_term_1E",
    #         "emi_4e_term_2A",
    #         "emi_4e_term_2B",
    #         "emi_4e_term_2C",
    #         "emi_4e_term_2D",
    #         "emi_4e_term_2E",
    #         "emi_4e_term_2F",
    #         "emi_4e_term_2G",
    #         "emi_4e_term_1_prefactor",
    #         "emi_4e_term_2_prefactor",
    #         "ent_joint_3rd_order",
    #         "k_20",
    #         "k_02",
    #         "k_11",
    #         "k_30",
    #         "k_21",
    #         "k_12",
    #         "k_03",
    #         "k_20*k_02",
    #         "k_11**2",
    #         "(1 - k_11**2)",
    #         "(1 - k_11**2)**-3",
    #         "k_30_4th",
    #         "k_30_2nd",
    #         "k_21_2nd",
    #         "k_12_2nd",
    #         "k_03_2nd",
    #         "k_03_4th",
    #     ],
    #     [
    #         ent_dst,
    #         ent_src,
    #         ent_joint,
    #         emi_3e_term_A,
    #         emi_3e_term_B,
    #         emi_3e_term_C,
    #         emi_3e_term_D,
    #         emi_4e_term_1A,
    #         emi_4e_term_1B,
    #         emi_4e_term_1C,
    #         emi_4e_term_1D,
    #         emi_4e_term_1E,
    #         emi_4e_term_2A,
    #         emi_4e_term_2B,
    #         emi_4e_term_2C,
    #         emi_4e_term_2D,
    #         emi_4e_term_2E,
    #         emi_4e_term_2F,
    #         emi_4e_term_2G,
    #         emi_4e_term_1_prefactor,
    #         emi_4e_term_2_prefactor,
    #         ent_joint_3rd_order,
    #         k_20,
    #         k_02,
    #         k_11,
    #         k_30,
    #         k_21,
    #         k_12,
    #         k_03,
    #         k_20 * k_02,
    #         k_11**2,
    #         (1 - k_11**2),
    #         (1 - k_11**2) ** -3,
    #         k_30_4th,
    #         k_30_2nd,
    #         k_21_2nd,
    #         k_12_2nd,
    #         k_03_2nd,
    #         k_03_4th,
    #     ],
    # ):
    #     # mask out only where overlap is greater than 0.25
    #     termv = term[olap_frac > 0.25]
    #     # print name and minmax
    #     print(
    #         f"abs min {termv.abs().min().item():.9f}, min {termv.min().item():.9f}, max {termv.max().item():.9f} for {termname}"
    #     )

    # print(
    #     f"ent_joint_min {ent_joint.min().item():.9f}, ent_joint_max {ent_joint.max().item():.9f}"
    # )
    # print(
    #     f"ent_src_min {ent_src.min().item():.9f}, ent_src_max {ent_src.max().item():.9f}"
    # )
    # print(
    #     f"ent_dst_min {ent_dst.min().item():.9f}, ent_dst_max {ent_dst.max().item():.9f}"
    # )

    # # remove negative entropy values as they are not possible
    # ent_joint[ent_joint < 0.0] = 0.0
    # ent_src[ent_src < 0.0] = 0.0
    # ent_dst[ent_dst < 0.0] = 0.0

    # normalize the mutual information
    nmi = _norm_mi(ent_joint, ent_src, ent_dst, norm_mode)

    # replace any NaNs or Infs or Negatives with zeros
    nmi[torch.isnan(nmi) | torch.isinf(nmi) | (nmi < 0.0)] = 0.0

    return nmi
