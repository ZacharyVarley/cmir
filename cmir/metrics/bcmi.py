"""

This file contains a PyTorch implementation of the binned normalized mutual
information metric for comparing two images. This file also contains a PyTorch
implementation of densely (like cross correlation) calculated normalized mutual
information metric for comparing two images as one of them shifts in discrete
pixel steps.

------------------------------------------------------------------------------
-------------------------------- Important -----------------------------------
------------------------------------------------------------------------------
The dense binned nmi approximation approach requires a quadratic number of cross
correlations with respect to the binning. For 16 bins, this is 256 cross
correlations. For reference the new approach that I have implemented using a
series approximation, and requires 6 cross correlations for both 3rd and 4th
order. The 4th order is more accurate, but the 3rd order is faster. Further, it
plays nicely with the automatic differentation engine (no Parzen windowing).
------------------------------------------------------------------------------
-------------------------------- Important -----------------------------------
------------------------------------------------------------------------------

The binned approach comes from the following literature:

--- 1997: Mutual information established for multimodal image registration.

Maes, Frederik, Andre Collignon, Dirk Vandermeulen, Guy Marchal, and Paul
Suetens. "Multimodality image registration by maximization of mutual
information." IEEE transactions on Medical Imaging 16, no. 2 (1997): 187-198.

--- 2005: A little known publication of Jonas August and Takeo Kanade at CMU
explains how mutual information can be densely computed via FFTs to quickly
count bin occupancy in the joint histogram over discrete translations.

August, Jonas, and Takeo Kanade. "The role of non-overlap in image
registration." In Biennial International Conference on Information Processing in
Medical Imaging, pp. 713-724. Berlin, Heidelberg: Springer Berlin Heidelberg,
2005. 

--- 2021: The same 2005 idea is iterated and extended by Öfverstedt et al.

Öfverstedt, Johan, Joakim Lindblad, and Nataša Sladoje. "Fast computation of
mutual information in the frequency domain with applications to global
multimodal image alignment." Pattern Recognition Letters 159 (2022): 196-203.

"""

from typing import Optional
import torch
from torch import Tensor
from torch.fft import rfft2, irfft2, fftshift


@torch.jit.script
def binned_nmi(
    x: Tensor,
    y: Tensor,
    binsx: int,
    binsy: int,
    mode: str = "nats",
    norm_mode: str = "none",
    exclude_lowest_bin: bool = False,
) -> float:
    """
    Calculate the mutual information between two variables x and y using PyTorch,
    with an option to exclude the lowest bin.

    Args:
        x: torch Tensor of shape (batch_size, ...), assumes binning 0 to 1
        y: torch Tensor of shape (batch_size, ...), assumes binning 0 to 1
        binsx: number of bins for x
        binsy: number of bins for y
        mode: 'nats' or 'bits'. This determines the units of the mutual information.
        normalization_mode: 'Yao', 'Kvalseth Sum', 'Kvalseth Min', 'Kvalseth Max', 'Strehl-Ghosh', 'Logarithmic', 'none'
        exclude_lowest_bin: If True, the lowest bin will be excluded from the calculation.
    """

    x = x.view(-1)
    y = y.view(-1)

    x_binned = torch.floor(x * binsx)
    y_binned = torch.floor(y * binsy)

    # Create a combined variable for 1D histogram
    xy_combined = x_binned * binsy + y_binned

    # Calculate the 1D histogram using a flattening trick so that it can run on the GPU
    hist = torch.histc(xy_combined, bins=(binsx * binsy), min=0, max=(binsx * binsy))
    joint_hist = hist.view(binsx, binsy)

    # Exclude the lowest bin if the flag is set
    if exclude_lowest_bin:
        joint_hist_trimmed = joint_hist[1:, 1:]
    else:
        joint_hist_trimmed = joint_hist

    # if you do this step before trimming the lowest bin, then you will get
    # a very different result and it will not be clear why :(
    joint_hist_trimmed /= joint_hist_trimmed.sum()

    # Calculate the marginal histograms
    marginal_x = joint_hist_trimmed.sum(dim=1)
    marginal_y = joint_hist_trimmed.sum(dim=0)

    # normalize the marginal histograms
    marginal_x /= marginal_x.sum()
    marginal_y /= marginal_y.sum()

    # Calculate the mutual information
    marginal_x_non_zero = marginal_x[marginal_x > 0]
    marginal_y_non_zero = marginal_y[marginal_y > 0]
    joint_non_zero = joint_hist_trimmed[joint_hist_trimmed > 0]

    # Calculate the entropies
    if mode == "bits":
        ent_x = -torch.sum(marginal_x_non_zero * torch.log2(marginal_x_non_zero))
        ent_y = -torch.sum(marginal_y_non_zero * torch.log2(marginal_y_non_zero))
        ent_joint = -torch.sum(joint_non_zero * torch.log2((joint_non_zero)))
    else:
        ent_x = -torch.sum(marginal_x_non_zero * torch.log(marginal_x_non_zero))
        ent_y = -torch.sum(marginal_y_non_zero * torch.log(marginal_y_non_zero))
        ent_joint = -torch.sum(joint_non_zero * torch.log((joint_non_zero)))

    # formula with -1 as the values have not been multiplied by -1
    mi = ent_x + ent_y - ent_joint

    # if the normalization mode is "none" then return the distances
    if norm_mode == "none":
        nmi = mi
    elif norm_mode == "Kvalseth Sum":  # divide by marginal arithmetic mean
        nmi = 2.0 * mi / (ent_x + ent_y)
    elif norm_mode == "Kvalseth Min":  # divide by max marginal
        nmi = mi / torch.min(ent_x, ent_y)
    elif norm_mode == "Kvalseth Max":  # divide by min marginal
        nmi = mi / torch.max(ent_x, ent_y)
    elif norm_mode == "Yao":  # divide by joint
        nmi = mi / ent_joint
    elif norm_mode == "Strehl-Ghosh":  # divide by marginal geometric mean
        nmi = mi / torch.sqrt(ent_x * ent_y)
    elif norm_mode == "Logarithmic":  # divide by marginal log mean
        # handle case if ent_src and ent_dst are the same
        nmi = mi / torch.where(
            (ent_x - ent_y).abs() < 1e-6,
            ent_y,
            (ent_x - ent_y) / (torch.log(ent_x) - torch.log(ent_y)),
        )
    else:
        raise ValueError(f"Unknown normalization mode: {norm_mode}")
    return nmi


@torch.jit.script
def batch_binned_nmi(
    x: Tensor,
    y: Tensor,
    binsx: int,
    binsy: int,
    mode: str = "nats",
    norm_mode: str = "none",
    exclude_lowest_bin: bool = False,
) -> Tensor:
    """
    Calculate the mutual information for batches of x and y along the first dimension,
    with broadcasting if one of them has a batch size of 1.

    Args:
        x: torch Tensor of shape (batch_size, ...), assumes binning 0 to 1
        y: torch Tensor of shape (batch_size, ...), assumes binning 0 to 1
        binsx: number of bins for x
        binsy: number of bins for y
        mode: 'nats' or 'bits'. This determines the units of the mutual information.
        normalization_mode: 'Yao', 'Kvalseth Sum', 'Kvalseth Min', 'Kvalseth Max', 'Strehl-Ghosh', 'Logarithmic', 'none'
        exclude_lowest_bin: If True, the lowest bin will be excluded from the calculation.
    """

    batch_size_x = x.shape[0]
    batch_size_y = y.shape[0]

    # Determine the batch size for iteration
    if batch_size_x != batch_size_y:
        if batch_size_x == 1 or batch_size_y == 1:
            batch_size = max(batch_size_x, batch_size_y)
        else:
            raise ValueError(
                "Batch sizes must be equal, or one of them must be 1 for broadcasting."
            )
    else:
        batch_size = batch_size_x

    mi_values = torch.empty(batch_size, dtype=x.dtype, device=x.device)

    for i in range(batch_size):
        xi = x if batch_size_x == 1 else x[i]
        yi = y if batch_size_y == 1 else y[i]
        mi_values[i] = binned_nmi(
            xi,
            yi,
            binsx,
            binsy,
            mode=mode,
            norm_mode=norm_mode,
            exclude_lowest_bin=exclude_lowest_bin,
        )

    return mi_values


@torch.jit.script
def bnmi_map_2D(
    images_dst: Tensor,
    images_src: Tensor,
    dst_ind: Optional[Tensor] = None,
    src_ind: Optional[Tensor] = None,
    bins_dst: int = 8,
    bins_src: int = 8,
    norm_mode: str = "none",
    overlap_threshold: float = 0.5,
    logmode: str = "log",
) -> Tensor:
    """
    Calculate the mutual information over discrete translation between two images using the 2D FFT.

    Args:
        images_src: a stack of source images of shape (B, C, H, W)
        images_dst: a stack of destination images of shape (B, C, H, W)
        bins_src: the number of bins to use for the source image
        bins_dst: the number of bins to use for the destination image
        overlap_min_fraction: the minimum fraction of overlap between the images
        background_src: the background level for the source image. If -1 then
            none of the pixel values are treated as background.
        background_dst: the background level for the destination image. If -1 then
            none of the pixel values are treated as background.
        logmode: the log mode to use for the entropy calculation
            Options are "log", "log2", "log10"
        normalization_mode: the normalization mode to use for the mutual information.
            Options are "none", "Kvalseth Sum", "Kvalseth Min", "Kvalseth Max", "Yao",
            "Strehl-Ghosh", "Logarithmic" (see Notes)

    Returns:
        Tensor: the mutual information map between the images


    Notes:

    For mutual information normalization approaches, see:

    Amelio, Alessia, and Clara Pizzuti. "Correction for closeness: Adjusting
    normalized mutual information measure for clustering comparison."
    Computational Intelligence 33.3 (2017): 579-601.

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

    src_levels = (bins_src * images_src).byte().repeat(1, bins_src, 1, 1)
    src_levels = (
        src_levels
        == torch.arange(bins_src, dtype=torch.uint8, device=images_src.device)[
            None, :, None, None
        ]
    )

    dst_levels = (bins_dst * images_dst).byte().repeat(1, bins_dst, 1, 1)
    dst_levels = (
        dst_levels
        == torch.arange(bins_dst, dtype=torch.uint8, device=images_dst.device)[
            None, :, None, None
        ]
    )

    # we need to zero mean and unit norm the images (only within the mask)
    if src_ind is None:
        src_ind = torch.ones(
            (b1, 1, h1, w1), dtype=torch.bool, device=images_src.device
        )

    if dst_ind is None:
        dst_ind = torch.ones(
            (b1, 1, h1, w1), dtype=torch.bool, device=images_src.device
        )

    src_levels = torch.cat(
        (
            src_ind,
            src_levels,
        ),
        dim=1,
    )

    dst_levels = torch.cat(
        (
            dst_ind,
            dst_levels,
        ),
        dim=1,
    )

    # calculate the padding fraction along each dimension that will be needed for the
    # images to partially overlap by the specified amount
    # find the padding needed for the FFT
    src_p_H = int(h1 * 0.25) + 1
    src_p_W = int(w1 * 0.25) + 1

    # pad by a quarter of the height on each side and a quarter of the width on each side
    padding = (
        src_p_W,
        src_p_W,
        src_p_H,
        src_p_H,
    )

    # pad the images
    src_levels = torch.nn.functional.pad(src_levels, padding)
    dst_levels = torch.nn.functional.pad(dst_levels, padding)

    # calculate the 2D FFT of the indicator images
    src_levels_fft = rfft2(src_levels.float()).conj()
    dst_levels_fft = rfft2(dst_levels.float())

    # element-wise multiply every possible pair of slices
    # the result is a tensor of shape (B, 1 + num_bins_src, 1 + num_bins_dst, H, W)
    # where the "1 +" are coming from the indicator bins for the image foreground
    cross_corr_f = torch.einsum("bihw,bjhw->bijhw", src_levels_fft, dst_levels_fft)

    # Inverse 2D real valued fourier transform and shift and abs
    cross_corr_real = fftshift(irfft2(cross_corr_f), dim=(-2, -1)).real

    # remove the padding
    cross_corr_real = cross_corr_real[
        :, :, :, src_p_H : (src_p_H + h1), src_p_W : (src_p_W + w1)
    ]

    # where the bin occupancy was less than 1, set it to zero
    cross_corr_real[cross_corr_real < 1.0] = 0.0

    # extract out the joint histograms
    joint_hists = cross_corr_real[:, 1:, 1:, :, :]

    # extract out the marginal histograms
    marg_hists_src = cross_corr_real[:, 1:, 0, :, :]
    marg_hists_dst = cross_corr_real[:, 0, 1:, :, :]

    olap_mask = (cross_corr_real[:, 0, 0, :, :] > overlap_threshold * h1 * w1).float()[
        :, None, :, :
    ]

    # individually normalize the histograms
    joint_hists /= joint_hists.sum(dim=(1, 2), keepdim=True)
    marg_hists_src /= marg_hists_src.sum(dim=1, keepdim=True)
    marg_hists_dst /= marg_hists_dst.sum(dim=1, keepdim=True)

    # calculate the entropy of the joint histograms
    # must be careful to ignore the zero bins
    if logmode == "log":
        joint_p_log_p = torch.where(
            joint_hists > 0.0,
            joint_hists * torch.log(joint_hists),
            torch.zeros_like(joint_hists),
        )
        marg_src_p_log_p = torch.where(
            marg_hists_src > 0.0,
            marg_hists_src * torch.log(marg_hists_src),
            torch.zeros_like(marg_hists_src),
        )
        marg_dst_p_log_p = torch.where(
            marg_hists_dst > 0.0,
            marg_hists_dst * torch.log(marg_hists_dst),
            torch.zeros_like(marg_hists_dst),
        )
    elif logmode == "log2":
        joint_p_log_p = torch.where(
            joint_hists > 0.0,
            joint_hists * torch.log2(joint_hists),
            torch.zeros_like(joint_hists),
        )
        marg_src_p_log_p = torch.where(
            marg_hists_src > 0.0,
            marg_hists_src * torch.log2(marg_hists_src),
            torch.zeros_like(marg_hists_src),
        )
        marg_dst_p_log_p = torch.where(
            marg_hists_dst > 0.0,
            marg_hists_dst * torch.log2(marg_hists_dst),
            torch.zeros_like(marg_hists_dst),
        )
    elif logmode == "log10":
        joint_p_log_p = torch.where(
            joint_hists > 0.0,
            joint_hists * torch.log10(joint_hists),
            torch.zeros_like(joint_hists),
        )
        marg_src_p_log_p = torch.where(
            marg_hists_src > 0.0,
            marg_hists_src * torch.log10(marg_hists_src),
            torch.zeros_like(marg_hists_src),
        )
        marg_dst_p_log_p = torch.where(
            marg_hists_dst > 0.0,
            marg_hists_dst * torch.log10(marg_hists_dst),
            torch.zeros_like(marg_hists_dst),
        )
    else:
        raise ValueError(f"Unknown logmode: {logmode}")

    # find the entropies of the marginals and the joint
    ent_src = -torch.sum(marg_src_p_log_p, dim=1)
    ent_dst = -torch.sum(marg_dst_p_log_p, dim=1)
    ent_joint = -torch.sum(joint_p_log_p, dim=(1, 2))

    # mutual information is the sum of the marginal entropies minus the joint entropy
    mi = (ent_src + ent_dst - ent_joint) * olap_mask

    # replace any NaNs or Infs or Negatives with zeros
    mi[torch.isnan(mi) | torch.isinf(mi) | (mi < 0.0)] = 0.0

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
