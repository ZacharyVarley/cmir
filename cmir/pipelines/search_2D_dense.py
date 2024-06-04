"""
This file implements logic to surf around in homography warp space, until
suitable initial conditions are found for derivative based local optimization.
Translation is removed from this search space, as it is densely checked in the
global search for every single candidate transformation.

"""

import torch
from typing import Optional, Tuple
from torch import Tensor
from cmir.metrics.ecmi import enmi_4_map_2D, enmi_3_map_2D
from cmir.metrics.bcmi import bnmi_map_2D
from cmir.warps.lie_homography_exp import LieHomographyLayer
from cmir.warps.apply_homography import (
    apply_homographies_to_images,
    denormalize_homography,
)
from cmir.optimizers.cmaes import CMAESWrapper
from cmir.optimizers.progessbar import ProgressBar


@torch.jit.script
def std_norm_via_validity_mask(images: Tensor, masks: Tensor) -> Tensor:
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


def search_homography_cmaes_enmi(
    img_dst: torch.Tensor,
    img_src: torch.Tensor,
    dst_ind: Optional[torch.Tensor],
    src_ind: Optional[torch.Tensor],
    pad_frac: float,
    n_iterations: int,
    enmi_order: int,
    cmaes_pop: int,
    cmaes_sigma: float,
    guess_lie: Optional[Tensor] = None,
    x_translation_weight: float = 1.0,
    y_translation_weight: float = 1.0,
    rotation_weight: float = 1.0,
    scale_weight: float = 1.0,
    stretch_weight: float = 1.0,
    shear_weight: float = 1.0,
    x_keystone_weight: float = 1.0,
    y_keystone_weight: float = 1.0,
    verbose: bool = True,
):
    """
    This function fits a homography between two images using CMA-ES and a grid of patches
    to use for CCA calculations.

    Args:
        n_iterations (int): Number of iterations to run CMA-ES.
        enmi_order (int): Order of the enmi calculation. Must be 3 or 4.
        cmaes_population (int): Number of candidate solutions to generate per iteration.
        cmaes_sigma (float): Sigma parameter for CMA-ES.
        img_src (torch.Tensor): Source image.
        img_dst (torch.Tensor): Destination image.
        guess_lie (Optional[Tensor]): Initial guess for the lie vector. If None, then
            the lie vector is initialized to zero.
        x_translation_weight (float): Weight for the x translation parameter.
        y_translation_weight (float): Weight for the y translation parameter.
        rotation_weight (float): Weight for the rotation parameter.
        scale_weight (float): Weight for the isotropic scale parameter.
        stretch_weight (float): Weight for the anisotropic stretch parameter.
        shear_weight (float): Weight for the shear parameter.
        x_keystone_weight (float): Weight for the x keystone parameter.
        y_keystone_weight (float): Weight for the y keystone parameter.

    Returns:
        A tuple of the best homography and the best lie algebra vector.

    """

    if guess_lie is None:
        guess_lie = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            device=img_src.device,
        )
    else:
        # if its a 1x8 vector, then make it a 8 vector
        if len(guess_lie.shape) > 1:
            guess_lie = guess_lie.squeeze(0)

    # wrapper for CMA-ES optimizer
    cmaes_wrap = CMAESWrapper(
        device=img_src.device,
        dimension=8,
        population_size=cmaes_pop,
        mean=guess_lie,
        sigma=cmaes_sigma,
        minimize=False,
    )

    # class for homography
    lie_homog = LieHomographyLayer(
        x_translation_weight=x_translation_weight,
        y_translation_weight=y_translation_weight,
        rotation_weight=rotation_weight,
        scale_weight=scale_weight,
        stretch_weight=stretch_weight,
        shear_weight=shear_weight,
        x_keystone_weight=x_keystone_weight,
        y_keystone_weight=y_keystone_weight,
    ).to(img_src.device)

    if src_ind is None:
        src_ind = torch.ones(
            (1, 1, img_src.shape[-2], img_src.shape[-1]),
            device=img_src.device,
            dtype=torch.float32,
        )
    if dst_ind is None:
        dst_ind = torch.ones(
            (1, 1, img_dst.shape[-2], img_dst.shape[-1]),
            device=img_dst.device,
            dtype=torch.float32,
        )

    # find the output shape after padding
    pad_h_dst = int(img_dst.shape[-2] * pad_frac)
    pad_w_dst = int(img_dst.shape[-1] * pad_frac)
    pad_h_src = int(img_src.shape[-2] * pad_frac)
    pad_w_src = int(img_src.shape[-1] * pad_frac)

    dst_padding = (pad_w_dst, pad_w_dst, pad_h_dst, pad_h_dst)
    src_padding = (pad_w_src, pad_w_src, pad_h_src, pad_h_src)

    # pad the images
    img_dst_padded = torch.nn.functional.pad(
        img_dst, dst_padding, mode="constant", value=0.0
    )
    img_src_padded = torch.nn.functional.pad(
        img_src, src_padding, mode="constant", value=0.0
    )

    # pad the indicators
    src_ind_padded = torch.nn.functional.pad(
        src_ind, src_padding, mode="constant", value=0.0
    )
    dst_ind_padded = torch.nn.functional.pad(
        dst_ind, dst_padding, mode="constant", value=0.0
    )

    # compute the current score and tell the optimizer
    if enmi_order == 4:
        at_dst_map = enmi_4_map_2D(
            img_dst_padded,
            img_src_padded,
            dst_ind_padded,
            src_ind_padded,
        )
    elif enmi_order == 3:
        at_dst_map = enmi_3_map_2D(
            img_dst_padded,
            img_src_padded,
            dst_ind_padded,
            src_ind_padded,
        )
    else:
        raise ValueError(f"enmi_order must be 3 or 4, got {enmi_order}")

    # compute the current score and tell the optimizer
    if enmi_order == 4:
        at_src_map = enmi_4_map_2D(
            img_src_padded,
            img_dst_padded,
            src_ind_padded,
            dst_ind_padded,
        )
    elif enmi_order == 3:
        at_src_map = enmi_3_map_2D(
            img_src_padded,
            img_dst_padded,
            src_ind_padded,
            dst_ind_padded,
        )

    # tell the optimizer the fitnesses
    dst_score = torch.max(at_dst_map.reshape((1, -1)), dim=-1).values
    src_score = torch.max(at_src_map.reshape((1, -1)), dim=-1).values

    scores = torch.min(src_score, dst_score)

    cmaes_wrap.first_tell(guess_lie.unsqueeze(0), scores)

    pb = ProgressBar(range(n_iterations), prefix="CMA-ES: ", size=60)

    for _ in pb:
        # ask for candidate solutions
        candidates = cmaes_wrap.ask()

        # denormalize the homography
        homographies = denormalize_homography(
            lie_homog(candidates), img_src.shape[-2:], img_dst.shape[-2:]
        ).float()

        # offset the homography into the center of the canvas by the padding amount
        homographies[..., 0, 2] += pad_w_dst
        homographies[..., 1, 2] += pad_h_dst

        # compute the fitness of the candidates
        img_src_warp = apply_homographies_to_images(
            img_src, homographies, img_dst_padded.shape[-2:]
        )

        img_dst_warp = apply_homographies_to_images(
            img_dst, torch.inverse(homographies), img_src_padded.shape[-2:]
        )

        # warp the indicators
        src_ind_warp = apply_homographies_to_images(
            src_ind,
            homographies,
            img_dst_padded.shape[-2:],
            mode="nearest",
            padding_mode="zeros",
        )
        dst_ind_warp = apply_homographies_to_images(
            dst_ind,
            torch.inverse(homographies),
            img_src_padded.shape[-2:],
            mode="nearest",
            padding_mode="zeros",
        )

        # densely compute a mutual information map (analogous to cross correlation)
        if enmi_order == 4:
            dst_score = torch.max(
                enmi_4_map_2D(
                    img_dst_padded,
                    img_src_warp,
                    dst_ind_padded,
                    src_ind_warp,
                ).reshape((cmaes_pop, -1)),
                dim=-1,
            ).values
            src_score = torch.max(
                enmi_4_map_2D(
                    img_dst_warp,
                    img_src_padded,
                    dst_ind_warp,
                    src_ind_padded,
                ).reshape((cmaes_pop, -1)),
                dim=-1,
            ).values
        elif enmi_order == 3:
            dst_score = torch.max(
                enmi_3_map_2D(
                    img_dst_padded,
                    img_src_warp,
                    dst_ind_padded,
                    src_ind_warp,
                ).reshape((cmaes_pop, -1)),
                dim=-1,
            ).values
            src_score = torch.max(
                enmi_3_map_2D(
                    img_dst_warp,
                    img_src_padded,
                    dst_ind_warp,
                    src_ind_padded,
                ).reshape((cmaes_pop, -1)),
                dim=-1,
            ).values
        else:
            raise ValueError(f"enmi_order must be 3 or 4, got {enmi_order}")

        # we optimize the minimum across the source moved into the destination canvas
        # and the inverse: the destination moved into the source canvas. This is very
        # important to rule out nonsensical solutions that spuriously maximize enmi.
        scores = torch.min(src_score, dst_score)
        # scores = 0.5 * (src_score + dst_score)
        # scores = torch.max(src_score, dst_score)
        # scores = (src_score * dst_score) ** 0.5

        # tell the CMA-ES optimizer the fitnesses
        cmaes_wrap.tell(candidates, scores)

        if verbose:
            pb.set_description(f"score: {cmaes_wrap.get_best_fitness():.9f}")

    best_lie_solution = cmaes_wrap.get_best_solution()
    best_homography = denormalize_homography(
        lie_homog(best_lie_solution),
        img_src.shape[-2:],
        img_dst.shape[-2:],
    )

    best_homography[..., 0, 2] += pad_w_dst
    best_homography[..., 1, 2] += pad_h_dst

    # recalculate the map using the best homography
    img_src_warp = apply_homographies_to_images(
        img_src, best_homography, img_dst_padded.shape[-2:]
    )

    img_dst_warp = apply_homographies_to_images(
        img_dst, torch.inverse(best_homography), img_src_padded.shape[-2:]
    )

    # warp the indicators
    src_ind_warp = apply_homographies_to_images(
        src_ind,
        best_homography,
        img_dst_padded.shape[-2:],
        mode="nearest",
        padding_mode="zeros",
    )

    dst_ind_warp = apply_homographies_to_images(
        dst_ind,
        torch.inverse(best_homography),
        img_src_padded.shape[-2:],
        mode="nearest",
        padding_mode="zeros",
    )

    # densely compute a mutual information map (analogous to cross correlation)
    if enmi_order == 4:
        at_dst_map = enmi_4_map_2D(
            img_dst_padded,
            img_src_warp,
            dst_ind_padded,
            src_ind_warp,
        )
        at_src_map = enmi_3_map_2D(
            img_src_padded,
            img_dst_warp,
            src_ind_padded,
            dst_ind_warp,
        )
    elif enmi_order == 3:
        at_dst_map = enmi_3_map_2D(
            img_dst_padded,
            img_src_warp,
            dst_ind_padded,
            src_ind_warp,
        )
        at_src_map = enmi_3_map_2D(
            img_src_padded,
            img_dst_warp,
            src_ind_padded,
            dst_ind_warp,
        )

    # remove the padding offset from the homography
    best_homography[..., 0, 2] -= pad_w_dst
    best_homography[..., 1, 2] -= pad_h_dst

    return best_homography, best_lie_solution, at_dst_map, at_src_map


def search_homography_cmaes_bnmi(
    img_dst: torch.Tensor,
    img_src: torch.Tensor,
    dst_ind: Optional[torch.Tensor],
    src_ind: Optional[torch.Tensor],
    pad_frac: float,
    n_iterations: int,
    bnmi_bins: int,
    cmaes_pop: int,
    cmaes_sigma: float,
    guess_lie: Optional[Tensor] = None,
    x_translation_weight: float = 1.0,
    y_translation_weight: float = 1.0,
    rotation_weight: float = 1.0,
    scale_weight: float = 1.0,
    stretch_weight: float = 1.0,
    shear_weight: float = 1.0,
    x_keystone_weight: float = 1.0,
    y_keystone_weight: float = 1.0,
    verbose: bool = True,
):
    """
    This function fits a homography between two images using CMA-ES and a grid of patches
    to use for CCA calculations.

    Args:
        n_iterations (int): Number of iterations to run CMA-ES.
        enmi_order (int): Order of the enmi calculation. Must be 3 or 4.
        cmaes_population (int): Number of candidate solutions to generate per iteration.
        cmaes_sigma (float): Sigma parameter for CMA-ES.
        img_src (torch.Tensor): Source image.
        img_dst (torch.Tensor): Destination image.
        guess_lie (Optional[Tensor]): Initial guess for the lie vector. If None, then
            the lie vector is initialized to zero.
        x_translation_weight (float): Weight for the x translation parameter.
        y_translation_weight (float): Weight for the y translation parameter.
        rotation_weight (float): Weight for the rotation parameter.
        scale_weight (float): Weight for the isotropic scale parameter.
        stretch_weight (float): Weight for the anisotropic stretch parameter.
        shear_weight (float): Weight for the shear parameter.
        x_keystone_weight (float): Weight for the x keystone parameter.
        y_keystone_weight (float): Weight for the y keystone parameter.

    Returns:
        A tuple of the best homography and the best lie algebra vector.

    """

    if guess_lie is None:
        guess_lie = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            device=img_src.device,
        )
    else:
        # if its a 1x8 vector, then make it a 8 vector
        if len(guess_lie.shape) > 1:
            guess_lie = guess_lie.squeeze(0)

    # wrapper for CMA-ES optimizer
    cmaes_wrap = CMAESWrapper(
        device=img_src.device,
        dimension=8,
        population_size=cmaes_pop,
        mean=guess_lie,
        sigma=cmaes_sigma,
        minimize=False,
    )

    # class for homography
    lie_homog = LieHomographyLayer(
        x_translation_weight=x_translation_weight,
        y_translation_weight=y_translation_weight,
        rotation_weight=rotation_weight,
        scale_weight=scale_weight,
        stretch_weight=stretch_weight,
        shear_weight=shear_weight,
        x_keystone_weight=x_keystone_weight,
        y_keystone_weight=y_keystone_weight,
    ).to(img_src.device)

    if src_ind is None:
        src_ind = torch.ones(
            (1, 1, img_src.shape[-2], img_src.shape[-1]),
            device=img_src.device,
            dtype=torch.float32,
        )
    if dst_ind is None:
        dst_ind = torch.ones(
            (1, 1, img_dst.shape[-2], img_dst.shape[-1]),
            device=img_dst.device,
            dtype=torch.float32,
        )

    # find the output shape after padding
    pad_h_dst = int(img_dst.shape[-2] * pad_frac)
    pad_w_dst = int(img_dst.shape[-1] * pad_frac)
    pad_h_src = int(img_src.shape[-2] * pad_frac)
    pad_w_src = int(img_src.shape[-1] * pad_frac)

    dst_padding = (pad_w_dst, pad_w_dst, pad_h_dst, pad_h_dst)
    src_padding = (pad_w_src, pad_w_src, pad_h_src, pad_h_src)

    # pad the images
    img_dst_padded = torch.nn.functional.pad(
        img_dst, dst_padding, mode="constant", value=0.0
    )
    img_src_padded = torch.nn.functional.pad(
        img_src, src_padding, mode="constant", value=0.0
    )

    # pad the indicators
    src_ind = torch.nn.functional.pad(src_ind, src_padding, mode="constant", value=0.0)
    dst_ind = torch.nn.functional.pad(dst_ind, dst_padding, mode="constant", value=0.0)

    pb = ProgressBar(range(n_iterations), prefix="CMA-ES: ", size=60)

    for _ in pb:
        # ask for candidate solutions
        candidates = cmaes_wrap.ask()

        # denormalize the homography
        homographies = denormalize_homography(
            lie_homog(candidates), img_src_padded.shape[-2:], img_dst_padded.shape[-2:]
        ).float()

        # compute the fitness of the candidates
        img_src_warp = apply_homographies_to_images(
            img_src_padded, homographies, img_dst_padded.shape[-2:]
        )

        img_dst_warp = apply_homographies_to_images(
            img_dst_padded, torch.inverse(homographies), img_src_padded.shape[-2:]
        )

        # warp the indicators
        src_ind_warp = apply_homographies_to_images(
            src_ind,
            homographies,
            img_dst_padded.shape[-2:],
            mode="nearest",
            padding_mode="zeros",
        )
        dst_ind_warp = apply_homographies_to_images(
            dst_ind,
            torch.inverse(homographies),
            img_src_padded.shape[-2:],
            mode="nearest",
            padding_mode="zeros",
        )

        # densely compute a mutual information map (analogous to cross correlation)
        dst_score = torch.max(
            bnmi_map_2D(
                img_dst_padded,
                img_src_warp,
                dst_ind,
                src_ind_warp,
                bins_dst=bnmi_bins,
                bins_src=bnmi_bins,
            ).reshape((cmaes_pop, -1)),
            dim=-1,
        ).values
        src_score = torch.max(
            bnmi_map_2D(
                img_dst_warp,
                img_src_padded,
                dst_ind_warp,
                src_ind,
                bins_dst=bnmi_bins,
                bins_src=bnmi_bins,
            ).reshape((cmaes_pop, -1)),
            dim=-1,
        ).values

        # we optimize the minimum across the source moved into the destination canvas
        # and the inverse: the destination moved into the source canvas. This is very
        # important to rule out nonsensical solutions that spuriously maximize enmi.
        # scores = torch.min(src_score, dst_score)
        scores = 0.5 * (src_score + dst_score)

        # tell the CMA-ES optimizer the fitnesses
        cmaes_wrap.tell(candidates, scores)

        if verbose:
            pb.set_description(f"score: {cmaes_wrap.get_best_fitness():.9f}")

    best_lie_solution = cmaes_wrap.get_best_solution()
    best_homography = denormalize_homography(
        lie_homog(best_lie_solution),
        img_src_padded.shape[-2:],
        img_dst_padded.shape[-2:],
    )

    # recalculate the map using the best homography
    img_src_warp = apply_homographies_to_images(
        img_src_padded, best_homography, img_dst_padded.shape[-2:]
    )

    img_dst_warp = apply_homographies_to_images(
        img_dst_padded, torch.inverse(best_homography), img_src_padded.shape[-2:]
    )

    # warp the indicators
    src_ind_warp = apply_homographies_to_images(
        src_ind,
        best_homography,
        img_dst_padded.shape[-2:],
        mode="nearest",
        padding_mode="zeros",
    )

    dst_ind_warp = apply_homographies_to_images(
        dst_ind,
        torch.inverse(best_homography),
        img_src_padded.shape[-2:],
        mode="nearest",
        padding_mode="zeros",
    )

    # densely compute a mutual information map (analogous to cross correlation)
    at_dst_map = bnmi_map_2D(
        img_dst_padded,
        img_src_warp,
        dst_ind,
        src_ind_warp,
        bins_dst=bnmi_bins,
        bins_src=bnmi_bins,
    )
    at_src_map = bnmi_map_2D(
        img_src_padded,
        img_dst_warp,
        src_ind,
        dst_ind_warp,
        bins_dst=bnmi_bins,
        bins_src=bnmi_bins,
    )
    return best_homography, best_lie_solution, at_dst_map, at_src_map
