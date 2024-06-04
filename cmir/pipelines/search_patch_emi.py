"""
This file implements image registration using: 

1) 3rd/4th order Edgeworth mutual information in small image patches

2) Adam optimization of homography parameters in their Lie algebra space

3) Halfway symmetric loss for the homography optimization

We are warping the source image halfway in homography space towards the
destination image and warp the destination image halfway in homography space
towards the source image. I have scoured the literature and I cannot find the
original source for this symmetric loss, but I remember that it was
fundamentally different from a vanilla symmetric loss. I will update this
comment when I find the original source.

"""

import torch
from typing import Optional
from torch import Tensor

from cmir.metrics.patch_emi import patch_emi_3
from cmir.warps.lie_homography_exp import LieHomographies
from torch.optim import Adam
from cmir.warps.apply_homography import (
    apply_homographies_to_images,
    denormalize_homography,
)
from cmir.optimizers.progessbar import ProgressBar


def enmi_diffable_half(
    img_dst: torch.Tensor,
    img_src: torch.Tensor,
    dst_ind: Optional[torch.Tensor],
    src_ind: Optional[torch.Tensor],
    pad_frac: float,
    n_iterations: int,
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
        img_dst: The destination image.
        img_src: The source image.
        dst_ind: The indicator for the destination image.
        src_ind: The indicator for the source image.
        pad_frac: The fraction of padding to add to the images.
        n_iterations: The number of iterations to run the optimizer.
        guess_lie: The initial guess for the lie algebra vector.
        x_translation_weight: The weight for the x translation.
        y_translation_weight: The weight for the y translation.
        rotation_weight: The weight for the rotation.
        scale_weight: The weight for the scale.
        stretch_weight: The weight for the stretch.
        shear_weight: The weight for the shear.
        x_keystone_weight: The weight for the x keystone.
        y_keystone_weight: The weight for the y keystone.
        verbose: Whether to print the progress of the optimizer.

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

    # put the guess Lie vector into the lie homography
    lie_homog.lie_vectors.data = guess_lie[None, :, None, None]

    # class for homography
    lie_homog = LieHomographies(
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

    # find the canvas shape
    dst_canvas_shape = (
        img_dst.shape[-2] + 2 * pad_h_dst,
        img_dst.shape[-1] + 2 * pad_w_dst,
    )
    src_canvas_shape = (
        img_src.shape[-2] + 2 * pad_h_src,
        img_src.shape[-1] + 2 * pad_w_src,
    )

    # Adam optimizer
    cmaes_wrap = Adam(
        lie_homog.parameters(),
        lr=0.01,
    )

    pb = ProgressBar(range(n_iterations), prefix="Adam: ", size=60)

    for _ in pb:
        # compute the lie homography
        half_forword_H, half_backwrd_H = lie_homog.half_forward_backward()

        # make homographies defined on [-1, 1] x [-1, 1] use pixel coordinates
        src_H = denormalize_homography(
            half_forword_H,
            img_src.shape[-2:],
            img_dst.shape[-2:],
        )

        dst_H = denormalize_homography(
            half_backwrd_H,
            img_dst.shape[-2:],
            img_src.shape[-2:],
        )

        # compute the fitness of the candidates
        img_src_warp = apply_homographies_to_images(img_src, src_H, img_dst.shape[-2:])

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
