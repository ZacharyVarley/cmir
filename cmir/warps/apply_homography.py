"""
Much of the logic here is slightly modified from Kornia:

https://github.com/kornia/kornia

I have removed some verbose type checking and wrappers around homogrpahy 
inversions in favor of making sure that everything can be jitted with
torchscript. I also needed to isolate the coordinate transformation as
I want to subsample which pixels are obtained in the warped image to 
speed up registration loops. The native warp_perspective in Kornia warps
the entire image.

"""

import torch
from torch import Tensor
from torch.nn import Module
from typing import Tuple


@torch.jit.script
def transform_points(trans_01: Tensor, points_1: Tensor) -> Tensor:
    r"""Function that applies transformations to a set of points.

    Args:
        trans_01: tensor for transformations of shape
          :math:`(B, D+1, D+1)`.
        points_1: tensor of points of shape :math:`(B, N, D)`.
    Returns:
        a tensor of N-dimensional points.

    Shape:
        - Output: :math:`(B, N, D)`

    """
    if not trans_01.shape[0] == points_1.shape[0] and trans_01.shape[0] != 1:
        raise ValueError(
            "Input batch size must be the same for both tensors or 1."
            f"Got {trans_01.shape} and {points_1.shape}"
        )
    if not trans_01.shape[-1] == (points_1.shape[-1] + 1):
        raise ValueError(
            "Last input dimensions must differ by one unit"
            f"Got{trans_01} and {points_1}"
        )

    # We reshape to BxNxD in case we get more dimensions, e.g., MxBxNxD
    shape_inp = list(points_1.shape)
    points_1 = points_1.reshape(-1, points_1.shape[-2], points_1.shape[-1])
    trans_01 = trans_01.reshape(-1, trans_01.shape[-2], trans_01.shape[-1])
    # We expand trans_01 to match the dimensions needed for bmm
    trans_01 = torch.repeat_interleave(
        trans_01, repeats=points_1.shape[0] // trans_01.shape[0], dim=0
    )
    # to homogeneous
    points_1_h = torch.nn.functional.pad(points_1, [0, 1], "constant", 1.0)  # BxNxD+1
    # transform coordinates
    points_0_h = torch.bmm(points_1_h, trans_01.permute(0, 2, 1))
    points_0_h = torch.squeeze(points_0_h, dim=-1)

    # we check for points at max_val
    z_vec: Tensor = points_0_h[..., -1:]
    mask: Tensor = torch.abs(z_vec) > 1e-8
    scale = torch.where(mask, 1.0 / (z_vec + 1e-8), torch.ones_like(z_vec))
    points_0 = scale * points_0_h[..., :-1]  # BxNxD
    # reshape to the input shape
    shape_inp[-2] = points_0.shape[-2]
    shape_inp[-1] = points_0.shape[-1]
    points_0 = points_0.reshape(shape_inp)
    return points_0


@torch.jit.script
def denormalize_homography(
    homographies: Tensor, start_HW: tuple[int, int], final_HW: tuple[int, int]
) -> Tensor:
    r"""
    Denormalize a given homography defined over [-1, 1] to be in terms of image sizes.

    Args:
        homographies: homography matrices to denormalize of shape :math:`(B, 3, 3)`.
        start_HW: source image size tuple :math:`(H, W)`.
        final_HW: destination image size tuple :math:`(H, W)`.

    """
    if not isinstance(homographies, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(homographies)}")

    if not (len(homographies.shape) == 3 or homographies.shape[-2:] == (3, 3)):
        raise ValueError(
            f"Input homographies must be a Bx3x3 tensor. Got {homographies.shape}"
        )

    # source and destination sizes
    src_h, src_w = start_HW
    dst_h, dst_w = final_HW

    # (B, 3, 3) source unto [0, 1]
    src_to_square = torch.tensor(
        [
            [2.0 / (src_w - 1), 0.0, -1.0],
            [0.0, 2.0 / (src_h - 1), -1.0],
            [0.0, 0.0, 1.0],
        ],
        device=homographies.device,
        dtype=homographies.dtype,
    )[None]

    # (B, 3, 3) destination unto [0, 1]
    dst_to_square = torch.tensor(
        [
            [2.0 / (dst_w - 1), 0.0, -1.0],
            [0.0, 2.0 / (dst_h - 1), -1.0],
            [0.0, 0.0, 1.0],
        ],
        device=homographies.device,
        dtype=homographies.dtype,
    )[None]

    square_to_dst = torch.inverse(dst_to_square)

    # compute the denormed homography from source to destination
    homographies_denormed = square_to_dst @ (homographies @ src_to_square)
    # make sure the bottom right element is 1.0
    homographies_denormed = homographies_denormed / homographies_denormed[:, 2:3, 2:3]
    return homographies_denormed


@torch.jit.script
def normalize_homography(
    homographies: Tensor, start_HW: tuple[int, int], final_HW: tuple[int, int]
) -> Tensor:
    r"""Normalize a given homography in pixels to operate on coordinates in range [-1, 1].
    This is convenient because grid_sample is defined in PyTorch over the [-1, 1] square.

    Args:
        homographies: homography matrices to normalize of shape :math:`(B, 3, 3)`.
        start_HW: source image size tuple :math:`(H, W)`.
        final_HW: destination image size tuple :math:`(H, W)`.

    Returns:
        Normalized homography of shape :math:`(B, 3, 3)` that can be directly applied
        to the normalized destination coordinates that are in range [-1, 1].

    """
    if not isinstance(homographies, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(homographies)}")

    if not (len(homographies.shape) == 3 or homographies.shape[-2:] == (3, 3)):
        raise ValueError(
            f"Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {homographies.shape}"
        )

    # source and destination sizes
    src_h, src_w = start_HW
    dst_h, dst_w = final_HW

    # (B, 3, 3) source unto [-1, 1]
    src_to_square = torch.tensor(
        [
            [2.0 / (src_w - 1), 0.0, -1.0],
            [0.0, 2.0 / (src_h - 1), -1.0],
            [0.0, 0.0, 1.0],
        ],
        device=homographies.device,
        dtype=homographies.dtype,
    )[None]

    # (B, 3, 3) destination unto [-1, 1]
    dst_to_square = torch.tensor(
        [
            [2.0 / (dst_w - 1), 0.0, -1.0],
            [0.0, 2.0 / (dst_h - 1), -1.0],
            [0.0, 0.0, 1.0],
        ],
        device=homographies.device,
        dtype=homographies.dtype,
    )[None]

    # compute the normed homography from destination to source (backwards) so that we can
    # easily obtain the placements of the destination pixels in the source image.
    square_to_src = torch.inverse(src_to_square)
    square_to_src_to_dst_to_square = dst_to_square @ (homographies @ square_to_src)
    square_to_dst_to_src_to_square = torch.inverse(square_to_src_to_dst_to_square)
    # make sure the bottom right element is 1.0
    square_to_dst_to_src_to_square = (
        square_to_dst_to_src_to_square / square_to_dst_to_src_to_square[:, 2:3, 2:3]
    )
    return square_to_dst_to_src_to_square


@torch.jit.script
def get_homography_grid(
    output_shape: Tuple[int, int],
    device: torch.device = torch.device("cpu"),
):
    """
    Get a grid of coordinates for a homography.

    Args:
        output_shape: Output shape of the grid (H, W).
        device: Device to put the grid on.

    Returns:
        The grid shape (H, W, 2).

    """
    height_out, width_out = output_shape

    xs: Tensor = torch.linspace(
        0, width_out - 1, width_out, device=device, dtype=torch.float32
    )
    ys: Tensor = torch.linspace(
        0, height_out - 1, height_out, device=device, dtype=torch.float32
    )

    xs = (xs / (width_out - 1) - 0.5) * 2
    ys = (ys / (height_out - 1) - 0.5) * 2

    grid = torch.stack(torch.meshgrid([xs, ys], indexing="ij"), dim=-1).permute(1, 0, 2)
    return grid


@torch.jit.script
def pad_homography(
    homographies: Tensor,
    padding_amounts: Tuple[int, int],
) -> Tensor:
    """

    Often we want to expand the canvas at the destination image to avoid clipping.

    Args:
        homographes: Homographies to pad shape (B, 3, 3), defined in pixel coordinates.
        output_shape: Output shape of the grid (H, W).
        padding_amounts: Amounts to pad the image (left, right, top, bottom).

    Returns:
        The modified homographies shape (B, 3, 3).

    """
    left, top = padding_amounts

    homographies_padded = homographies
    homographies_padded[:, 0, 2] += left
    homographies_padded[:, 1, 2] += top

    return homographies_padded


@torch.jit.script
def apply_homographies_to_images(
    images: Tensor,
    homographies: Tensor,
    output_shape: Tuple[int, int],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> Tensor:
    """
    Apply a batch of homographies to an image.

    Args:
        img: Image to apply the homographies to shape (B, C, H, W).
        homographies: Homographies to apply to img shape (B, 3, 3).
        output_shape: Output shape of the transformed images (H, W).
        mode: Interpolation mode to use.
        padding_mode: Padding mode to use.
        align_corners: Align corners when interpolating.

    Returns:
        The transformed images shape (B, C, H, W).

    """

    B, _, height_in, width_in = images.shape
    B2, _, _ = homographies.shape

    height_out, width_out = output_shape

    if B != B2 and B2 != 1 and B != 1:
        raise ValueError(
            f"Batch size of images {B} and homographies {B2} do not match and broadcast is not possible."
        )

    if B == 1:
        images_repeated = images.repeat(B2, 1, 1, 1)
    else:
        images_repeated = images

    xs: Tensor = torch.linspace(
        0, width_out - 1, width_out, device=images.device, dtype=images.dtype
    )
    ys: Tensor = torch.linspace(
        0, height_out - 1, height_out, device=images.device, dtype=images.dtype
    )

    xs = (xs / (width_out - 1) - 0.5) * 2
    ys = (ys / (height_out - 1) - 0.5) * 2

    grid = (
        torch.stack(torch.meshgrid([xs, ys], indexing="ij"), dim=-1)
        .permute(1, 0, 2)
        .repeat(B2, 1, 1, 1)
    )

    homographies_normed = normalize_homography(
        homographies, (height_in, width_in), (height_out, width_out)
    )  # Bx3x3

    # (B, 1, 1, 3, 3) @ (B, W, H, 3, 1) -> (B, H, W, 3, 1) -> (B, H, W, 3)
    grid_transformed = transform_points(
        homographies_normed[:, None, None], grid
    ).squeeze(-1)

    img_transformed = torch.nn.functional.grid_sample(
        images_repeated,
        grid_transformed,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    return img_transformed


@torch.jit.script
def normed_homographies_on_points(
    points: Tensor,
    homographies: Tensor,
) -> Tensor:
    """
    Apply to a grid of points, a batch of homographies defined between [-1, 1]
    squares of the source and destination images.


    Args:
        points: Points to apply the homographies to shape (B, N, 2) or (B, H, W, 2).
        homographies: Homographies to apply to points shape (B, 3, 3).

    Returns:
        The transformed points shape (B, N, 2) or (B, H, W, 2).

    """
    B, N, K2, _ = points.shape
    B2, _, _ = homographies.shape

    if B != B2 and B2 != 1 and B != 1:
        raise ValueError(
            f"Batch size of points {B} and homographies {B2} do not match and broadcast is not possible."
        )

    if B == 1:
        points_repeated = points.repeat(B2, 1, 1, 1)
    else:
        points_repeated = points

    points_transformed = transform_points(homographies, points_repeated)

    return points_transformed


@torch.jit.script
def unnormed_homographies_on_points(
    points: Tensor,
    homographies: Tensor,
    start_HW: Tuple[int, int],
    final_HW: Tuple[int, int],
) -> Tensor:
    """
    Apply a batch of homographies to a set of points.

    Args:
        points: Points to apply the homographies to shape (B, N, K2, 2).
        homographies: Homographies to apply to points shape (B, 3, 3).
        start_HW: Source image size tuple :math:`(H, W)`.
        final_HW: Destination image size tuple :math:`(H, W)`.

    Returns:
        The transformed points shape (B, N, 2).

    """
    B, N, K2, _ = points.shape
    B2, _, _ = homographies.shape

    if B != B2 and B2 != 1 and B != 1:
        raise ValueError(
            f"Batch size of points {B} and homographies {B2} do not match and broadcast is not possible."
        )

    if B == 1:
        points_repeated = points.repeat(B2, 1, 1, 1)
    else:
        points_repeated = points

    homographies_normed = normalize_homography(homographies, start_HW, final_HW)
    points_transformed = transform_points(homographies_normed, points_repeated)

    return points_transformed


@torch.jit.script
def apply_bspline_grid_to_images(
    control_grids: Tensor,
    base_homographies: Tensor,
    images: Tensor,
    output_shape: Tuple[int, int],
    mode: str = "bilinear",
):
    """
    Apply a batch of bspline control grids to an image.

    Args:
        control_grids: Control grids to apply to img shape (B, 2, H, W).
        base_homographies: Base homographies to apply before warping with the control grids
            at the source images sshape (B, 3, 3).
        img: Image to apply the control grids to shape (B, C, H, W).
        output_shape: Output shape of the transformed images (H, W).
        mode: Interpolation mode to use.

    Returns:
        The transformed images shape (B, C, H, W).

    """

    # starting identity grid defined on [-1, 1] shape (B, target_H, target_W, 2)
    coord_grid_at_target = get_homography_grid(output_shape, images.device).repeat(
        control_grids.shape[0], 1, 1, 1
    )

    # normalize the homographies
    homographies_normed = normalize_homography(
        torch.inverse(base_homographies),
        (images.shape[-2], images.shape[-1]),
        output_shape,
    )

    # we warp the points defined in the target canvas back to their source positions in the source canvas
    coord_grid_at_source = normed_homographies_on_points(
        coord_grid_at_target, torch.inverse(homographies_normed)
    )

    # use torch's grid_sample to warp the source image with bicubic interpolation
    displacements = torch.nn.functional.grid_sample(
        control_grids,
        coord_grid_at_source,
        mode="bicubic",
        padding_mode="border",
        align_corners=True,
    )

    # add the displacements to the original grid
    coord_grid_at_source = coord_grid_at_source + displacements.permute(0, 2, 3, 1)

    # compute the fitness of the candidates
    img_source_warped = torch.nn.functional.grid_sample(
        images,
        coord_grid_at_source,
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    )

    return img_source_warped
