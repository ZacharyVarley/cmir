"""
This is a PyTorch implementation of 2D/3D mclahe for (multidimensional 
contrast limited adaptive histogram equalization) from:

V. Stimper, S. Bauer, R. Ernstorfer, B. Schölkopf and R. P. Xian, 
"Multidimensional Contrast Limited Adaptive Histogram Equalization," 
in IEEE Access, vol. 7, pp. 165437-165447, 2019.

https://github.com/VincentStimper/mclahe

That codebase is also MIT licensed.

--------------------------------

This implementation uses higher order splines (I think first time in a CLAHE 
implementation) to fit the CDF's themselves, and to interpolate between them. 
Splines are from Yael Balbastre's PyTorch package:

https://github.com/balbasty/torch-interpol/

Also MIT licensed.

--------------------------------

This implementation is also heavily inspired by Kornia's CLAHE implementation:

https://github.com/kornia/kornia

Which is Apache 2.0 licensed so I have to highlight which parts are from there.

I originally tried to implement the clip limit in different ways but reverted 
the changes and exactly mirror the discretization of the PDFs.

"""

import math
from typing import Tuple
import torch

from cmir.warps.splines.pushpull import grid_pull as grid_pull_backend
from cmir.warps.splines.autograd import GridPull, bound_to_nitorch, make_list

"""
For now all CDFs are in units corresponding to [0, 255] uint8 
values despite being floats. I want to keep the default clip 
limits that everyone uses, but I am fitting a 2+ order spline 
to the CDFs so I don't floor the values to match the uint8 or 
it would not make much sense to fit a spline to their values.

grid_pull_api accepts string arguments for interpolation and 
extrapolation, while grid_pull_backend takes integer arguments. 

Yael's Conventions for Spline backend inputs:

ExtrapolateTypes:
    0: no    # hard threshold: (0, n-1)
    1: yes 
    2: hist  # hard threshold: (-0.5, n-0.5)

BoundTypes:
    0 : 'zero', 'zeros', 'constant'
    1 : 'replicate', 'repeat', 'border', 'nearest'
    2: 'dct1', 'mirror'
    3: 'dct2', 'reflect', 'reflection', 'neumann'
    4: 'dst1', 'antimirror'
    5: 'dst2', 'antireflect', 'dirichlet'
    6: 'dft', 'wrap', 'circular'

"""


@torch.jit.script
def clahe_2d_undiffable(
    x: torch.Tensor,
    clip_limit: float,
    n_bins: int,
    grid_shape: Tuple[int, int],
    spline_order: int = 5,
    spline_extrapolate: int = 1,
    spline_bound: int = 3,
) -> torch.Tensor:
    """
    Calculates the tile CDFs for CLAHE on a 2D tensor.

    Args:
        x (torch.Tensor): Input image tensor of shape (B, C, H, W).
        clip_limit (float): Clipping limit for the histogram.
        n_bins (int): Number of bins for the histogram.
        grid_shape (Tuple[int, int]): Shape of the grid for dividing the image into tiles.
        units (str, optional): Units for the clipping limit. Can be 'pixels' or

    Returns:
        torch.Tensor: Tensor of shape (B, C, GH, GW, n_bins) with the CDFs for each tile.
        torch.Tensor: Tensor of shape (B, C,


    """
    # get shapes
    B, C, H, W = x.shape
    grid_height, grid_width = grid_shape
    n_tiles = grid_height * grid_width
    tile_shape = [int(math.ceil(H / grid_height)), int(math.ceil(W / grid_width))]
    voxels_per_tile = tile_shape[0] * tile_shape[1]

    # pad the input to be divisible by the tile counts in each dimension
    paddings = [0, 0, 0, 0]
    if H % grid_height != 0:
        paddings[0] = grid_height - (H % grid_height)
    if W % grid_width != 0:
        paddings[2] = grid_width - (W % grid_width)

    padding_h = paddings[0]
    padding_w = paddings[2]

    # torch.nn.functional.pad uses last dimension to first in pairs
    with torch.no_grad():
        x_padded = torch.nn.functional.pad(x, paddings[::-1], mode="reflect")

    # meshgrid of all of the coordinates for interpolation
    # the tile centers are at the corners of the interpolated image
    # so extrapolation outside of the unit square is needed of
    # the grid_pull function. How far out also depends on the tile counts
    coords_h = torch.linspace(
        -0.5 - (0.25 / grid_height),
        (grid_height - 1) + 0.5 + (0.25 / grid_height),
        x_padded.shape[-2],
        device=x.device,
    )
    coords_w = torch.linspace(
        -0.5 - (0.25 / grid_width),
        (grid_width - 1) + 0.5 + (0.25 / grid_width),
        x_padded.shape[-1],
        device=x.device,
    )
    coords = torch.meshgrid([coords_h, coords_w], indexing="ij")
    coords = torch.stack(coords, dim=-1)

    # unfold the input into tiles of shape (B, C*voxels_per_tile, -1)
    with torch.no_grad():
        tiles = torch.nn.functional.unfold(
            x_padded, kernel_size=tile_shape, stride=tile_shape
        )

    # reshape from (B, C*voxels, n_tiles) to (B, C, voxels_per_tile, n_tiles)
    tiles = tiles.reshape((B, C, voxels_per_tile, n_tiles))

    # permute from (B, C, voxels_per_tile, n_tiles) to (B, C, n_tiles, voxels_per_tile)
    tiles = tiles.movedim(-1, -2)

    # here we pre-allocate the pdf tensor to avoid having all residuals in memory at once
    pdf = torch.zeros((B, C, n_tiles, n_bins), device=x.device)

    # don't need grad so use torch.histc instead of soft histogram
    for b in range(B):
        for c in range(C):
            for t in range(n_tiles):
                pdf[b, c, t] = torch.histc(
                    input=tiles[b, c, t], bins=n_bins, min=0.0, max=1.0
                ).to(x.dtype)
                pdf[b, c, t] = pdf[b, c, t] / (torch.sum(pdf[b, c, t]) + 1e-10)

    # ------------ Begin from kornia's implementation ------------

    # pdf is handled in "pixels" viewed as a large batch of 1D pdfs
    histos = (pdf * voxels_per_tile).view(-1, n_bins)
    if clip_limit > 0:
        # calc limit
        limit = max(clip_limit * voxels_per_tile // n_bins, 1)
        histos.clamp_(max=limit)
        # calculate the clipped pdf of shape (B, C, n_tiles, n_bins)
        clipped = voxels_per_tile - histos.sum(-1)

        # calculate the excess of shape (B, C, n_tiles, n_bins)
        residual = torch.remainder(clipped, n_bins)
        redist = (clipped - residual).div(n_bins)
        histos += redist[..., None]
        # trick to avoid using a loop to assign the residual
        v_range: torch.Tensor = torch.arange(n_bins, device=histos.device)
        mat_range: torch.Tensor = v_range.repeat(histos.shape[0], 1)
        histos += mat_range < residual[None].transpose(0, 1)
    # cdf (B, C, n_tiles, n_bins)
    cdfs = torch.cumsum(histos, dim=-1) * (n_bins - 1) / voxels_per_tile
    cdfs = cdfs.clamp(min=0.0, max=(n_bins - 1))

    # --------------- End from Kornia's implementation section ----------------

    # (B*C*n_tiles, n_bins) -> (B, C, GH, GW, n_bins) -> (B, C, n_bins, GH, GW)
    # if you use transpose, the interpolation will be wrong
    cdfs = cdfs.reshape(
        B,
        C,
        grid_height,
        grid_width,
        n_bins,
    ).movedim(-1, -3)

    # meshgrid of all of the coordinates for interpolation
    # the tile centers are at the corners of the interpolated image
    # so extrapolation outside of the unit square is needed of
    # the grid_pull function. How far out also depends on the tile counts
    coords_h = torch.linspace(
        -0.5 - (0.25 / grid_height),
        (grid_height - 1) + 0.5 + (0.25 / grid_height),
        H + padding_h,
        device=x.device,
    )[:H]
    coords_w = torch.linspace(
        -0.5 - (0.25 / grid_width),
        (grid_width - 1) + 0.5 + (0.25 / grid_width),
        W + padding_w,
        device=x.device,
    )[:W]
    coords = torch.meshgrid([coords_h, coords_w], indexing="ij")
    coords = torch.stack(coords, dim=-1)

    # iterate over combined batch and channel dimension and interpolate
    for b in range(B):
        for c in range(C):
            # (H, W, 1) with (H, W, 2) -> (H, W, 3) to get the interpolation coordinates
            coords_with_channel = torch.cat(
                [x[b, c][..., None] * (n_bins - 1.0), coords], dim=-1
            )
            x[b, c] = grid_pull_backend(
                cdfs[b, c][None, None],  # (1, 1, n_bins, GH, GW)
                coords_with_channel[None],  # (1, H, W, 3)
                interpolation=[
                    spline_order,
                ],  # 2nd+ order makes a big difference
                extrapolate=spline_extrapolate,
                bound=[
                    spline_bound,
                ],
            )[0, 0, 0]

    # min max normalization
    x = (x - x.min()) / (x.max() - x.min() + 1e-10)
    return x


@torch.jit.script
def clahe_2d_diffable(
    x: torch.Tensor,
    clip_limit: float,
    n_bins: int,
    grid_shape: Tuple[int, int],
    spline_order: int = 5,
    spline_extrapolate: int = 1,
    spline_bound: str = "reflect",
    bandwidth: float = 0.001,
) -> torch.Tensor:
    """
    Calculates the tile CDFs for CLAHE on a 2D tensor. Differentiable version.
    Soft histogram via Parzen/KDE windowing with a Gaussian kernel.

    Args:
        x (torch.Tensor): Input image tensor of shape (B, C, H, W).
        clip_limit (float): Clipping limit for the histogram.
        n_bins (int): Number of bins for the histogram.
        grid_shape (Tuple[int, int]): Shape of the grid for dividing the image into tiles.
        units (str, optional): Units for the clipping limit. Can be 'pixels' or

    Returns:
        torch.Tensor: Tensor of shape (B, C, GH, GW, n_bins) with the CDFs for each tile.
        torch.Tensor: Tensor of shape (B, C,


    """
    # get shapes
    B, C, H, W = x.shape
    grid_height, grid_width = grid_shape
    n_tiles = grid_height * grid_width
    tile_shape = [int(math.ceil(H / grid_height)), int(math.ceil(W / grid_width))]
    voxels_per_tile = tile_shape[0] * tile_shape[1]

    # pad the input to be divisible by the tile counts in each dimension
    paddings = [0, 0, 0, 0]
    if H % grid_height != 0:
        paddings[0] = grid_height - (H % grid_height)
    if W % grid_width != 0:
        paddings[2] = grid_width - (W % grid_width)

    padding_h = paddings[0]
    padding_w = paddings[2]

    # torch.nn.functional.pad uses last dimension to first in pairs
    with torch.no_grad():
        x_padded = torch.nn.functional.pad(x, paddings[::-1], mode="reflect")

    # meshgrid of all of the coordinates for interpolation
    # the tile centers are at the corners of the interpolated image
    # so extrapolation outside of the unit square is needed of
    # the grid_pull function. How far out also depends on the tile counts
    coords_h = torch.linspace(
        -0.5 - (0.25 / grid_height),
        (grid_height - 1) + 0.5 + (0.25 / grid_height),
        x_padded.shape[-2],
        device=x.device,
    )
    coords_w = torch.linspace(
        -0.5 - (0.25 / grid_width),
        (grid_width - 1) + 0.5 + (0.25 / grid_width),
        x_padded.shape[-1],
        device=x.device,
    )
    coords = torch.meshgrid([coords_h, coords_w], indexing="ij")
    coords = torch.stack(coords, dim=-1)

    # unfold the input into tiles of shape (B, C*voxels_per_tile, -1)
    with torch.no_grad():
        tiles = torch.nn.functional.unfold(
            x_padded, kernel_size=tile_shape, stride=tile_shape
        )

    # reshape from (B, C*voxels, n_tiles) to (B, C, voxels_per_tile, n_tiles)
    tiles = tiles.reshape((B, C, voxels_per_tile, n_tiles))

    # permute from (B, C, voxels_per_tile, n_tiles) to (B, C, n_tiles, voxels_per_tile)
    tiles = tiles.movedim(-1, -2)

    # here we pre-allocate the pdf tensor to avoid having all residuals in memory at once
    pdf = torch.zeros((B, C, n_tiles, n_bins), device=x.device)

    # don't need grad so use torch.histc instead of soft histogram
    for b in range(B):
        for c in range(C):
            for t in range(n_tiles):
                # calculate the pdf which is shape (B, C, n_tiles, n_bins)
                residuals_batch = (
                    tiles[b, c, t, :, None]
                    - torch.linspace(0.0, 1.0, n_bins, device=x.device)[None, :]
                )
                weights_batch = torch.exp(-0.5 * (residuals_batch / bandwidth).pow(2))
                pdf[b, c, t] = torch.mean(weights_batch, dim=-2)
                pdf[b, c, t] = pdf[b, c, t] / (
                    torch.sum(pdf[b, c, t], dim=-1, keepdim=True) + 1e-10
                )

    # ------------ Begin from kornia's implementation ------------

    # pdf is handled in "pixels" viewed as a large batch of 1D pdfs
    histos = (pdf * voxels_per_tile).view(-1, n_bins)

    if clip_limit > 0:
        # calc limit
        limit = max(clip_limit * voxels_per_tile // n_bins, 1)

        histos.clamp_(max=limit)

        # calculate the clipped pdf of shape (B, C, n_tiles, n_bins)
        clipped = voxels_per_tile - histos.sum(-1)

        # calculate the excess of shape (B, C, n_tiles, n_bins)
        residual = torch.remainder(clipped, n_bins)
        redist = (clipped - residual).div(n_bins)
        histos += redist[..., None]

        # trick to avoid using a loop to assign the residual
        v_range: torch.Tensor = torch.arange(n_bins, device=histos.device)
        mat_range: torch.Tensor = v_range.repeat(histos.shape[0], 1)
        histos += mat_range < residual[None].transpose(0, 1)

    # cdf (B, C, n_tiles, n_bins)
    cdfs = torch.cumsum(histos, dim=-1) * (n_bins - 1) / voxels_per_tile

    cdfs = cdfs.clamp(min=0.0, max=(n_bins - 1))

    # --------------- End from Kornia's implementation section ----------------

    # (B*C*n_tiles, n_bins) -> (B, C, GH, GW, n_bins) -> (B, C, n_bins, GH, GW)
    # if you use transpose, the interpolation will be wrong
    cdfs = cdfs.reshape(
        B,
        C,
        grid_height,
        grid_width,
        n_bins,
    ).movedim(-1, -3)

    # meshgrid of all of the coordinates for interpolation
    # the tile centers are at the corners of the interpolated image
    # so extrapolation outside of the unit square is needed of
    # the grid_pull function. How far out also depends on the tile counts
    coords_h = torch.linspace(
        -0.5 - (0.25 / grid_height),
        (grid_height - 1) + 0.5 + (0.25 / grid_height),
        H + padding_h,
        device=x.device,
    )[:H]
    coords_w = torch.linspace(
        -0.5 - (0.25 / grid_width),
        (grid_width - 1) + 0.5 + (0.25 / grid_width),
        W + padding_w,
        device=x.device,
    )[:W]
    coords = torch.meshgrid([coords_h, coords_w], indexing="ij")
    coords = torch.stack(coords, dim=-1)

    # iterate over combined batch and channel dimension and interpolate
    for b in range(B):
        for c in range(C):
            # (H, W, 1) with (H, W, 2) -> (H, W, 3) to get the interpolation coordinates
            coords_with_channel = torch.cat(
                [x[b, c][..., None] * (n_bins - 1.0), coords], dim=-1
            )
            x[b, c] = GridPull.apply(
                cdfs[b, c][None, None],  # (1, 1, n_bins, GH, GW)
                coords_with_channel[None],  # (1, H, W, 3)
                spline_order,
                spline_bound,
                spline_extrapolate,
            )[0, 0, 0]

    # min max normalization
    x = (x - x.min()) / (x.max() - x.min() + 1e-10)
    return x


class Clahe2D(torch.nn.Module):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to a 2D tensor.

    Args:
        clip_limit (float): Clipping limit for the histogram.
        n_bins (int): Number of bins for the histogram.
        grid_shape (Tuple[int, int]): Shape of the grid for dividing the image into tiles.
        spline_order (int, optional): Order of the interpolation used to compute the output tensor.
            Defaults to 5. This makes a big difference in CLAHE quality.
        spline_extrapolate (bool, optional): Whether to extrapolate outside of the unit square.
            Defaults to True.
        spline_bound (str, optional): Boundary condition for extrapolation.
            Can be 'reflect', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'. Defaults to 'reflect'.
        diffable (bool, optional): Whether to use the differentiable binning and explicit Autograd spline
    Returns:
        torch.Tensor: Output tensor of shape (B, C, H, W) with CLAHE applied.

    """

    def __init__(
        self,
        clip_limit: float,
        n_bins: int,
        grid_shape: Tuple[int, int],
        spline_order: int = 5,
        spline_bound: str = "reflect",
        diffable: bool = False,
        spline_extrapolate: bool = True,
    ):
        super().__init__()
        self.clip_limit = clip_limit
        self.n_bins = n_bins
        self.grid_shape = grid_shape
        self.spline_order = spline_order
        self.spline_extrapolate = spline_extrapolate
        self.spline_bound = spline_bound
        self.diffable = diffable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # sanitize input
        if x.ndim != 4:
            raise ValueError("Input tensor must be 4D")
        if (self.grid_shape[0] > x.shape[-2]) or (self.grid_shape[1] > x.shape[-1]):
            raise ValueError(
                "Grid shape must be smaller than the input tensor in both dimensions"
            )
        if self.diffable:
            # the autograd function takes string arguments for interpolation and extrapolation
            return clahe_2d_diffable(
                x,
                self.clip_limit,
                self.n_bins,
                self.grid_shape,
                self.spline_order,
                self.spline_extrapolate,
                self.spline_bound,
            )
        else:
            # the backend function takes integer arguments for interpolation and extrapolation
            bound = bound_to_nitorch(make_list(self.spline_bound), as_type="int")[0]
            extrapolate = int(self.spline_extrapolate)
            return clahe_2d_undiffable(
                x,
                clip_limit=self.clip_limit,
                n_bins=self.n_bins,
                grid_shape=self.grid_shape,
                spline_order=self.spline_order,
                spline_extrapolate=extrapolate,
                spline_bound=bound,
            )


@torch.jit.script
def clahe_3d_undiffable(
    x: torch.Tensor,
    clip_limit: float,
    n_bins: int,
    grid_shape: Tuple[int, int, int],
    spline_order: int = 5,
    spline_extrapolate: int = 1,
    spline_bound: int = 3,
) -> torch.Tensor:
    # get shapes
    B, C, D, H, W = x.shape
    grid_depth, grid_height, grid_width = grid_shape
    n_tiles = grid_depth * grid_height * grid_width
    tile_shape = [
        int(math.ceil(D / grid_depth)),
        int(math.ceil(H / grid_height)),
        int(math.ceil(W / grid_width)),
    ]
    voxels_per_tile = tile_shape[0] * tile_shape[1] * tile_shape[2]

    # pad the input to be divisible by the tile counts in each dimension
    paddings = [0, 0, 0, 0, 0, 0]
    if D % grid_depth != 0:
        paddings[0] = grid_depth - (D % grid_depth)
    if H % grid_height != 0:
        paddings[2] = grid_height - (H % grid_height)
    if W % grid_width != 0:
        paddings[4] = grid_width - (W % grid_width)

    padding_d = paddings[0]
    padding_h = paddings[2]
    padding_w = paddings[4]

    # PyTorch functional.pad uses last dimension to first in pairs
    with torch.no_grad():
        x_padded = torch.nn.functional.pad(x, paddings[::-1], mode="reflect")

    # meshgrid of all of the coordinates for interpolation
    # the tile centers are at the corners of the interpolated image
    # so extrapolation outside of the unit square is needed of
    # the grid_pull function. How far out also depends on the tile counts
    coords_d = torch.linspace(
        -0.5 - (0.25 / grid_depth),
        (grid_depth - 1) + 0.5 + (0.25 / grid_depth),
        x_padded.shape[-3],
        device=x.device,
    )
    coords_h = torch.linspace(
        -0.5 - (0.25 / grid_height),
        (grid_height - 1) + 0.5 + (0.25 / grid_height),
        x_padded.shape[-2],
        device=x.device,
    )
    coords_w = torch.linspace(
        -0.5 - (0.25 / grid_width),
        (grid_width - 1) + 0.5 + (0.25 / grid_width),
        x_padded.shape[-1],
        device=x.device,
    )
    coords = torch.meshgrid([coords_d, coords_h, coords_w], indexing="ij")
    coords = torch.stack(coords, dim=-1)

    # unfolding in 3D case requires successive calls with .unfold() instead
    tiles = x_padded.unfold(-3, tile_shape[0], tile_shape[0])
    tiles = tiles.unfold(-3, tile_shape[1], tile_shape[1])
    tiles = tiles.unfold(-3, tile_shape[2], tile_shape[2])

    # reshape from (B, C,) + (n_tiles_each_dim,)*3 + (voxels_along_each_dims)*3 to
    # be shape (B, C, n_tiles, voxels_per_tile)
    tiles = tiles.reshape((B, C, n_tiles, voxels_per_tile))

    # here we pre-allocate the pdf tensor to avoid having all residuals in memory at once
    pdf = torch.zeros((B, C, n_tiles, n_bins), device=x.device)

    # don't need grad so use torch.histc instead of soft histogram
    for b in range(B):
        for c in range(C):
            for t in range(n_tiles):
                pdf[b, c, t] = torch.histc(
                    input=tiles[b, c, t], bins=n_bins, min=0.0, max=1.0
                ).to(x.dtype)
                pdf[b, c, t] = pdf[b, c, t] / (torch.sum(pdf[b, c, t]) + 1e-10)

    # ------------ Taken directly from Kornia's implementation ------------

    # pdf is handled in "pixels" viewed as a large batch of 1D pdfs
    histos = (pdf * voxels_per_tile).view(-1, n_bins)

    if clip_limit > 0:
        # calc limit
        limit = max(clip_limit * voxels_per_tile // n_bins, 1)

        histos.clamp_(max=limit)

        # calculate the clipped pdf of shape (B, C, n_tiles, n_bins)
        clipped = voxels_per_tile - histos.sum(-1)

        # calculate the excess of shape (B, C, n_tiles, n_bins)
        residual = torch.remainder(clipped, n_bins)
        redist = (clipped - residual).div(n_bins)
        histos += redist[..., None]

        # trick to avoid using a loop to assign the residual
        v_range: torch.Tensor = torch.arange(n_bins, device=histos.device)
        mat_range: torch.Tensor = v_range.repeat(histos.shape[0], 1)
        histos += mat_range < residual[None].transpose(0, 1)

    # cdf (B, C, n_tiles, n_bins)
    cdf = torch.cumsum(histos, dim=-1) * (n_bins - 1) / voxels_per_tile

    # --------------- End of Kornia's implementation section ----------------

    # (B*C*n_tiles, n_bins) -> (B, C, GD, GH, GW, n_bins) -> (B, C, n_bins, GD, GH, GW)
    # if you use transpose, the interpolation will be wrong
    cdf = cdf.reshape(
        B,
        C,
        grid_depth,
        grid_height,
        grid_width,
        n_bins,
    ).movedim(-1, -4)

    # meshgrid of all of the coordinates for interpolation
    # the tile centers are at the corners of the interpolated image
    # so extrapolation outside of the unit square is needed of
    # the grid_pull function. How far out also depends on the tile counts
    coords_d = torch.linspace(
        -0.5 - (0.25 / grid_depth),
        (grid_depth - 1) + 0.5 + (0.25 / grid_depth),
        D + padding_d,
        device=x.device,
    )[:D]
    coords_h = torch.linspace(
        -0.5 - (0.25 / grid_height),
        (grid_height - 1) + 0.5 + (0.25 / grid_height),
        H + padding_h,
        device=x.device,
    )[:H]
    coords_w = torch.linspace(
        -0.5 - (0.25 / grid_width),
        (grid_width - 1) + 0.5 + (0.25 / grid_width),
        W + padding_w,
        device=x.device,
    )[:W]
    coords = torch.meshgrid([coords_d, coords_h, coords_w], indexing="ij")
    coords = torch.stack(coords, dim=-1)

    # iterate over combined batch and channel dimension and interpolate
    for b in range(B):
        for c in range(C):
            # (D, H, W, 1) with (D, H, W, 3) -> (D, H, W, 4) to get the interpolation coordinates
            coords_with_channel = torch.cat(
                [x[b, c][..., None] * (n_bins - 1.0), coords], dim=-1
            )
            x[b, c] = grid_pull_backend(
                cdf[b, c][None, None],  # (1, 1, n_bins, GD, GH, GW)
                coords_with_channel[None],  # (1, D, H, W, 4)
                interpolation=[
                    spline_order,
                ],  # 2nd+ order makes a big difference
                extrapolate=spline_extrapolate,
                bound=[
                    spline_bound,
                ],
            )[0, 0, 0]

    # min max normalization
    x = (x - x.min()) / (x.max() - x.min() + 1e-10)
    return x


@torch.jit.script
def clahe_3d_diffable(
    x: torch.Tensor,
    clip_limit: float,
    n_bins: int,
    grid_shape: Tuple[int, int, int],
    spline_order: int = 5,
    spline_extrapolate: int = 1,
    spline_bound: int = 3,
    bandwidth: float = 0.001,
) -> torch.Tensor:
    # get shapes
    B, C, D, H, W = x.shape
    grid_depth, grid_height, grid_width = grid_shape
    n_tiles = grid_depth * grid_height * grid_width
    tile_shape = [
        int(math.ceil(D / grid_depth)),
        int(math.ceil(H / grid_height)),
        int(math.ceil(W / grid_width)),
    ]
    voxels_per_tile = tile_shape[0] * tile_shape[1] * tile_shape[2]

    # pad the input to be divisible by the tile counts in each dimension
    paddings = [0, 0, 0, 0, 0, 0]
    if D % grid_depth != 0:
        paddings[0] = grid_depth - (D % grid_depth)
    if H % grid_height != 0:
        paddings[2] = grid_height - (H % grid_height)
    if W % grid_width != 0:
        paddings[4] = grid_width - (W % grid_width)

    padding_d = paddings[0]
    padding_h = paddings[2]
    padding_w = paddings[4]

    # PyTorch functional.pad uses last dimension to first in pairs
    with torch.no_grad():
        x_padded = torch.nn.functional.pad(x, paddings[::-1], mode="reflect")

    # meshgrid of all of the coordinates for interpolation
    # the tile centers are at the corners of the interpolated image
    # so extrapolation outside of the unit square is needed of
    # the grid_pull function. How far out also depends on the tile counts
    coords_d = torch.linspace(
        -0.5 - (0.25 / grid_depth),
        (grid_depth - 1) + 0.5 + (0.25 / grid_depth),
        x_padded.shape[-3],
        device=x.device,
    )
    coords_h = torch.linspace(
        -0.5 - (0.25 / grid_height),
        (grid_height - 1) + 0.5 + (0.25 / grid_height),
        x_padded.shape[-2],
        device=x.device,
    )
    coords_w = torch.linspace(
        -0.5 - (0.25 / grid_width),
        (grid_width - 1) + 0.5 + (0.25 / grid_width),
        x_padded.shape[-1],
        device=x.device,
    )
    coords = torch.meshgrid([coords_d, coords_h, coords_w], indexing="ij")
    coords = torch.stack(coords, dim=-1)

    # unfolding in 3D case requires successive calls with .unfold() instead
    tiles = x_padded.unfold(-3, tile_shape[0], tile_shape[0])
    tiles = tiles.unfold(-3, tile_shape[1], tile_shape[1])
    tiles = tiles.unfold(-3, tile_shape[2], tile_shape[2])

    # reshape from (B, C,) + (n_tiles_each_dim,)*3 + (voxels_along_each_dims)*3 to
    # be shape (B, C, n_tiles, voxels_per_tile)
    tiles = tiles.reshape((B, C, n_tiles, voxels_per_tile))

    # here we pre-allocate the pdf tensor to avoid having all residuals in memory at once
    pdf = torch.zeros((B, C, n_tiles, n_bins), device=x.device)

    # set up bin tensor
    bins = torch.linspace(0.0, 1.0, n_bins, device=x.device)

    # do need grad so use soft histogram
    for b in range(B):
        for c in range(C):
            for t in range(n_tiles):
                # calculate the pdf which is shape (B, C, n_tiles, n_bins)
                residuals_batch = tiles[b, c, t, :, None] - bins[None, :]
                weights_batch = torch.exp(-0.5 * (residuals_batch / bandwidth).pow(2))
                pdf[b, c, t] = torch.mean(weights_batch, dim=-2)
                pdf[b, c, t] = pdf[b, c, t] / (
                    torch.sum(pdf[b, c, t], dim=-1, keepdim=True) + 1e-10
                )

    # ------------ Taken directly from Kornia's implementation ------------

    # pdf is handled in "pixels" viewed as a large batch of 1D pdfs
    histos = (pdf * voxels_per_tile).view(-1, n_bins)

    if clip_limit > 0:
        # calc limit
        limit = max(clip_limit * voxels_per_tile // n_bins, 1)

        histos.clamp_(max=limit)

        # calculate the clipped pdf of shape (B, C, n_tiles, n_bins)
        clipped = voxels_per_tile - histos.sum(-1)

        # calculate the excess of shape (B, C, n_tiles, n_bins)
        residual = torch.remainder(clipped, n_bins)
        redist = (clipped - residual).div(n_bins)
        histos += redist[..., None]

        # trick to avoid using a loop to assign the residual
        v_range: torch.Tensor = torch.arange(n_bins, device=histos.device)
        mat_range: torch.Tensor = v_range.repeat(histos.shape[0], 1)
        histos += mat_range < residual[None].transpose(0, 1)

    # cdf (B, C, n_tiles, n_bins)
    cdf = torch.cumsum(histos, dim=-1) * (n_bins - 1) / voxels_per_tile

    # --------------- End of Kornia's implementation section ----------------

    # (B*C*n_tiles, n_bins) -> (B, C, GD, GH, GW, n_bins) -> (B, C, n_bins, GD, GH, GW)
    # if you use transpose, the interpolation will be wrong
    cdf = cdf.reshape(
        B,
        C,
        grid_depth,
        grid_height,
        grid_width,
        n_bins,
    ).movedim(-1, -4)

    # meshgrid of all of the coordinates for interpolation
    # the tile centers are at the corners of the interpolated image
    # so extrapolation outside of the unit square is needed of
    # the grid_pull function. How far out also depends on the tile counts
    coords_d = torch.linspace(
        -0.5 - (0.25 / grid_depth),
        (grid_depth - 1) + 0.5 + (0.25 / grid_depth),
        D + padding_d,
        device=x.device,
    )[:D]
    coords_h = torch.linspace(
        -0.5 - (0.25 / grid_height),
        (grid_height - 1) + 0.5 + (0.25 / grid_height),
        H + padding_h,
        device=x.device,
    )[:H]
    coords_w = torch.linspace(
        -0.5 - (0.25 / grid_width),
        (grid_width - 1) + 0.5 + (0.25 / grid_width),
        W + padding_w,
        device=x.device,
    )[:W]
    coords = torch.meshgrid([coords_d, coords_h, coords_w], indexing="ij")
    coords = torch.stack(coords, dim=-1)

    # iterate over combined batch and channel dimension and interpolate
    # this time we need to use the autograd spline interpolation
    for b in range(B):
        for c in range(C):
            # (D, H, W, 1) with (D, H, W, 3) -> (D, H, W, 4) to get the interpolation coordinates
            coords_with_channel = torch.cat(
                [x[b, c][..., None] * (n_bins - 1.0), coords], dim=-1
            )
            x[b, c] = GridPull.apply(
                cdf[b, c][None, None],  # (1, 1, n_bins, GD, GH, GW)
                coords_with_channel[None],  # (1, D, H, W, 4)
                spline_order,
                spline_bound,
                spline_extrapolate,
            )[0, 0, 0]

    # min max normalization
    x = (x - x.min()) / (x.max() - x.min() + 1e-10)
    return x


class Clahe3D(torch.nn.Module):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to a 3D tensor.

    Args:
        clip_limit (float): Clipping limit for the histogram.
        n_bins (int): Number of bins for the histogram.
        grid_shape (Tuple[int, int]): Shape of the grid for dividing the image into tiles.
        spline_order (int, optional): Order of the interpolation used to compute the output tensor.
            Defaults to 5. This makes a big difference in CLAHE quality.
        spline_extrapolate (bool, optional): Whether to extrapolate outside of the unit square.
            Defaults to True.
        spline_bound (str, optional): Boundary condition for extrapolation.
            Can be 'reflect', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'. Defaults to 'reflect'.
        diffable (bool, optional): Whether to use the differentiable binning and explicit Autograd spline
    Returns:
        torch.Tensor: Output tensor of shape (B, C, D, H, W) with CLAHE applied.

    """

    def __init__(
        self,
        clip_limit: float,
        n_bins: int,
        grid_shape: Tuple[int, int, int],
        spline_order: int = 5,
        spline_bound: str = "reflect",
        diffable: bool = False,
        spline_extrapolate: bool = True,
    ):
        super().__init__()
        self.clip_limit = clip_limit
        self.n_bins = n_bins
        self.grid_shape = grid_shape
        self.spline_order = spline_order
        self.spline_extrapolate = spline_extrapolate
        self.spline_bound = spline_bound
        self.diffable = diffable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input sanitization
        if len(x.shape) != 5:
            raise ValueError(
                f"Input tensor must have 5 dimensions, got {len(x.shape)}."
            )
        if (
            (x.shape[2] < self.grid_shape[0])
            or (x.shape[3] < self.grid_shape[1])
            or (x.shape[4] < self.grid_shape[2])
        ):
            raise ValueError(
                f"Input tensor must be larger than grid shape, got {x.shape[2:]} and {self.grid_shape}."
            )
        if self.diffable:
            # the autograd function takes string arguments for interpolation and extrapolation
            return clahe_3d_diffable(
                x,
                self.clip_limit,
                self.n_bins,
                self.grid_shape,
                self.spline_order,
                self.spline_extrapolate,
                self.spline_bound,
            )
        else:
            # the backend function takes integer arguments for interpolation and extrapolation
            bound = bound_to_nitorch(make_list(self.spline_bound), as_type="int")[0]
            extrapolate = int(self.spline_extrapolate)
            return clahe_3d_undiffable(
                x,
                clip_limit=self.clip_limit,
                n_bins=self.n_bins,
                grid_shape=self.grid_shape,
                spline_order=self.spline_order,
                spline_extrapolate=extrapolate,
                spline_bound=bound,
            )


@torch.jit.script
def clahe_3d(
    x: torch.Tensor,
    clip_limit: float,
    n_bins: int,
    grid_shape: Tuple[int, int, int],
    bandwidth: float = 0.001,
    spline_order: int = 5,
) -> torch.Tensor:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to a 3D tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, D, H, W).
        clip_limit (float): Clipping limit in pixels for the histogram bins.
        n_bins (int): Number of bins for the histogram.
        grid_shape (List[int]): Shape of the grid used to divide the input tensor into tiles.
        bandwidth (float, optional): Bandwidth parameter for the Gaussian kernel used to weight the histogram bins.
            Defaults to 0.001.
        spline_order (int, optional): Order of the interpolation used to compute the output tensor.
            Defaults to 5. This makes a big difference in CLAHE quality.

    Returns:
        torch.Tensor: Output tensor of shape (B, C, D, H, W) with the CLAHE applied.

    """

    # get shapes
    B, C, D, H, W = x.shape
    n_tiles = grid_shape[0] * grid_shape[1] * grid_shape[2]
    tile_shape = [
        int(math.ceil(D / grid_shape[0])),
        int(math.ceil(H / grid_shape[1])),
        int(math.ceil(W / grid_shape[2])),
    ]
    voxels_per_tile = tile_shape[0] * tile_shape[1] * tile_shape[2]

    # pad the input to be divisible by the tile counts in each dimension
    paddings = [0, 0, 0, 0, 0, 0]
    if D % grid_shape[0] != 0:
        paddings[0] = grid_shape[0] - (D % grid_shape[0])
    if H % grid_shape[1] != 0:
        paddings[2] = grid_shape[1] - (H % grid_shape[1])
    if W % grid_shape[2] != 0:
        paddings[4] = grid_shape[2] - (W % grid_shape[2])

    # PyTorch functional.pad uses last dimension to first in pairs
    x_padded = torch.nn.functional.pad(x, paddings[::-1], mode="reflect")

    # meshgrid of all of the coordinates for interpolation
    # the tile centers are at the corners of the interpolated image
    # so extrapolation outside of the unit square is needed of
    # the grid_pull function. How far out also depends on the tile counts
    coords_d = torch.linspace(
        -0.5 - (0.25 / grid_shape[0]),
        (grid_shape[0] - 1) + 0.5 + (0.25 / grid_shape[0]),
        x_padded.shape[-3],
        device=x.device,
    )
    coords_h = torch.linspace(
        -0.5 - (0.25 / grid_shape[1]),
        (grid_shape[1] - 1) + 0.5 + (0.25 / grid_shape[1]),
        x_padded.shape[-2],
        device=x.device,
    )
    coords_w = torch.linspace(
        -0.5 - (0.25 / grid_shape[2]),
        (grid_shape[2] - 1) + 0.5 + (0.25 / grid_shape[2]),
        x_padded.shape[-1],
        device=x.device,
    )
    coords = torch.meshgrid([coords_d, coords_h, coords_w], indexing="ij")
    coords = torch.stack(coords, dim=-1)

    # unfolding in 3D case requires successive calls with .unfold() instead
    tiles = x_padded.unfold(-3, tile_shape[0], tile_shape[0])
    tiles = tiles.unfold(-3, tile_shape[1], tile_shape[1])
    tiles = tiles.unfold(-3, tile_shape[2], tile_shape[2])

    # reshape from (B, C,) + (n_tiles_each_dim,)*3 + (voxels_along_each_dims)*3 to
    # be shape (B, C, n_tiles, voxels_per_tile)
    tiles = tiles.reshape((B, C, n_tiles, voxels_per_tile))

    # make the bin centers and reshape to (1, 1, n_bins)
    bins = torch.linspace(0.0, 1.0, n_bins, device=x.device)

    # here we pre-allocate the pdf tensor to avoid having all residuals in memory at once
    pdf = torch.zeros((B, C, n_tiles, n_bins), device=x.device)

    # iterate over combined batch and channel dimension
    for b in range(B):
        for c in range(C):
            for t in range(n_tiles):
                # calculate the pdf which is shape (B, C, n_tiles, n_bins)
                residuals_batch = tiles[b, c, t, :, None] - bins[None, :]
                weights_batch = torch.exp(-0.5 * (residuals_batch / bandwidth).pow(2))
                pdf[b, c, t] = torch.mean(weights_batch, dim=-2)
                pdf[b, c, t] = pdf[b, c, t] / (
                    torch.sum(pdf[b, c, t], dim=-1, keepdim=True) + 1e-10
                )

    # ------------ Taken directly from Kornia's implementation ------------

    # pdf is handled in "pixels" viewed as a large batch of 1D pdfs
    histos = (pdf * voxels_per_tile).view(-1, n_bins)

    if clip_limit > 0:
        # calc limit
        limit = max(clip_limit * voxels_per_tile // n_bins, 1)

        histos.clamp_(max=limit)

        # calculate the clipped pdf of shape (B, C, n_tiles, n_bins)
        clipped = voxels_per_tile - histos.sum(-1)

        # calculate the excess of shape (B, C, n_tiles, n_bins)
        residual = torch.remainder(clipped, n_bins)
        redist = (clipped - residual).div(n_bins)
        histos += redist[..., None]

        # trick to avoid using a loop to assign the residual
        v_range: torch.Tensor = torch.arange(n_bins, device=histos.device)
        mat_range: torch.Tensor = v_range.repeat(histos.shape[0], 1)
        histos += mat_range < residual[None].transpose(0, 1)

    # cdf (B, C, n_tiles, n_bins)
    cdf = torch.cumsum(histos, dim=-1) * (n_bins - 1) / voxels_per_tile

    # --------------- End of Kornia's implementation section ----------------

    # (B*C*n_tiles, n_bins) -> (B, C, GD, GH, GW, n_bins) -> (B, C, n_bins, GD, GH, GW)
    # if you use transpose, the interpolation will be wrong
    cdf = cdf.reshape(
        B,
        C,
        grid_shape[0],
        grid_shape[1],
        grid_shape[2],
        n_bins,
    ).movedim(-1, -4)

    # use grid_pull for high quality equalization between tile centers
    out = torch.empty_like(x_padded)
    for b in range(B):
        for c in range(C):
            # (D, H, W, 1) with (D, H, W, 3) -> (D, H, W, 4) to get the interpolation coordinate for each pixel
            coords_with_channel = torch.cat(
                [(x_padded[b, c][..., None] * (n_bins - 1.0)), coords], dim=-1
            )
            out[b, c] = grid_pull_backend(
                cdf[b, c][None, None],  # (1, 1, n_bins, GD, GH, GW)
                coords_with_channel[None, None],  # (1, D, H, W, 4)
                interpolation=[
                    spline_order,
                ],  # 2nd+ order makes a big difference
                extrapolate=1,
                bound=[
                    3,
                ],
            )[0, 0, 0]

    # slice the output to cut it from padded shape to original input shape
    out = out[:, :, : x.shape[-3], : x.shape[-2], : x.shape[-1]]

    # min max normalization
    out = (out - out.min()) / (out.max() - out.min() + 1e-10)
    return out


# from PIL import Image
# import numpy as np
# import os

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # load all images in current directory *.tif
# imgs = []

# for i in range(-15, 16):
#     # print(f'./ISE_{i+15:03d}_tilt_{i}.tif')
#     img = Image.open(f"scratch_data/ise/RegisteredISEImages/ISE_{i+15:03d}_tilt_{i}.tif")
#     img = np.array(img)
#     img = (img - img.min()) / (img.max() - img.min())
#     img = torch.from_numpy(img).to(device)[None, None, None]
#     img = img.float()
#     imgs.append(img)

# # stack all images into a single tensor
# stack = torch.cat(imgs, dim=2)  # (1, 1, 31, H, W)
# print("Stack shape:", stack.shape)

# # crop 100 pixels in on the H and W dimensions
# stack = stack[:, :, :, 100:-100, 100:-100]

# # only take the first 16 slices
# stack = stack[:, :, :16]

# # iterate over the D dimension and subtract the mean of each slice
# for i in range(stack.shape[2]):
#     stack[0, 0, i] -= stack[0, 0, i].mean()

# # add back in 0.5 to make the range [0, 1]
# stack += 0.5

# # if a folder for the pngs is not already made, make it
# if not os.path.exists("scratch_data/ise/ise_pngs"):
#     os.makedirs("scratch_data/ise/ise_pngs")

# # resave all the images as pngs
# for i in range(16):
#     img = Image.fromarray((stack[0, 0, i] * 255).byte().cpu().numpy())
#     img.save(f"scratch_data/ise/ise_pngs/ISE_{i:03d}.png")


# for spline in [0, 1, 2, 3, 4, 5]:
#     if not os.path.exists(f"scratch_data/ise/clahe_3d_output_spline_{spline}"):
#         os.makedirs(f"scratch_data/ise/clahe_3d_output_spline_{spline}")

#     # make output folder if it doesn't already exist
#     if not os.path.exists(f"scratch_data/ise/clahe_2d_output_spline_{spline}"):
#         os.makedirs(f"scratch_data/ise/clahe_2d_output_spline_{spline}")

#     # perform clahe on the entire stack
#     clahe_for_3d = Clahe3D(
#         clip_limit=40.0, n_bins=264, grid_shape=[2, 8, 8], spline_order=spline, diffable=False
#     )
#     clahe_from_3d = clahe_for_3d(stack.clone())
#     for i in range(16):
#         img_from_3d = Image.fromarray((clahe_from_3d[0, 0, i] * 255).byte().cpu().numpy())
#         img_from_3d.save(f"scratch_data/ise/clahe_3d_output_spline_{spline}/ISE_{i:03d}_clahe_3d_spline_{spline}.png")

#     # perform clahe individually on each image
#     clahe_for_2d = Clahe2D(
#         clip_limit=40.0, n_bins=264, grid_shape=[8, 8], spline_order=spline, diffable=False
#     )
#     clahe_from_2d = clahe_for_2d(stack.clone().movedim(2, 0)[:, :, 0])
#     for i in range(16):
#         img_from_2d = Image.fromarray((clahe_from_2d[i, 0] * 255).byte().cpu().numpy())
#         img_from_2d.save(f"scratch_data/ise/clahe_2d_output_spline_{spline}/ISE_{i:03d}_clahed_2d_spline_{spline}.png")


# # load all images in current directory *.tif
# imgs = []

# for i in range(100):
#     img = Image.open(f"scratch_data/coni_se/se_tifs/{i}.tif")
#     img = np.array(img)
#     img = (img - img.min()) / (img.max() - img.min())
#     img = torch.from_numpy(img).to(device)[None, None, None]
#     img = img.float()
#     imgs.append(img)

# # stack all images into a single tensor
# stack = torch.cat(imgs, dim=2)  # (1, 1, 100, H, W)
# print("Stack shape:", stack.shape)

# # crop off the top and bottom 100 pixels in the H and W dimensions
# stack = stack[:, :, :, 100:-100, 250:-150]
# print("Trimmed shape:", stack.shape)

# # zero mean each slice then add back 0.5 and clamp to [0, 1]
# for i in range(stack.shape[2]):
#     stack[0, 0, i] -= stack[0, 0, i].mean()
#     stack[0, 0, i] += 0.5
#     stack[0, 0, i] = torch.clamp(stack[0, 0, i], 0.0, 1.0)

# # if a folder for the pngs is not already made, make it
# if not os.path.exists("scratch_data/coni_se/se_pngs"):
#     os.makedirs("scratch_data/coni_se/se_pngs")

# # resave all the images as pngs
# for i in range(100):
#     img = Image.fromarray((stack[0, 0, i] * 255).byte().cpu().numpy())
#     img.save(f"scratch_data/coni_se/se_pngs/SE_{i:03d}.png")


# for spline in [0, 1, 2, 3, 4, 5]:
#     # make folder for each spline order
#     if not os.path.exists(f"scratch_data/coni_se/clahe_2d_{spline}"):
#         os.makedirs(f"scratch_data/coni_se/clahe_2d_{spline}")

#     if not os.path.exists(f"scratch_data/coni_se/clahe_3d_{spline}"):
#         os.makedirs(f"scratch_data/coni_se/clahe_3d_{spline}")

#     # perform clahe on the entire stack
#     clahe_for_3d = Clahe3D(
#         clip_limit=40.0,
#         n_bins=256,
#         grid_shape=[8, 8, 8],
#         spline_order=spline,
#         diffable=False,
#     )
#     clahe_from_3d = clahe_for_3d(stack.clone())

#     # save clahe 3d output
#     for i in range(100):
#         img_from_3d = Image.fromarray(
#             (clahe_from_3d[0, 0, i] * 255).byte().cpu().numpy()
#         )
#         img_from_3d.save(
#             f"scratch_data/coni_se/clahe_3d_{spline}/SE_{i:03d}_clahe_3d_spline_{spline}.png"
#         )

#     # perform clahe individually on each image
#     clahe_for_2d = Clahe2D(
#         clip_limit=40.0,
#         n_bins=256,
#         grid_shape=[8, 8],
#         spline_order=spline,
#         diffable=False,
#     )
#     clahe_from_2d = clahe_for_2d(stack.clone().movedim(2, 0)[:, :, 0])

#     # save clahe 2d output
#     for i in range(100):
#         img_from_2d = Image.fromarray((clahe_from_2d[i, 0] * 255).byte().cpu().numpy())
#         img_from_2d.save(
#             f"scratch_data/coni_se/clahe_2d_{spline}/SE_{i:03d}_clahed_2d_spline_{spline}.png"
#         )

# # @torch.jit.script
# def mclahe(x: torch.Tensor,
#            n_clahe_dims: int,
#            clip_limit: float,
#            n_bins: int,
#            grid_shape: Optional[List[int]] = None,
#            bandwidth: float = 0.001,) -> torch.Tensor:
#     """
#     Multidimensional Contrast Limited Adaptive Histogram Equalization.

#     Args:
#     - x (torch.Tensor): The input tensor.
#     - n_clahe_dims (int): The number of spatial dimensions to perform CLAHE on.
#     - clip_limit (float): The clip limit for CLAHE. Not in pixels, but in the
#         range [0, 1]. If 0, no clipping is performed. Used on the actual PDF values.
#     - n_bins (int): The number of bins to use for the histogram.
#     - bandwidth (float): The bandwidth of the Gaussian kernel.
#     - tile_counts (list): The number of tiles to use in each dimension.
#         If None, defaults to 8 for each dimension.

#     """
#     # Sanitization
#     if n_clahe_dims > 4:
#         raise ValueError("CLAHE is only supported for 2D, 3D, 4D spatiotemporal inputs")
#     if n_clahe_dims < 2:
#         raise ValueError("CLAHE must be performed on at least 2 dimensions")
#     if n_clahe_dims > len(x.shape) - 2:
#         raise ValueError("At most, two excess dimensions are allowed for batch and channel")

#     # if there are two excess dimensions, assume they are batch and channel
#     clahe_shape = x.shape[-n_clahe_dims:]

#     B = 1 if x.ndim == n_clahe_dims else x.shape[0]
#     C = 1 if x.ndim == n_clahe_dims else x.shape[1]

#     if grid_shape is None:
#         grid_shape = [8 for _ in range(n_clahe_dims)]

#     # pad the input to be divisible by the tile counts in each dimension
#     paddings = torch.zeros((2 * n_clahe_dims,), dtype=torch.int64)
#     for i in range(n_clahe_dims):
#         if clahe_shape[i] % grid_shape[i] != 0:
#             paddings[2*i + 0] = grid_shape[i] - (clahe_shape[i] % grid_shape[i])
#             paddings[2*i + 1] = 0
#         else:
#             paddings[2*i + 0] = 0
#             paddings[2*i + 1] = 0

#     paddings: List[int] = paddings.tolist()

#     # paddings must be reversed for torch.nn.functional.pad as it accepts last dimension to first
#     if n_clahe_dims == 2 or n_clahe_dims == 3:
#         x_padded = torch.nn.functional.pad(x, paddings[::-1], mode='reflect')
#     else:
#         # manually pad wihtout reflection
#         x_padded = torch.zeros(
#             (x.shape[0], x.shape[1],
#                 x.shape[2] + paddings[0],
#                 x.shape[3] + paddings[2],
#                 x.shape[4] + paddings[4],
#                 x.shape[5] + paddings[6]),
#         )
#     print(x_padded.shape)

#     # calculate the tile dimensions, pixels per tile, number of tiles, and tile centers
#     # tile_centers = [torch.linspace(0, tile_counts[-i], x_padded.shape[-i], device=x.device) for
#     #                 i in range(1, n_clahe_dims + 1)][::-1]
#     # tile_dims = [int(math.ceil(x_padded.shape[-i] / tile_counts[-i])) for i in range(1, n_clahe_dims + 1)][::-1]
#     # voxels_per_tile = int(torch.prod(torch.tensor(tile_dims)).item())
#     # n_tiles = int(torch.prod(torch.tensor(tile_counts)).item())

#     tile_centers: List[torch.Tensor] = []
#     for i in range(1, n_clahe_dims + 1):
#         tile_center = torch.linspace(0.5 / grid_shape[-i], 1 - (0.5 / grid_shape[-i]), grid_shape[-i], device=x.device)
#         tile_centers.append(tile_center)
#     tile_centers.reverse()

#     tile_dims: List[int] = []
#     for i in range(1, n_clahe_dims + 1):
#         tile_dim = int(math.ceil(x_padded.shape[-i] / grid_shape[-i]))
#         tile_dims.append(tile_dim)
#     tile_dims.reverse()

#     voxels_per_tile = int(torch.prod(torch.tensor(tile_dims)).item())
#     n_tiles = int(torch.prod(torch.tensor(grid_shape)).item())

#     # perform the interpolation using a meshgrid of all of the coordinates
#     # first, make the meshgrid with all coordinates, not just the tile centers
#     # offset = 0.5 * (n_clahe_dims ** 0.5)
#     coords = torch.meshgrid(*[torch.linspace(-0.5 - (0.25 / grid_shape[-i]),
#                                             (grid_shape[-i] - 1) + 0.5 + (0.25 / grid_shape[-i]),
#                                              x_padded.shape[-i], device=x.device
#                                              ) for
#                                    i in range(1, n_clahe_dims + 1)][::-1], indexing='ij')
#     coords = torch.stack(coords, dim=-1)

#     # unfold the input into tiles of shape (B, C*voxels_per_tile, -1)
#     print("x shape", x.shape)
#     print("x_padded.shape", x_padded.shape)

#     if n_clahe_dims == 2:
#         tiles = torch.nn.functional.unfold(x_padded, tile_dims, stride=tile_dims)
#         # reshape from (B, C*voxels, n_tiles) to (B, C, voxels_per_tile, n_tiles)
#         tiles = tiles.reshape((B, C, voxels_per_tile, n_tiles))
#         # permute from (B, C, voxels_per_tile, n_tiles) to (B, C, n_tiles, voxels_per_tile)
#         tiles = tiles.transpose(-1, -2)
#     elif n_clahe_dims == 3:
#         # need to use .unfold() instead
#         tiles = x_padded.unfold(-3, tile_dims[0], tile_dims[0])
#         tiles = tiles.unfold(-3, tile_dims[1], tile_dims[1])
#         tiles = tiles.unfold(-3, tile_dims[2], tile_dims[2])
#         # reshape from (B, C,) + (n_tiles_each_dim,)*3 + (voxels_along_each_dims)*3 to
#         # be shape (B, C, n_tiles, voxels_per_tile)
#         tiles = tiles.reshape((B, C, n_tiles, voxels_per_tile))
#     else:
#         # need to use .unfold() instead
#         tiles = x_padded.unfold(-4, tile_dims[0], tile_dims[0])
#         tiles = tiles.unfold(-4, tile_dims[1], tile_dims[1])
#         tiles = tiles.unfold(-4, tile_dims[2], tile_dims[2])
#         tiles = tiles.unfold(-4, tile_dims[3], tile_dims[3])
#         print(tiles.shape)
#         # reshape from (B, C,) + (n_tiles_each_dim,)*4 + (voxels_along_each_tile_dim)*4 to
#         # be shape (B, C, n_tiles, voxels_per_tile)
#         tiles = tiles.reshape((B, C, n_tiles, voxels_per_tile))

#     print("Final tiles shape: ", tiles.shape)


#     # make the bin centers
#     # bins = torch.linspace(0.5 / n_bins, 1 - (0.5 / n_bins), n_bins, device=x.device)
#     bins = torch.linspace(0, 1, n_bins, device=x.device)
#     # subtract the tiles from the bin centers using broadcasting
#     diffs = tiles[:, :, :, :, None] - bins[None, None, None, None, :]
#     print("Residuals min max", diffs.min().item(), diffs.max().item())
#     # calculate the bandwidth using silverman's rule of thumb
#     # bandwidth = 1.06 * torch.std(residuals, dim=-1, keepdim=True) * (n_tiles)**(-1/5)
#     # use Gaussian kernel to calculate the weights
#     print("diffs.shape", diffs.shape)
#     weights = torch.exp(-0.5 * (diffs / bandwidth).pow(2))
#     # calculate the pdf which is shape (B, C, n_tiles, n_bins)
#     pdf = torch.mean(weights, dim=-2)
#     # normalize the pdf which is shape (B, C, n_tiles, n_bins)
#     pdf = pdf / (torch.sum(pdf, dim=-1, keepdim=True) + 1e-10)

#     print("Starting pdf min max", pdf.min().item(), pdf.max().item())

#     # pdf is handled in "pixels"
#     histos = pdf * voxels_per_tile

#     histos = histos.view(-1, n_bins)

#     if clip_limit > 0:
#         # calc limit
#         limit = max(clip_limit * voxels_per_tile // n_bins, 1)

#         histos.clamp_(max=limit)

#         # calculate the clipped pdf of shape (B, C, n_tiles, n_bins)
#         clipped = voxels_per_tile - histos.sum(-1)

#         # calculate the excess of shape (B, C, n_tiles, n_bins)
#         residual = torch.remainder(clipped, n_bins)
#         redist = (clipped - residual).div(n_bins)
#         histos += redist[..., None]

#         # trick to avoid using a loop to assign the residual
#         v_range: torch.Tensor = torch.arange(n_bins, device=histos.device)
#         mat_range: torch.Tensor = v_range.repeat(histos.shape[0], 1)
#         histos += mat_range < residual[None].transpose(0, 1)

#         # # add the excess per bin to the clipped pdf
#         # clipped_pdf += total_excess_per_bin

#         # clipped_pdf = torch.clamp(clipped_pdf, max=clip_limit)
#         # excess_2 = pdf - clipped_pdf
#         # total_excess_2_per_bin = torch.sum(excess_2, dim=-1, keepdim=True) / n_bins
#         # clipped_pdf += total_excess_2_per_bin

#         # for i in range(10):
#         #     clipped_pdf = torch.clamp(clipped_pdf, max=clip_limit)
#         #     excess_2 = pdf - clipped_pdf
#         #     total_excess_2_per_bin = torch.sum(excess_2, dim=-1, keepdim=True) / n_bins
#         #     clipped_pdf += total_excess_2_per_bin

#         # # normalize the clipped pdf
#         # pdf = torch.nn.functional.normalize(clipped_pdf, p=1.0, dim=-1)


#     # calculate the cdf
#     print("pdf before cumsum shape", histos.shape)
#     cdf = torch.cumsum(histos, dim=-1) * (n_bins - 1) / voxels_per_tile
#     cdf.clamp_(min=0, max=n_bins-1)
#     print("cdf shape after cumsum", cdf.shape)

#     # normalize the cdf so it is in the range [0, 1]
#     # cdf = (cdf - torch.min(cdf, dim=-1, keepdim=True)[0]) / (torch.max(cdf, dim=-1, keepdim=True)[0] - torch.min(cdf, dim=-1, keepdim=True)[0])

#     print("CDF first entry min max", cdf[..., 0].min().item(), cdf[..., 0].max().item())

#     print("CDF last entry min max", cdf[..., -1].min().item(), cdf[..., -1].max().item())

#     print("cdf shape", cdf.shape)

#     # use grid_pull from cmir.warps.splines.api to perform the interpolation
#     # first, reshape the cdf to be (B, C, n_tiles, n_bins, n_clahe_dims)
#     print("cdf before reshape: ", cdf.shape)
#     # if a transpose was used it would not be the same as movedim
#     cdf = cdf.reshape(B, C, *grid_shape, n_bins,).movedim(-1, -n_clahe_dims-1)
#     print("cdf after reshape: ", cdf.shape)

#     # the bin dimension will also be linearly interpolated so prepend the original
#     # pixel values to the out_coords as this are used to index into the cdf dimension
#     out = []
#     for b in range(B):
#         for i in range(C):
#             coords_per_channel = torch.cat([x_padded[b, i][..., None] * (n_bins - 1.0), coords], dim=-1)
#             print("coords_per_channel_min_max", x_padded[b, i][..., None].min(), x_padded[b, i][..., None].max())
#             out.append(grid_pull(cdf[b, i], coords_per_channel, interpolation=3, extrapolate=True, bound='nearest'))
#     out = torch.stack(out, dim=0).reshape((B, C, *x_padded.shape[-n_clahe_dims:]))

#     # # slice the output to cut it from padded shape to original input shape
#     # if n_clahe_dims == 2:
#     #     out = out[:, :, :x.shape[-2], :x.shape[-1]]
#     # elif n_clahe_dims == 3:
#     #     out = out[:, :, :x.shape[-3], :x.shape[-2], :x.shape[-1]]
#     # else:
#     #     out = out[:, :, :x.shape[-4], :x.shape[-3], :x.shape[-2], :x.shape[-1]]

#     # # min max normalization
#     out = (out - out.min()) / (out.max() - out.min() + 1e-10)

#     print("Final out shape", out.shape)
#     return out, x_padded


# # test it out on a real image "cmir_banner.png"
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt

# img = Image.open("4000.png")
# img = np.array(img)
# img = (img - img.min()) / (img.max() - img.min())

# print(img.min(), img.max())
# print("img shape", img.shape)

# # convert to torch tensor
# img = torch.from_numpy(img).float().to('cuda')[None, None]
# img = img.repeat(1, 1, 1, 1)

# # perform mclahe
# n_clahe = 2
# clip_limit = 40.0
# nbins = 256
# # out, x_padded = mclahe(img, n_clahe_dims=n_clahe, clip_limit=clip_limit, n_bins=nbins, tile_counts=[8, 8])
# out = clahe_2d(img, clip_limit=clip_limit, n_bins=nbins, grid_shape=[8, 8], spline_order=5)

# # convert back to numpy
# out = out.cpu().numpy().squeeze()

# print(out.min(), out.max())

# # save via PIL
# out = Image.fromarray((out * 255).astype(np.uint8)).convert('RGB')

# # out = Image.fromarray(out)

# out.save("4000_mclahe.png")


# # test out 3D case
# img_3d = img[None].repeat(1, 1, 8, 1, 1)
# print(img_3d.shape)

# # perform mclahe with clahe_3d
# out_3d = clahe_3d(img_3d, clip_limit=clip_limit, n_bins=nbins, grid_shape=[8, 8, 8], spline_order=3)

# # visualize the top image of the stack
# out_3d = out_3d[0, 0, 0].cpu().numpy().squeeze()
# out_3d = Image.fromarray((out_3d * 255).astype(np.uint8)).convert('RGB')
# out_3d.save("4000_mclahe_3d.png")


# # test out 3D case
# img_3d = img[None].repeat(1, 1, 16, 1, 1)
# print(img_3d.shape)

# # perform mclahe with clahe_3d
# out_3d = clahe_3d(img_3d, clip_limit=clip_limit, n_bins=nbins, grid_shape=[8, 8, 8], spline_order=3)

# # visualize the top image of the stack
# out_3d = out_3d[0, 0, 0].cpu().numpy().squeeze()
# out_3d = Image.fromarray((out_3d * 255).astype(np.uint8)).convert('RGB')
# out_3d.save("4000_mclahe_3d.png")


# @torch.jit.script
# def single_dim_select(input: torch.Tensor, indices: torch.Tensor, dim: int):
#     """
#     Perform index_select on each batch along the specified dimension.

#     Args:
#     - input (torch.Tensor): The input tensor.
#     - dim (int): The dimension to select along.
#     - index (torch.Tensor): The indices to select.

#     Returns:
#     - torch.Tensor: The selected slices of the input tensor along the dimension.
#     """

#     assert dim < len(input.shape), "Axis out of bounds"
#     # Use torch.gather and indices expanded to the right shape
#     indices = indices.reshape((1,) * dim + (-1,) + (1,)* (len(input.shape) - dim - 1))
#     new_shape = [input.shape[i] if i != dim else -1 for i in range(len(input.shape))]
#     indices = indices.expand(new_shape)
#     return torch.gather(input, dim, indices)
