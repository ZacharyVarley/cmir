"""
This is a PyTorch implementation of mclahe (multidimensional 
contrast limited adaptive histogram equalization) from:

V. Stimper, S. Bauer, R. Ernstorfer, B. Schölkopf and R. P. Xian, 
"Multidimensional Contrast Limited Adaptive Histogram Equalization," 
in IEEE Access, vol. 7, pp. 165437-165447, 2019.

https://github.com/VincentStimper/mclahe

That codebase is also MIT licensed.

"""

import math
from typing import List, Optional, Union
import torch
from cmir.warps.splines.api import grid_pull


@torch.jit.script
def single_dim_select(input: torch.Tensor, indices: torch.Tensor, dim: int):
    """
    Perform index_select on each batch along the specified dimension.

    Args:
    - input (torch.Tensor): The input tensor.
    - dim (int): The dimension to select along.
    - index (torch.Tensor): The indices to select.

    Returns:
    - torch.Tensor: The selected slices of the input tensor along the dimension.
    """

    assert dim < len(input.shape), "Axis out of bounds"
    # Use torch.gather and indices expanded to the right shape
    indices = indices.reshape((1,) * dim + (-1,) + (1,)* (len(input.shape) - dim - 1))
    new_shape = [input.shape[i] if i != dim else -1 for i in range(len(input.shape))]
    indices = indices.expand(new_shape)
    return torch.gather(input, dim, indices)


# @torch.jit.script
def batch_pdf_2D_input(x: torch.Tensor,
                             bins: int):
    print(x.shape[0], bins)
    pdf = torch.zeros((x.shape[0], bins), device=x.device)
    for i in range(x.shape[0]):
        pdf[i] = torch.histogram(x[i], bins=bins, density=True)
    return pdf




# @torch.jit.script
def mclahe(x: torch.Tensor,
           n_clahe_dims: int,
           clip_limit: float,
           n_bins: int,
           tile_counts: Optional[List[int]] = None,
           bandwidth: float = 0.001,) -> torch.Tensor:
    """
    Multidimensional Contrast Limited Adaptive Histogram Equalization.

    Args:
    - x (torch.Tensor): The input tensor.
    - n_clahe_dims (int): The number of spatial dimensions to perform CLAHE on.
    - clip_limit (float): The clip limit for CLAHE. Not in pixels, but in the 
        range [0, 1]. If 0, no clipping is performed. Used on the actual PDF values.
    - n_bins (int): The number of bins to use for the histogram.
    - bandwidth (float): The bandwidth of the Gaussian kernel.
    - tile_counts (list): The number of tiles to use in each dimension. 
        If None, defaults to 8 for each dimension.

    """
    # Sanitization
    if n_clahe_dims > 4:
        raise ValueError("CLAHE is only supported for 2D, 3D, 4D spatiotemporal inputs")
    if n_clahe_dims < 2:
        raise ValueError("CLAHE must be performed on at least 2 dimensions")
    if n_clahe_dims > len(x.shape) - 2:
        raise ValueError("At most, two excess dimensions are allowed for batch and channel")

    # if there are two excess dimensions, assume they are batch and channel
    clahe_shape = x.shape[-n_clahe_dims:]

    B = 1 if x.ndim == n_clahe_dims else x.shape[0]
    C = 1 if x.ndim == n_clahe_dims else x.shape[1]

    if tile_counts is None:
        tile_counts = [8 for _ in range(n_clahe_dims)]

    # pad the input to be divisible by the tile counts in each dimension
    paddings = torch.zeros((2 * n_clahe_dims,), dtype=torch.int64)
    for i in range(n_clahe_dims):
        if clahe_shape[i] % tile_counts[i] != 0:
            paddings[2*i + 0] = tile_counts[i] - (clahe_shape[i] % tile_counts[i])
            paddings[2*i + 1] = 0
        else:
            paddings[2*i + 0] = 0
            paddings[2*i + 1] = 0
    
    paddings: List[int] = paddings.tolist()

    # paddings must be reversed for torch.nn.functional.pad as it accepts last dimension to first
    if n_clahe_dims == 2 or n_clahe_dims == 3:
        x_padded = torch.nn.functional.pad(x, paddings[::-1], mode='reflect')
    else:
        # manually pad wihtout reflection
        x_padded = torch.zeros(
            (x.shape[0], x.shape[1], 
                x.shape[2] + paddings[0], 
                x.shape[3] + paddings[2], 
                x.shape[4] + paddings[4], 
                x.shape[5] + paddings[6]),
        )
    print(x_padded.shape)

    # # calculate the tile dimensions, pixels per tile, number of tiles, and tile centers
    # tile_centers = [torch.linspace(0.5 / tile_counts[-i], 1 - (0.5 / tile_counts[-i]), tile_counts[-i], device=x.device) for 
    #                 i in range(1, n_clahe_dims + 1)][::-1]
    # tile_dims = [int(math.ceil(x_padded.shape[-i] / tile_counts[-i])) for i in range(1, n_clahe_dims + 1)][::-1]
    # voxels_per_tile = int(torch.prod(torch.tensor(tile_dims)).item())
    # n_tiles = int(torch.prod(torch.tensor(tile_counts)).item())

    tile_centers: List[torch.Tensor] = []
    for i in range(1, n_clahe_dims + 1):
        tile_center = torch.linspace(0.5 / tile_counts[-i], 1 - (0.5 / tile_counts[-i]), tile_counts[-i], device=x.device)
        tile_centers.append(tile_center)
    tile_centers.reverse()

    # perform the interpolation using a meshgrid of all of the coordinates
    # first, make the meshgrid with all coordinates, not just the tile centers
    coords = torch.meshgrid(*[torch.linspace(0, 1, x_padded.shape[-i], device=x.device) for
                                   i in range(1, n_clahe_dims + 1)][::-1], indexing='ij')
    coords = torch.stack(coords, dim=-1)

    tile_dims: List[int] = []
    for i in range(1, n_clahe_dims + 1):
        tile_dim = int(math.ceil(x_padded.shape[-i] / tile_counts[-i]))
        tile_dims.append(tile_dim)
    tile_dims.reverse()

    voxels_per_tile = int(torch.prod(torch.tensor(tile_dims)).item())
    n_tiles = int(torch.prod(torch.tensor(tile_counts)).item())

    # unfold the input into tiles of shape (B, C*voxels_per_tile, -1)
    print("x shape", x.shape)
    print("x_padded.shape", x_padded.shape)

    if n_clahe_dims == 2:
        tiles = torch.nn.functional.unfold(x_padded, tile_dims, stride=tile_dims)
        # reshape from (B, C*voxels, n_tiles) to (B, C, voxels_per_tile, n_tiles)
        tiles = tiles.reshape((B, C, voxels_per_tile, n_tiles))
        # permute from (B, C, voxels_per_tile, n_tiles) to (B, C, n_tiles, voxels_per_tile)
        tiles = tiles.transpose(-1, -2)
    elif n_clahe_dims == 3:
        # need to use .unfold() instead
        tiles = x_padded.unfold(-3, tile_dims[0], tile_dims[0])
        tiles = tiles.unfold(-3, tile_dims[1], tile_dims[1])
        tiles = tiles.unfold(-3, tile_dims[2], tile_dims[2])
        # reshape from (B, C,) + (n_tiles_each_dim,)*3 + (voxels_along_each_dims)*3 to
        # be shape (B, C, n_tiles, voxels_per_tile)
        tiles = tiles.reshape((B, C, n_tiles, voxels_per_tile))
    else:
        # need to use .unfold() instead
        tiles = x_padded.unfold(-4, tile_dims[0], tile_dims[0])
        tiles = tiles.unfold(-4, tile_dims[1], tile_dims[1])
        tiles = tiles.unfold(-4, tile_dims[2], tile_dims[2])
        tiles = tiles.unfold(-4, tile_dims[3], tile_dims[3])
        print(tiles.shape)
        # reshape from (B, C,) + (n_tiles_each_dim,)*4 + (voxels_along_each_tile_dim)*4 to 
        # be shape (B, C, n_tiles, voxels_per_tile)
        tiles = tiles.reshape((B, C, n_tiles, voxels_per_tile))

    print("Final tiles shape: ", tiles.shape)

        
    # make the bin centers
    bins = torch.linspace(0.5 / n_bins, 1 - (0.5 / n_bins), n_bins, device=x.device)
    # subtract the tiles from the bin centers using broadcasting
    residuals = tiles[:, :, :, :, None] - bins[None, None, None, None, :]
    # use Gaussian kernel to calculate the weights
    weights = torch.exp(-0.5 * (residuals / bandwidth).pow(2))
    # calculate the pdf which is shape (B, C, n_tiles, n_bins)
    pdf = torch.mean(weights, dim=-2)
    # normalize the pdf then scale by the number of voxels per tile
    pdf = torch.nn.functional.normalize(pdf, p=1.0, dim=-1)

    if clip_limit > 0:
        # calculate the clipped pdf of shape (B, C, n_tiles, n_bins)
        clipped_pdf = torch.clamp(pdf, max=clip_limit)

        # calculate the excess of shape (B, C, n_tiles, n_bins)
        excess = pdf - clipped_pdf

        # total excess of shape (B, C, n_tiles)
        total_excess = torch.sum(excess, dim=-1, keepdim=True)

        # redistribute the excess
        excess_per_bin = total_excess / n_bins

        # add the excess per bin to the clipped pdf
        clipped_pdf += excess_per_bin

        # normalize the clipped pdf
        pdf = torch.nn.functional.normalize(clipped_pdf, p=1.0, dim=-1)

    # calculate the cdf
    cdf = torch.cumsum(pdf, dim=-1)

    # use bilinear or trilinear interpolattion with cdf's at the tile centers
    # first, reshape the tile centers to be (B, C, n_tiles, n_clahe_dims)
    # this will require meshgrid with indexing='ij'
    tile_centers = torch.meshgrid(*tile_centers, indexing='ij')
    tile_centers = torch.stack(tile_centers, dim=-1)

    # use grid_pull from cmir.warps.splines.api to perform the interpolation
    # first, reshape the cdf to be (B, C, n_tiles, n_bins, n_clahe_dims)
    cdf = cdf.reshape(B, C, *tile_counts, n_bins,).transpose(-1, 2)
    print("cdf shape", cdf.shape)

    # out_coords = out_coords.reshape((1, *(out_coords.shape))).expand((1, *(out_coords.shape)))
    # the bin dimension will also be linearly interpolated so prepend the original
    # pixel values to the out_coords as this are used to index into the cdf dimension
    # for i in range(C):
    
    out = []
    for b in range(B):
        for i in range(C):
            coords_per_channel = torch.cat([x_padded[b, i][..., None], coords], dim=-1)
            out.append(grid_pull(cdf[b, i], coords_per_channel, interpolation='linear'))
    out = torch.stack(out, dim=0).reshape((B, C, *x_padded.shape[-n_clahe_dims:]))
    
    # slice the output to cut it from padded shape to original input shape
    if n_clahe_dims == 2:
        out = out[:, :, :x.shape[-2], :x.shape[-1]]
    elif n_clahe_dims == 3:
        out = out[:, :, :x.shape[-3], :x.shape[-2], :x.shape[-1]]
    else:
        out = out[:, :, :x.shape[-4], :x.shape[-3], :x.shape[-2], :x.shape[-1]]

    print("Final out shape", out.shape)
    return out



# test mclahe
x = torch.randn(2, 3, 100, 128)
n_clahe = 2
clip_limit = 0.2
nbins = 16
out = mclahe(x, n_clahe_dims=n_clahe, clip_limit=clip_limit, n_bins=nbins)

print("-"*80)

# test mclahe
x = torch.randn(2, 3, 45, 29, 31)
n_clahe = 3
clip_limit = 0.2
nbins = 16
out = mclahe(x, n_clahe_dims=n_clahe, clip_limit=clip_limit, n_bins=nbins)

print("-"*80)

# test mclahe
x = torch.randn(2, 3, 20, 20, 20, 20)
n_clahe = 4
clip_limit = 0.2
nbins = 16
out = mclahe(x, n_clahe_dims=n_clahe, clip_limit=clip_limit, n_bins=nbins)
