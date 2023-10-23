"""
Weights and derivatives of spline orders 0 to 7. 

From Yael Balbastre's package interpol:

https://github.com/balbasty/torch-interpol/

Includes Yael's AutoGrad version of pull/push/count/grad

"""

import torch
from typing import Tuple

from .jit_utils import movedim1
from .api import grid_pull, grid_push, identity_grid


__all__ = [
    "BSplineWarp"
]


class BSplineWarp(torch.nn.Module):
    """BSpline coordinate warp. 

    This module modified input pixel coordinates to new locations.
    
    The nn.Parameter ``displacements`` is the focus of this module. 
    It is a coarse grid of displacements that can be updated via 
    gradient descent. The displacements are used in the forward pass 
    to smoothly distort a dense grid of pixel coordinates to new locations. 

    All image coordinates are assumed to be in the range ``[0, 1]``.

    Parameters
    ----------
    shape : Tuple[int, int]
        Shape of the images for which the coordinates are defined
    control_points : Tuple[int, int]
        Number of control points along each dimension.
    bound : BoundType or sequence[BoundType], default='zero'
        Boundary conditions.
    extrapolate : bool or int, default=False
        Extrapolate out-of-bound data.
    interpolation : int or sequence[int], default=3
        Interpolation order. Up to order 7 is supported.

    """

    def __init__(
        self,
        n_transforms: int,
        image_shape: Tuple[int, int],
        control_shape: Tuple[int, int],
        interpolation="cubic", 
        bound="nearest", 
        extrapolate=1,
    ):
        super().__init__()
        self.bound = bound
        self.extrapolate = extrapolate
        self.interpolation = interpolation
        self.control_shape = control_shape
        
        # make a coarse grid of displacements
        self.displacements = torch.nn.Parameter(torch.zeros(n_transforms, 2, *control_shape))

        coordinate_grid = identity_grid(image_shape, torch.float32, self.displacements.device)
        coordinate_grid = coordinate_grid.expand(n_transforms, *coordinate_grid.shape)
        self.register_buffer("coordinate_grid", coordinate_grid)

    def forward(self, image_coordinates):
        displacements_upsampled = torch.nn.functional.interpolate(
            self.displacements, 
            size=image_coordinates.shape[-3:-1], 
            mode='bicubic', 
            align_corners=False
        )
        coodinate_deltas = grid_pull(
            displacements_upsampled,
            self.coordinate_grid,
            bound=self.bound,
            extrapolate=self.extrapolate,
            interpolation=self.interpolation,
        )
        warped_coordinates = image_coordinates + movedim1(coodinate_deltas, -3, -1)
        return warped_coordinates
    
    def inverse(self, image_coordinates):
        displacements_upsampled = torch.nn.functional.interpolate(
            self.displacements, 
            size=image_coordinates.shape[-3:-1], 
            mode='bicubic', 
            align_corners=False
        )
        coodinate_deltas = grid_push(
            displacements_upsampled,
            self.coordinate_grid,
            bound=self.bound,
            extrapolate=self.extrapolate,
            interpolation=self.interpolation,
        )
        warped_coordinates = image_coordinates - movedim1(coodinate_deltas, -3, -1)
        return warped_coordinates
        