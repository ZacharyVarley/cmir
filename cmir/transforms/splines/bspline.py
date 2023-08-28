"""
Weights and derivatives of spline orders 0 to 7. 

From Yael Balbastre's package interpol:

https://github.com/balbasty/torch-interpol/

Includes Yael's AutoGrad version of pull/push/count/grad

"""

import torch
from typing import Tuple

from .jit_utils import movedim1
from .api import grid_pull, identity_grid


__all__ = [
    "BSplineTransform"
]


class BSplineTransform(torch.nn.Module):
    """BSpline transform.

    All image coordinates are assumed to be in the range ``[0, 1]``.

    Parameters
    ----------
    shape : Tuple[int, int]
        Shape of the images for which the coordinates are defined
    control_points : Tuple[int, int]
        Number of control points along each dimension.
    bound : BoundType or sequence[BoundType], default='dct2'
        Boundary conditions.
    extrapolate : bool or int, default=False
        Extrapolate out-of-bound data.
    interpolation : int or sequence[int], default=3
        Interpolation order. Up to order 7 is supported.

    """

    def __init__(
        self, 
        image_shape: Tuple[int, int],
        control_shape: Tuple[int, int],
        interpolation="linear", 
        bound="zero", 
        extrapolate=False
    ):
        super().__init__()
        self.image_shape = image_shape
        self.control_shape = control_shape
        self.bound = bound
        self.extrapolate = extrapolate
        self.interpolation = interpolation
        
        # make a coarse grid of control points
        control_points = identity_grid(control_shape, dtype=torch.float32)

        # normalize to [0, 1]
        control_points = control_points / (torch.tensor(control_shape, dtype=torch.float32) - 1)

        # (..., H, W, 2) grid -> (..., 2, H, W) "image" of coeffs
        control_points = movedim1(control_points, -1, -3)

        # make a buffer self.control_points that follows the parent module device / type
        self.register_buffer('control_points', control_points)

        # displacements from the control points are the focus
        self.displacements = torch.nn.Parameter(torch.zeros_like(control_points))

    def forward(self, image_coordinates):
        # "image" here is a coarse grid of values that indicate new pixel positions
        # we pull this coarse image from a dense grid with all of the pixel coordinates
        # each pixel in the image is 2 channel, on for each deformation dimension
        # (..., 2, H, W) image and (..., H, W, 2) grid -> (..., 2, H, W) output
        coordinate_image = grid_pull(self.control_points + self.displacements, 
                                image_coordinates,
                                interpolation=self.interpolation,
                                bound=self.bound,
                                extrapolate=self.extrapolate)
        # (..., 2, H, W) output -> (..., H, W, 2) grid
        return movedim1(coordinate_image, -3, -1)
    
