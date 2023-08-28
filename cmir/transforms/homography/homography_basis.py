"""
This module contains the implementation of the HomographyLinearCombination 
transform. This transform defines a dense flow field by linearly combining 
a set of basis homographies. There are 8 basis homographies, which are 
updated during registration. The basis homographies are defined by the
following parameters: translation, shear, scale, and perspective in x and y.
"""

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "HomographyBasisTransform",
]


@torch.jit.script
def gen_flow_bases(
    height: int,
    width: int,
    weights: Tensor,
):
    w_translation, w_shear, w_scale, w_perspective = weights.unbind()
    # create a tensor of homography matrices and fill with the weights
    homography_mats = torch.zeros(8, 3, 3, dtype=torch.float32, device=weights.device)
    homography_mats[:, 0, 0] = 1.0
    homography_mats[:, 1, 1] = 1.0
    homography_mats[:, 2, 2] = 1.0
    homography_mats[0, 0, 2] = w_translation
    homography_mats[1, 1, 2] = w_translation
    homography_mats[2, 0, 1] = w_shear
    homography_mats[3, 1, 0] = w_shear
    homography_mats[4, 0, 0] = w_scale
    homography_mats[5, 1, 1] = w_scale
    homography_mats[6, 2, 0] = w_perspective
    homography_mats[7, 2, 1] = w_perspective

    # create a grid of coordinates
    grid = torch.stack(
        torch.meshgrid(
            [
                torch.linspace(0.0, 1.0, width, device=weights.device),
                torch.linspace(0.0, 1.0, height, device=weights.device),
            ],
            indexing="ij",
        ),
        dim=-1,
    ).reshape(height * width, 2)

    homogenous_grid = torch.cat([grid, torch.ones_like(grid[..., :1])], dim=-1).reshape(
        1, height * width, 3, 1
    )

    # use broadcasting to apply the homographies to the grid
    vectors = torch.matmul(
        homography_mats.reshape(8, 1, 3, 3), homogenous_grid
    ).squeeze(-1)

    # calculate the flow fields from the homogeneous coordinates
    flow_fields = (vectors[..., :2] / vectors[..., 2:]) - grid[None, ...]

    # norm by the max value
    flow_fields = flow_fields / torch.max(torch.abs(flow_fields))

    # perform QR decomposition to ensure the flow fields are orthogonal
    flow_fields_q, _ = torch.linalg.qr(flow_fields.reshape(8, height * width * 2).T)

    # reshape back to the correct shape (B, H, W, 2)
    return flow_fields_q.T.reshape(8, height, width, 2)


class HomographyBasisTransform(nn.Module):
    """
    A module that uses the Lie algebra homography parameterization to
    make a flow field that can be used to warp an image.

    Args:
        w_translation: The weight for the translation basis homography.
        w_shear: The weight for the shear basis homography.
        w_scale: The weight for the scale basis homography.
        w_perspective: The weight for the perspective basis homography.

    Attributes:
        basis_weights: The weights for the basis homographies.
        homography_weights: The weights for the homography parameters.

    Methods:
        build_bases: Builds the basis homographies. This should be called
            before the module is used.

    Returns:
        A flow field of shape (1, H, W, 2) which can be used to warp an image.

    """
    def __init__(
        self,
        w_translation: float = 0.05,
        w_shear: float = 0.2,
        w_scale: float = 0.5,
        w_perspective: float = 0.001,
    ):
        super().__init__()

        homography_weights = torch.tensor(
            [w_translation, w_shear, w_scale, w_perspective], dtype=torch.float32
        )
        self.register_buffer("homography_weights", homography_weights)

        self.basis_weights = nn.Parameter(torch.zeros(8, dtype=torch.float32)).reshape(
            8, 1, 1, 1
        )

    def build_bases(self, height: int, width: int) -> None:
        self.flow_bases = gen_flow_bases(height, width, self.homography_weights)

    def forward(self, coordinates: Tensor) -> Tensor:
        # Apply the basis weights to the flow bases
        weighted_bases = self.flow_bases * self.basis_weights

        # Use the flow field to move the coordinates
        return coordinates + weighted_bases.sum(dim=0)
    