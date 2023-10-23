"""
This module contains the implementation of the Lie algebra homography.
This transform defines a homography as an additive combination of elements
from a Lie algebra. Matrix exponentiation is used to convert the additive
combination to a homography in the Lie group. The Lie algebra elements are
arbitrarily chosen generators. 

Eade, Ethan. "Lie groups for computer vision." Cambridge Univ., Cam-bridge, 
UK, Tech. Rep 2 (2014).

"""

import torch
import torch.nn as nn
from typing import Optional

__all__ = [
    "HomographyLieTransform",
]

# list of allowed groups
ALLOWED_GROUPS = [
    "so2",  # rotation
    "r2",  # translation
    "se2",  # rotation + translation
    "sim2",  # rotation + translation + scaling
    "as2",  # rotation + translation + scaling + axial stretching
    "aff2",  # affine transformation "as2" + shear
    "sl3",
]  # projective transformation


class LieAlgebraParameterization(nn.Module):
    """
    A module that parameterizes Lie group sl3 using its Lie algebra.

    Args:
        group (str): parameter to select a common subgroup.

        bias (torch.Tensor, optional): A tensor of shape (n,), where n
            is the number of basis elements of the Lie algebra. If given,
            the bias terms are added to the linear combination of the basis
            elements. Defaults to None.

    Attributes:
        elements (torch.Tensor): A tensor of shape (n, 3, 3) that contains
          the basis elements of the Lie algebra.
        weights (nn.Parameter): A parameter tensor of shape (n, 1, 1) that
        contains the weights of the linear combination
            of the basis elements.
        bias (torch.Tensor): A tensor of shape (n, 1, 1) that contains the bias terms.

    Returns:
        A tensor of shape (3, 3) that represents an element of the Lie group.

    Raises:
        NotImplementedError: If the given group is not one of the allowed values.
    """

    def __init__(self, algebra: str, bias: Optional[torch.Tensor] = None):
        super(LieAlgebraParameterization, self).__init__()

        self.algebra = algebra

        if algebra == "so2":
            elements = torch.zeros((1, 3, 3))
            elements[0, 0, 1] = -1  # rotation
            elements[0, 1, 0] = 1  # rotation
        elif algebra == "r2":
            elements = torch.zeros((2, 3, 3))
            elements[0, 0, 2] = 1  # translation in x
            elements[1, 1, 2] = 1  # translation in y
        elif algebra == "se2":
            elements = torch.zeros((3, 3, 3))
            elements[0, 0, 2] = 1  # translation in x
            elements[1, 1, 2] = 1  # translation in y
            elements[2, 0, 1] = -1  # rotation
            elements[2, 1, 0] = 1  # rotation
        elif algebra == "sim2":
            elements = torch.zeros((4, 3, 3))
            elements[0, 0, 2] = 1  # translation in x
            elements[1, 1, 2] = 1  # translation in y
            elements[2, 0, 1] = -1  # rotation
            elements[2, 1, 0] = 1  # rotation
            elements[3, 2, 2] = -1  # isotropic scaling
        elif algebra == "as2":
            # this is the shear-less affine group
            elements = torch.zeros((5, 3, 3))
            elements[0, 0, 2] = 1  # translation in x
            elements[1, 1, 2] = 1  # translation in y
            elements[2, 0, 1] = -1  # rotation
            elements[2, 1, 0] = 1  # rotation
            elements[3, 0, 0] = 1  # isotropic scaling
            elements[3, 1, 1] = 1  # isotropic scaling
            elements[4, 0, 0] = 1  # stretching
            elements[4, 1, 1] = -1  # stretching
        elif algebra == "aff2":
            elements = torch.zeros((6, 3, 3))
            elements[0, 0, 2] = 1  # translation in x
            elements[1, 1, 2] = 1  # translation in y
            elements[2, 0, 1] = -1  # rotation
            elements[2, 1, 0] = 1  # rotation
            elements[3, 0, 0] = 1  # isotropic scaling
            elements[3, 1, 1] = 1  # isotropic scaling
            elements[4, 0, 0] = 1  # stretching
            elements[4, 1, 1] = -1  # stretching
            elements[5, 0, 1] = 1  # shear
            elements[5, 1, 0] = 1  # shear
        elif algebra == "sl3":
            elements = torch.zeros((8, 3, 3))
            elements[0, 0, 2] = 1  # translation in x
            elements[1, 1, 2] = 1  # translation in y
            elements[2, 0, 1] = -1  # rotation
            elements[2, 1, 0] = 1  # rotation
            elements[3, 0, 0] = 1  # isotropic scaling
            elements[3, 1, 1] = 1  # isotropic scaling
            elements[3, 2, 2] = -2  # isotropic scaling
            elements[4, 0, 0] = 1  # stretching
            elements[4, 1, 1] = -1  # stretching
            elements[5, 0, 1] = 1  # shear
            elements[5, 1, 0] = 1  # shear
            elements[
                6, 2, 0
            ] = 1  # projective keystone in x (I might have these swapped for x/y)
            elements[
                7, 2, 1
            ] = 1  # projective keystone in y (I might have these swapped for x/y)
        else:
            raise NotImplementedError(
                f"Group {algebra} not implemented. Allowed groups are {ALLOWED_GROUPS}"
            )

        # set elements buffer (without grad attribute)
        self.register_buffer("elements", elements)

        # set the parameters (the linear combination the lie algebra elements)
        self.weights = nn.Parameter(torch.zeros(len(elements), 1, 1))

        if bias is not None:
            # if bias are given, check if they are of the right shape
            assert len(bias.shape) == 1
            assert bias.shape[0] == len(
                self.elements
            ), f"Number of bias terms ({bias.shape[0]}) does not match number of elements ({len(self.elements)})"
            # set the bias buffer (no grad attribute)
            self.register_buffer("bias", bias[:, None, None])
        else:
            # if no bias are given, set them to ones
            self.register_buffer("bias", torch.ones(len(elements), 1, 1))

    def __call__(self):
        # multiply the basis components with the bias tensor and sum them up and take the matrix exponential
        return torch.linalg.matrix_exp(
            (self.weights * self.elements * self.bias).sum(dim=0)
        )


@torch.jit.script
def apply_homography(
    coordinates: torch.Tensor,
    homography: torch.Tensor,
):
    """
    Apply a homography to an image.

    Args:
        coordinates (torch.Tensor): A tensor of shape (b, h, w, 2) that
            represents the coordinates of the image.

        homography (torch.Tensor): A tensor of shape (b, 3, 3) that represents
            each of the homographies to apply to the images.

        align_corners (bool, optional): Whether to align the corners of the image.
            Defaults to False.
        
    Returns:
        A tensor of shape (b, c, h, w) that represents the warped image.
    """

    # get the shape of the coordinates
    b, h, w, _ = coordinates.shape

    # reshape the coordinates to (b, h*w, 2)
    coordinates = coordinates.reshape(b, h * w, 2)

    # add a ones dimension to the coordinates
    coordinates = torch.cat([coordinates, torch.ones_like(coordinates[..., :1])], dim=-1)

    # use broadcasting to apply the homographies to the coordinates
    coordinates = torch.matmul(homography, coordinates.unsqueeze(-1)).squeeze(-1)

    # normalize the coordinates
    coordinates = coordinates[..., :2] / coordinates[..., 2:]

    # reshape the coordinates back to (b, h, w, 2)
    coordinates = coordinates.reshape(b, h, w, 2)

    # warp the image
    return coordinates


class HomographyLieTransform(nn.Module):
    """
    A module that uses the Lie algebra homography parameterization to
    deform input coordinates.

    Args:
        group (str): parameter to select a common subgroup.

        bias (torch.Tensor, optional): A tensor of shape (n,), where n
            is the number of basis elements of the Lie algebra. If given,
            the bias terms are added to the linear combination of the basis
            elements. Defaults to None.

    Attributes:
        elements (torch.Tensor): A tensor of shape (n, 3, 3) that contains
          the basis elements of the Lie algebra.
        weights (nn.Parameter): A parameter tensor of shape (n, 1, 1) that
        contains the weights of the linear combination
            of the basis elements.
        bias (torch.Tensor): A tensor of shape (n, 1, 1) that contains the bias terms.

    Returns:
        A tensor of shape (3, 3) that represents an element of the Lie group.

    Raises:
        NotImplementedError: If the given group is not one of the allowed values.
    """

    def __init__(self, algebra: str, bias: Optional[torch.Tensor] = None):
        super(HomographyLieTransform, self).__init__()
        # create the lie algebra parameterization
        self.lie_algebra = LieAlgebraParameterization(algebra, bias)

    def __call__(self, image_coordinates: torch.Tensor):
        # get the homography
        homography = self.lie_algebra()

        # apply the homography
        return apply_homography(image_coordinates, homography)
    
    def inverse(self, image_coordinates: torch.Tensor):
        # get the homography
        homography = self.lie_algebra()
        
        # invert the homography
        homography = torch.inverse(homography)
        
        # apply the homography
        return apply_homography(image_coordinates, homography)
