"""

This module contains differentiable homography estimation using Lie algebra vectors. The additive
Lie algebra basis vectors are linearly combined according to the the internal parameter weighted
by the weights parameter. The resulting matrix exponential is the homography. This module is
meant to be used with a gradient descent optimizer.

"""

import torch
from torch import Tensor
from torch.nn import Module


class LieHomographies(Module):
    def __init__(
        self,
        n_homographies: int,
        dtype_cast_to: torch.dtype = torch.float64,
        dtype_out: torch.dtype = torch.float32,
        x_translation_weight: float = 1.0,
        y_translation_weight: float = 1.0,
        rotation_weight: float = 1.0,
        scale_weight: float = 1.0,
        stretch_weight: float = 1.0,
        shear_weight: float = 1.0,
        x_keystone_weight: float = 1.0,
        y_keystone_weight: float = 1.0,
    ):
        super(LieHomographies, self).__init__()

        self.dtype_out = dtype_out

        weights = torch.zeros((8,), dtype=dtype_cast_to)
        weights[0] = x_translation_weight
        weights[1] = y_translation_weight
        weights[2] = rotation_weight
        weights[3] = scale_weight
        weights[4] = stretch_weight
        weights[5] = shear_weight
        weights[6] = x_keystone_weight
        weights[7] = y_keystone_weight
        self.register_buffer("weights", weights[None, :, None, None])

        elements = torch.zeros((8, 3, 3), dtype=dtype_cast_to)
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
        elements[6, 2, 0] = (
            1  # projective keystone in x (I might have these swapped for x/y)
        )
        elements[7, 2, 1] = (
            1  # projective keystone in y (I might have these swapped for x/y)
        )
        self.register_buffer("elements", elements)

        self.lie_vectors = torch.nn.Parameter(
            torch.zeros((n_homographies, weights.shape[0], 1, 1), dtype=dtype_cast_to)
        )

    def forward(self) -> Tensor:
        """
        Convert a batch of Lie algebra vectors to Lie group elements (homographies).

        Returns:
            The homographies shape (B, 3, 3).

        """
        # lie_vectors is shape (B, 8, 1, 1)
        # elements is shape (8, 3, 3)
        # weights is shape (1, 8, 1, 1)
        homographies = torch.linalg.matrix_exp(
            (self.lie_vectors * self.elements * self.weights).sum(dim=1).double()
        ).float()
        # make sure the homographies are normalized (bottom right element is 1.0)
        homographies = homographies / homographies[:, 2:3, 2:3]
        return homographies.to(self.dtype_out)

    def half_forward_backward(self) -> Tensor:
        """
        Convert a batch of Lie algebra vectors to Lie group elements
        (homographies). The lie vectors are divided by 2 and the forward and
        backward half homographies are returned.

        Returns:
            The forward / backward half homographies ((B, 3, 3), (B, 3, 3)).

        """
        # lie_vectors is shape (B, 8)
        # elements is shape (8, 3, 3)
        # weights is shape (1, 8, 1, 1)
        lvec_half = self.lie_vectors / 2.0
        forward_half_H = torch.linalg.matrix_exp(
            (lvec_half[:, :, None, None] * self.elements * self.weights).sum(dim=1)
        )
        backward_half_H = torch.linalg.matrix_exp(
            (-lvec_half[:, :, None, None] * self.elements * self.weights).sum(dim=1)
        )
        # make sure the homographies are normalized (bottom right element is 1.0)
        forward_half_H = forward_half_H / forward_half_H[:, 2:3, 2:3]
        backward_half_H = backward_half_H / backward_half_H[:, 2:3, 2:3]
        return forward_half_H.to(self.dtype_out), backward_half_H.to(self.dtype_out)


class LieHomographyLayer(Module):
    """

    This is mean to be used as a transformation function, without storing the lie vectors. This
    is useful for derivative-less optimizers like CMA-ES.

    """

    def __init__(
        self,
        dtype_cast_to: torch.dtype = torch.float64,
        dtype_out: torch.dtype = torch.float32,
        x_translation_weight: float = 1.0,
        y_translation_weight: float = 1.0,
        rotation_weight: float = 1.0,
        scale_weight: float = 1.0,
        stretch_weight: float = 1.0,
        shear_weight: float = 1.0,
        x_keystone_weight: float = 1.0,
        y_keystone_weight: float = 1.0,
    ):
        super(LieHomographyLayer, self).__init__()

        self.dtype_cast_to = dtype_cast_to
        self.dtype_out = dtype_out

        weights = torch.zeros((8,), dtype=dtype_cast_to)
        weights[0] = x_translation_weight
        weights[1] = y_translation_weight
        weights[2] = rotation_weight
        weights[3] = scale_weight
        weights[4] = stretch_weight
        weights[5] = shear_weight
        weights[6] = x_keystone_weight
        weights[7] = y_keystone_weight
        self.register_buffer("weights", weights[None, :, None, None])

        elements = torch.zeros((8, 3, 3), dtype=dtype_cast_to)
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
        elements[6, 2, 0] = (
            1  # projective keystone in x (I might have these swapped for x/y)
        )
        elements[7, 2, 1] = (
            1  # projective keystone in y (I might have these swapped for x/y)
        )
        self.register_buffer("elements", elements)

    def forward(self, lie_vectors) -> Tensor:
        """
        Convert a batch of Lie algebra vectors to Lie group elements (homographies).

        Returns:
            The homographies shape (B, 3, 3).

        """
        # lie_vectors is shape (B, 8)
        # elements is shape (8, 3, 3)
        # weights is shape (1, 8, 1, 1)
        homographies = torch.linalg.matrix_exp(
            (
                lie_vectors[:, :, None, None].to(self.dtype_cast_to)
                * self.elements
                * self.weights
            ).sum(dim=1)
        )
        # make sure the homographies are normalized (bottom right element is 1.0)
        homographies = homographies / homographies[:, 2:3, 2:3]
        return homographies.to(self.dtype_out)
