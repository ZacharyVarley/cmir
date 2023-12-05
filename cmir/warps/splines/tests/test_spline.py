"""
Tests for the BSplineWarp class.
"""

import torch
import pytest

from cmir.warps.splines.bspline import BSplineWarp


@pytest.fixture
def bspline_instance():
    # Create a model instance
    model = BSplineWarp(
        n_transforms=5,
        image_shape=(128, 128),
        control_shape=(16, 16),
        bound="zeros",
        interpolation="cubic")
    return model

def test_control_shape(bspline_instance):
    model = bspline_instance
    # Test that the control shape is correct
    assert model.control_shape == (16, 16)

def test_bound(bspline_instance):
    model = bspline_instance
    # Test that the bound is correct
    assert model.bound == "zeros"    

def test_output_shape(bspline_instance):
    model = bspline_instance
    # Test that the output shape is correct
    output = model(torch.zeros(5, 128, 128, 2))
    assert output.shape == (5, 128, 128, 2)

def test_inverse_shape(bspline_instance):
    model = bspline_instance
    # Test that the output shape is correct
    output = model.inverse(torch.zeros(5, 128, 128, 2))
    assert output.shape == (5, 128, 128, 2)

def test_interpolation(bspline_instance):
    model = bspline_instance
    # Test that the displacements are all initialized to zero
    assert torch.allclose(model.displacements, torch.zeros(2, 16, 16))

def test_adjoint_displacements(bspline_instance):
    model = bspline_instance
    # remove grad to allow in-place modification of displacements
    model.requires_grad_(False)
    # set seed
    torch.manual_seed(0)
    # draw from Gaussian distribution for displacements
    model.displacements += torch.randn_like(model.displacements) * 4.0
    # test that the inverse of the forward is the identity
    # make a grid of image coordinates unperturbed
    image_coordinates = torch.stack(torch.meshgrid(torch.arange(128),
                                                   torch.arange(128),
                                                   indexing='ij'), 
                                                   dim=-1)
    image_coordinates = image_coordinates.float()
    # warp the image coordinates
    warped_coordinates = model(image_coordinates)
    # unwarp the warped coordinates
    unwarped_coordinates = model.inverse(warped_coordinates)
    assert torch.allclose(image_coordinates, unwarped_coordinates, atol=1e-6)