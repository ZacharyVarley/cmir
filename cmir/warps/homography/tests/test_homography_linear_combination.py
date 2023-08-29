"""
Tests for the HomographyBasisTransform transform.
"""

import pytest
import torch
from cmir.warps.homography.homography_basis import HomographyBasisTransform


@pytest.fixture
def homography_linear_combination_instance():
    # Create a model instance
    image_shape = (16, 3, 32, 32)
    model = HomographyBasisTransform(image_shape)
    return model

def test_output_shape(homography_linear_combination_instance):
    model = homography_linear_combination_instance
    # Test that the output shape is correct
    coordinates = torch.rand(16, 32, 32, 2)
    output = model(coordinates)
    assert output.shape == (16, 32, 32, 2)

def test_basis_weights_length(homography_linear_combination_instance):
    model = homography_linear_combination_instance
    # Test that the number of basis weights is 
    assert model.basis_weights.shape == (16, 8, 1, 1, 1)

def test_homography_weights_length(homography_linear_combination_instance):
    model = homography_linear_combination_instance
    # Test that the number of homography_weights is 4
    assert model.flow_bases.shape == (1, 8, 32, 32, 2)
