"""
Tests for the HomographyBasisTransform transform.
"""


import pytest
import torch
from cmir.transforms.homography.homography_basis import HomographyBasisTransform


@pytest.fixture
def homography_linear_combination_instance():
    # Create a model instance
    model = HomographyBasisTransform()
    model.build_bases(32, 32)
    return model

def test_output_shape(homography_linear_combination_instance):
    model = homography_linear_combination_instance
    # Test that the output shape is correct
    output = model(torch.zeros(1, 32, 32, 2))
    assert output.shape == (1, 32, 32, 2)

def test_basis_weights_length(homography_linear_combination_instance):
    model = homography_linear_combination_instance
    # Test that the number of basis weights is 8
    assert model.basis_weights.shape[0] == 8

def test_homography_weights_length(homography_linear_combination_instance):
    model = homography_linear_combination_instance
    # Test that the number of homography_weights is 4
    assert model.homography_weights.shape[0] == 4

