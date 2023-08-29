"""
Tests for levels transforms.
"""

import pytest
import torch
from cmir.transforms.levels import HardLevels, LinearLevels, GaussianLevels, GumbelLevels


@pytest.fixture
def hard_levels_instance():
    # Create a model instance
    model = HardLevels(n_levels=8)
    return model


# @pytest.fixture
# def linear_levels_instance():
#     # Create a model instance
#     model = LinearLevels(n_levels=8)
#     return model


# @pytest.fixture
# def gaussian_levels_instance():
#     # Create a model instance
#     model = GaussianLevels(n_levels=8)
#     return model


# @pytest.fixture
# def gumbel_levels_instance():
#     # Create a model instance
#     model = GumbelLevels(n_levels=8)
#     return model


def test_hard_levels_output_shape(hard_levels_instance):
    model = hard_levels_instance
    # Test that the output shape is correct
    input = torch.rand(16, 3, 32, 32)
    output = model(input)
    assert output.shape == (16, 24, 32, 32)


def test_hard_levels_output_dtype(hard_levels_instance):
    model = hard_levels_instance
    # Test that the output dtype is correct
    input = torch.rand(16, 3, 32, 32)
    output = model(input)
    assert output.dtype == torch.uint8


# def test_linear_levels_output_shape(linear_levels_instance):
#     model = linear_levels_instance
#     # Test that the output shape is correct
#     input = torch.rand(16, 3, 32, 32)
#     output = model(input)
#     assert output.shape == (16, 24, 32, 32)


# def test_linear_levels_output_dtype(linear_levels_instance):
#     model = linear_levels_instance
#     # Test that the output dtype is correct
#     input = torch.rand(16, 3, 32, 32)
#     output = model(input)
#     assert output.dtype == torch.float32


# def test_gaussian_levels_output_shape(gaussian_levels_instance):
#     model = gaussian_levels_instance
#     # Test that the output shape is correct
#     input = torch.rand(16, 3, 32, 32)
#     output = model(input)
#     assert output.shape == (16, 24, 32, 32)


# def test_gaussian_levels_output_dtype(gaussian_levels_instance):
#     model = gaussian_levels_instance
#     # Test that the output dtype is correct
#     input = torch.rand(16, 3, 32, 32)
#     output = model(input)
#     assert output.dtype == torch.float32


# def test_gumbel_levels_output_shape(gumbel_levels_instance):
#     model = gumbel_levels_instance
#     # Test that the output shape is correct
#     input = torch.rand(16, 3, 32, 32)
#     output = model(input)
#     assert output.shape == (16, 24, 32, 32)


# def test_gumbel_levels_output_dtype(gumbel_levels_instance):
#     model = gumbel_levels_instance
#     # Test that the output dtype is correct
#     input = torch.rand(16, 3, 32, 32)
#     output = model(input)
#     assert output.dtype == torch.float32
