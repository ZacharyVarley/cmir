"""
Tests for levels transforms.
"""

import pytest
import torch
from cmir.transforms.levels import (
    HardLevels,
    LinearLevels,
    GaussianLevels,
    GumbelLevels,
)


@pytest.fixture
def hard_levels_instance():
    # Create a model instance
    model = HardLevels(n_levels=4)
    return model


@pytest.fixture
def linear_levels_instance():
    # Create a model instance
    model = LinearLevels(n_levels=4)
    return model


@pytest.fixture
def gaussian_levels_instance():
    # Create a model instance
    model = GaussianLevels(n_levels=4)
    return model


@pytest.fixture
def gumbel_levels_instance():
    # Create a model instance
    model = GumbelLevels(n_levels=4)
    return model


def test_hard_levels_output_shape(hard_levels_instance):
    model = hard_levels_instance
    # Test that the output shape is correct
    input = torch.rand(4, 2, 16, 16)
    output = model(input)
    assert output.shape == (4, 8, 16, 16)


def test_hard_levels_output_dtype(hard_levels_instance):
    model = hard_levels_instance
    # Test that the output dtype is correct
    input = torch.rand(4, 2, 16, 16)
    output = model(input)
    assert output.dtype == torch.uint8


def test_hard_levels_output(hard_levels_instance):
    model = hard_levels_instance
    # Test that the output dtype is correct
    input = torch.tensor([0.1, 0.6]).reshape(1, 2, 1, 1)
    output = model(input)
    ground_truth = torch.tensor([1, 0, 0, 0, 0, 0, 1, 0]).reshape(1, 8, 1, 1)
    assert torch.all(output == ground_truth)


def test_linear_levels_output_shape(linear_levels_instance):
    model = linear_levels_instance
    # Test that the output shape is correct
    input = torch.rand(4, 2, 16, 16)
    output = model(input)
    assert output.shape == (4, 8, 16, 16)


def test_linear_levels_output_dtype(linear_levels_instance):
    model = linear_levels_instance
    # Test that the output dtype is correct
    input = torch.rand(4, 2, 16, 16)
    output = model(input)
    assert output.dtype == torch.float32


def test_linear_levels_output(linear_levels_instance):
    model = linear_levels_instance
    # Test that the output dtype is correct
    input = torch.tensor([0.1, 0.6]).reshape(1, 2, 1, 1)
    output = model(input)
    # centers will be [0.125, 0.375, 0.625, 0.875]
    # diff for 0.1 will be [0.025, 0.275, 0.525, 0.775]
    # 1 - (diff * 4) will be [0.9, -0.1, -1.1, -2.1] -> [0.9, 0, 0, 0]
    # diff for 0.6 will be [0.475, 0.225, 0.025, 0.275]
    # 1 - (diff * 4) will be [-0.9, 0.1, 0.9, -0.1] -> [0, 0.1, 0.9, 0]
    ground_truth = torch.tensor([[1.0, 0, 0, 0], [0, 0.1, 0.9, 0]]).reshape(1, 8, 1, 1)
    assert torch.allclose(output, ground_truth)


def test_gaussian_levels_output_shape(gaussian_levels_instance):
    model = gaussian_levels_instance
    # Test that the output shape is correct
    input = torch.rand(4, 2, 16, 16)
    output = model(input)
    assert output.shape == (4, 8, 16, 16)


def test_gaussian_levels_output_dtype(gaussian_levels_instance):
    model = gaussian_levels_instance
    # Test that the output dtype is correct
    input = torch.rand(4, 2, 16, 16)
    output = model(input)
    assert output.dtype == torch.float32


def test_gumbel_levels_output_shape(gumbel_levels_instance):
    model = gumbel_levels_instance
    # Test that the output shape is correct
    input = torch.rand(4, 2, 16, 16)
    output = model(input)
    assert output.shape == (4, 8, 16, 16)


def test_gumbel_levels_output_dtype(gumbel_levels_instance):
    model = gumbel_levels_instance
    # Test that the output dtype is correct
    input = torch.rand(4, 2, 16, 16)
    output = model(input)
    assert output.dtype == torch.float32
