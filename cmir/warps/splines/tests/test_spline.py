"""
Tests for the BSplineWarp class.
"""

import torch
import pytest

from cmir.warps.splines.bspline import BSplineWarp


@pytest.fixture
def bspline_instance():
    # Create a model instance
    model = BSplineWarp(image_shape=(32, 32), 
                             control_shape=(6, 6),
                             bound="dct2",
                             interpolation="cubic")
    return model


def test_control_shape(bspline_instance):
    model = bspline_instance
    # Test that the control shape is correct
    assert model.control_shape == (6, 6)


def test_bound(bspline_instance):
    model = bspline_instance
    # Test that the bound is correct
    assert model.bound == "dct2"    


def test_output_shape(bspline_instance):
    model = bspline_instance
    # Test that the output shape is correct
    output = model(torch.zeros(5, 32, 32, 2))
    assert output.shape == (5, 32, 32, 2)


def test_interpolation(bspline_instance):
    model = bspline_instance
    # Test that the displacements are all initialized to zero
    assert torch.allclose(model.displacements, torch.zeros(2, 6, 6))
