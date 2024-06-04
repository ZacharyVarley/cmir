"""
Tests for the PatchCCA module.
"""

import torch
import pytest

from cmir.metrics.local_cca.patch_cca import (
    PatchCCASparse,
    PatchCCADense,
    extract_patches,
    patch_cca_unfold,
    patch_cca_sparse,
)


@pytest.fixture
def patchccasparse_instance():
    # Create a model instance
    model = PatchCCASparse([1, 2, 3])
    return model


@pytest.fixture
def patchccadense_instance():
    # Create a model instance
    model = PatchCCADense([1, 2, 3])
    return model


def test_patchccasparse(patchccasparse_instance):
    # Create a random input
    x = torch.randn(2, 5, 64, 64)
    y = (x[:, :4, :, :] + 0.01 * torch.randn(2, 4, 64, 64)) / 1.01
    coords = torch.rand(2, 100, 2)
    cca_out = patchccasparse_instance(x, y, coords)
    assert cca_out.shape == (2, 100, 3)


def test_patchccasparse_zeros(patchccasparse_instance):
    # Create a random input
    x = torch.randn(2, 5, 64, 64) * 0.0001
    y = (x[:, :4, :, :] + 0.0001 * torch.randn(2, 4, 64, 64)) / 1.0001
    coords = torch.rand(2, 100, 2)
    cca_out = patchccasparse_instance(x, y, coords)
    assert cca_out.shape == (2, 100, 3)


def test_patchccadense(patchccadense_instance):
    # Create a random input
    x = torch.randn(2, 5, 32, 32)
    y = (x[:, :4, :, :] + 0.01 * torch.randn(2, 4, 32, 32)) / 1.01
    cca_out = patchccadense_instance(x, y)
    assert cca_out.shape == (2, 1024, 3)


def test_extract_patches():
    x = torch.arange(1, 122, 1).reshape(1, 1, 11, 11).float()
    coords = torch.tensor([[[0.1, 0.1]]])
    patch_radius = 1
    patches = extract_patches(x, coords, patch_radius)
    assert patches.shape == (1, 1, 1, 9)
    print(patches)
    assert torch.allclose(
        patches, torch.tensor([[1.0, 12.0, 23.0, 2.0, 13.0, 24.0, 3.0, 14.0, 25.0]])
    )
