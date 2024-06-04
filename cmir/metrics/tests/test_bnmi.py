"""
Tests for the PatchCCA module.
"""

import torch
import pytest
import cmir
from cmir.metrics.bcmi import batch_binned_nmi, binned_nmi, bnmi_map_2D


@pytest.fixture
def patchccasparse_instance():
    # Create a model instance
    model = cmir.metrics.mi.bnm
    return model
