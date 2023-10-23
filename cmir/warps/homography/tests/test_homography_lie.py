import torch
import pytest
from cmir.warps.homography.homography_lie import HomographyLieTransform


@pytest.fixture
def homography_lie_instance():
    # Create a model instance
    model = HomographyLieTransform(algebra='sl3')
    return model


def test_forward(homography_lie_instance):
    model = homography_lie_instance
    # Test that the forward transformation works
    input_coords = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=torch.float32)[None, None]
    output_coords = model(input_coords)
    expected_output_coords = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=torch.float32)[None, None]
    assert torch.allclose(output_coords, expected_output_coords)


def test_inverse(homography_lie_instance):
    model = homography_lie_instance
    # Test that the inverse transformation works
    input_coords = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=torch.float32)[None, None]
    output_coords = model.inverse(input_coords)
    expected_output_coords = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=torch.float32)[None, None]
    assert torch.allclose(output_coords, expected_output_coords)
