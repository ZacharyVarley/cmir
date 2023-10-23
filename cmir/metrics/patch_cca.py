from typing import Optional
import torch

""" 
This module contains an implementation of CCA carried out on patches of images.
Each corresponding patch in the two images is treated as two collections of vectors
each with dimension equal to the number of channels in the image. The CCA is then
computed between the two collections of vectors. The value reported for the patch is
the trace of the covariance matric over the minimum channel dimension between the 
images. To oversimplify, the mean "r^2" value across correlates is reported. This
module can carry out this computation over several patch sizes and then average those
results.

"""

@torch.jit.script
def batched_cca(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes the batched CCA between two matrices of
    shape (..., n, m) and (..., n, m). n is the number of
    data points and m is the dimension of the data points.
    The CCA is computed batch-wise on the last two dimensions.

    Args:
        x (torch.Tensor): The first input matrix of shape (..., n, m).
        y (torch.Tensor): The second input matrix of shape (..., n, m).
        eps (float, optional): A small value to add to the covariance matrices to
            prevent singular matrices. Defaults to 1e-6.

    Returns:
        torch.Tensor: The batched CCA between the two input matrices.

    Example:
        >>> x = torch.randn(10, 100, 50)
        >>> y = torch.randn(10, 100, 50)
        >>> batched_cca(x, y).shape # (10,)

    The canonical correlation coefficients are the square roots of the eigenvalues of the
    correlation matrix between the two sets of variables. The function returns the mean of 
    the first min(m_dim(x), m_dim(y)) absolute values of the correlations.

    """

    # Standardize the input matrices
    x = (x - x.mean(dim=-2, keepdim=True)) / x.std(dim=-2, keepdim=True)
    y = (y - y.mean(dim=-2, keepdim=True)) / y.std(dim=-2, keepdim=True)

    # Compute covariance matrices
    cov_xx = torch.matmul(x.transpose(-2, -1), x) / (x.shape[-2] - 1)
    cov_yy = torch.matmul(y.transpose(-2, -1), y) / (y.shape[-2] - 1)
    cov_xy = torch.matmul(x.transpose(-2, -1), y) / (x.shape[-2] - 1)

    # Compute the inverse square root of cov_xx and cov_yy
    inv_sqrt_xx = torch.linalg.inv(torch.linalg.cholesky(cov_xx + eps * torch.eye(cov_xx.shape[-1], device=x.device)))
    inv_sqrt_yy = torch.linalg.inv(torch.linalg.cholesky(cov_yy + eps * torch.eye(cov_yy.shape[-1], device=y.device)))

    # Compute the canonical correlation matrix
    corr_matrices = torch.matmul(torch.matmul(inv_sqrt_xx, cov_xy), inv_sqrt_yy)

    # return the trace over the max dimension (min rank)
    return corr_matrices.diagonal(dim1=-2, dim2=-1).abs().sum(dim=-1) / min(
        x.shape[-1], y.shape[-1]
    )


@torch.jit.script
def patch_cca_unfold(
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
    stride: int,
    padding: int,
) -> torch.Tensor:
    """
    Computes the patch-wise CCA between two image tensors
    shaped (B, C, H, W). C dimension is treated as data entries
    within each patch.

    Args:
        x (torch.Tensor): The first input images of shape (B, C, H, W).
        y (torch.Tensor): The second input images of shape (B, C, H, W).
        k (int): The side length of the patches.
        stride (int): The stride of the patches.
        padding (int): The padding of the patches.

    Returns:
        torch.Tensor: CCA correlations (B, n_patches) where n_patches
            depends on the image size, kernel size, stride, and padding.

    Reference:

    Heinrich, Mattias P., et al. "Multispectral image registration based on 
    local canonical correlation analysis." Medical Image Computing and 
    Computer-Assisted Intervention-MICCAI 2014: 17th International Conference, 
    Boston, MA, USA, September 14-18, 2014, Proceedings, Part I 17. Springer 
    International Publishing, 2014.

    """

    B, C1, _, _ = x.shape
    _, C2, _, _ = y.shape

    # Extract patches
    x = torch.nn.functional.unfold(
        x, kernel_size=k, stride=stride, padding=padding
    )
    y = torch.nn.functional.unfold(
        y, kernel_size=k, stride=stride, padding=padding
    )

    # Reshape from (B, C * k * k, N) to (B, C, k * k, N) to (B, N, k * k, C)
    x = x.reshape((B, C1, k * k, -1)).swapdims(-1, -3)
    y = y.reshape((B, C2, k * k, -1)).swapdims(-1, -3)

    # Compute the CCA
    return batched_cca(x, y)


@torch.jit.script
def extract_patches(x: torch.Tensor,
                    coords: torch.Tensor,
                    radius: int
                        ):
    """
    Extract patches centered at each coordinate specified.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W)
        coords (torch.Tensor): Coordinates of shape (B, N, 2) in [0, 1]
        radius (int): patch is shape (2 * radius + 1) x (2 * radius + 1)

    Returns:
        torch.Tensor: Patches of shape (B, N, C, k * k)
    """
    B, C, H, W = x.shape
    _, N, _ = coords.shape

    # Create sampling grid CMIR uses align_corners=True. Consider a 1D example with 2 pixels
    # locations at 0 and 1. When 2x upscaling new samples are at [0, 1/3, 2/3, 1] (align_corners=True).
    # OpenCV and others use [-1/4, 1/4, 3/4, 5/4] (align_corners=False). This means pixel sizes
    # are 1.0 / (Length - 1) so that the last pixel is at 1.0.
    delta_y = 1.0 / (H - 1)
    delta_x = 1.0 / (W - 1)

    rel_coords = torch.stack(torch.meshgrid(torch.linspace(-radius, radius, 2*radius + 1),
                                            torch.linspace(-radius, radius, 2*radius + 1),
                                            indexing='ij'), 
                                            dim=-1).reshape(-1, 2)
    
    rel_coords[..., 0] *= delta_y  # Scale y-coordinates
    rel_coords[..., 1] *= delta_x  # Scale x-coordinates

    # Add relative coordinates to each coordinate and then move to [-1, 1]
    grid = coords[:, :, None, :] + rel_coords[None, None, :, :]  # (B, N, k*k, 2)
    grid = 2.0 * grid - 1.0

    samples = torch.nn.functional.grid_sample(x, grid, align_corners=True)
    samples = samples.view(B, C, N, int((2*radius + 1)*(2*radius+1)))  # (B, C, N, k*k)

    return samples


@torch.jit.script
def patch_cca_sparse(
    x: torch.Tensor,
    y: torch.Tensor,
    coordinates: torch.Tensor,
    patch_radius: int,
) -> torch.Tensor:
    
    # Extract patches
    x = extract_patches(x, coordinates, patch_radius)
    y = extract_patches(y, coordinates, patch_radius)

    # Reshape from (B, C, N, k * k) to (B, N, k * k, C)
    x = x.permute(0, 2, 3, 1)
    y = y.permute(0, 2, 3, 1)

    # Compute the CCA
    return batched_cca(x, y)


@torch.jit.script
def patch_cca_conv(
    x: torch.Tensor,
    y: torch.Tensor,
    patch_radius: int,
) -> torch.Tensor:
    """
    Computes the patch-wise CCA between two image tensors
    shaped (B, C, H, W). C dimension is treated as data entries
    within each patch. This implementation is meant to dense 
    calculations of the CCA metric via signal processing principles.

    Args:
        x (torch.Tensor): The first input images of shape (B, C, H, W).
        y (torch.Tensor): The second input images of shape (B, C, H, W).
        patch_radius (int): The radius of the patches.

    Returns:
        torch.Tensor: CCA correlations (B, 1, H, W)

    Reference:

    Heinrich, Mattias P., et al. "Multispectral image registration based on 
    local canonical correlation analysis." Medical Image Computing and 
    Computer-Assisted Intervention-MICCAI 2014: 17th International Conference, 
    Boston, MA, USA, September 14-18, 2014, Proceedings, Part I 17. Springer 
    International Publishing, 2014.

    """
    # get shapes
    B, C1, H, W = x.shape
    _, C2, _, _ = y.shape

    # reflection pad by patch_radius
    padding = (patch_radius, ) * 4
    x = torch.nn.functional.pad(x, padding, mode='reflect')
    y = torch.nn.functional.pad(y, padding, mode='reflect')

    # xy, xx, yy are shape (B, C1*C2, H, W)
    xy = (x[:, :, None, :, :] * y[:, None, :, :, :]).flatten(1, 2)
    xx = (x[:, :, None, :, :] * x[:, None, :, :, :]).flatten(1, 2)
    yy = (y[:, :, None, :, :] * y[:, None, :, :, :]).flatten(1, 2)

    # compute the local means of outer products
    xx_mean = torch.nn.functional.avg_pool2d(xx, kernel_size=2*patch_radius + 1, stride=1)
    yy_mean = torch.nn.functional.avg_pool2d(yy, kernel_size=2*patch_radius + 1, stride=1)
    xy_mean = torch.nn.functional.avg_pool2d(xy, kernel_size=2*patch_radius + 1, stride=1)

    # compute the local mean values using pooling
    xmean = torch.nn.functional.avg_pool2d(x, kernel_size=2*patch_radius + 1, stride=1)
    ymean = torch.nn.functional.avg_pool2d(y, kernel_size=2*patch_radius + 1, stride=1)

    # computer the outer of local means using pooling
    xm_xm = (xmean[:, :, None, :, :] * xmean[:, None, :, :, :]).flatten(1, 2)
    ym_ym = (ymean[:, :, None, :, :] * ymean[:, None, :, :, :]).flatten(1, 2)
    xm_ym = (xmean[:, :, None, :, :] * ymean[:, None, :, :, :]).flatten(1, 2)

    # compute the local variances and covariances
    # reshape from (B, C1*C2, H, W) to (B, H, W, C1*C2) to (B * H * W, C1, C2)
    # do this in two stages to ensure no accidental transposes etc.
    cxx = (xx_mean - xm_xm).permute(0, 2, 3, 1).reshape(-1, C1, C1)
    cyy = (yy_mean - ym_ym).permute(0, 2, 3, 1).reshape(-1, C2, C2)
    cxy = (xy_mean - xm_ym).permute(0, 2, 3, 1).reshape(-1, C1, C2)

    cxx_inv_sqrt = torch.linalg.inv(torch.linalg.cholesky(cxx + 1e-6 * torch.eye(cxx.shape[-1], device=x.device)))
    cyy_inv_sqrt = torch.linalg.inv(torch.linalg.cholesky(cyy + 1e-6 * torch.eye(cyy.shape[-1], device=y.device)))

    # compute the canonical correlation matrix
    corr_matrices = torch.matmul(torch.matmul(cxx_inv_sqrt, cxy), cyy_inv_sqrt)

    # return the trace over the max dimension (min rank)
    similarity = corr_matrices.diagonal(dim1=-2, dim2=-1).abs().sum(dim=-1) / min(C1, C2)

    return similarity.reshape(B, 1, H, W)
    

class PatchCCADense(torch.nn.Module):
    """
    Computes the patch-wise CCA between two image tensors
    shaped (B, C, H, W). C dimension is treated as data entries
    within each patch. This is the naive version that computes
    each patch separately without any optimization (slow and 
    poor memory performance).

    Args:
        patch_radii (list[int]): Half of one minus the patch side lengths. Each patch
            is a square of size (2 * patch_radius + 1) x (2 * patch_radius + 1).
        strides Optional[list[int]]: The strides of the patches. If None, the
            strides are set to be 1.
        paddings Optional[list[int]]: The paddings of the patches. If None, the
            paddings are set to be half the patch sizes.

    """

    def __init__(
        self,
        patch_radii: list[int],
        strides: Optional[list[int]] = None,
        paddings: Optional[list[int]] = None,
    ):
        super().__init__()
        self.patch_sizes = [2 * radius + 1 for radius in patch_radii]
        self.strides =  [1] * len(self.patch_sizes) if strides is None else strides
        self.paddings = [radius for radius in patch_radii] if paddings is None else paddings

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the patch-wise CCA between two image tensors
        shaped (B, C, H, W). C dimension is treated as data entries
        within each patch.

        Args:
            x (torch.Tensor): The first image of shape (B, C, H, W).
            y (torch.Tensor): The second image of shape (B, C, H, W).


        Returns:
            torch.Tensor: The patch-wise CCA of shape (B,)

        """
        # Compute the patch-wise CCA for all patches using unfold
        cca_values = [
            patch_cca_unfold(x, y, patch_size, stride, padding)
            for patch_size, stride, padding in zip(
                self.patch_sizes, self.strides, self.paddings
            )
        ]

        # Take the list of [(B, N)] tensors and stack them to (B, N, len(patch_radii))
        return torch.stack(cca_values, dim=-1)
    

class PatchCCAConv(torch.nn.Module):
    """
    Computes the patch-wise CCA between two image tensors
    shaped (B, C, H, W). C dimension is treated as data entries
    within each patch. This version is meant for use with 
    only less than 50% (guessing) of the pixels in the image. 
    Otherwise, the dense version makes more sense.

    Args:
        patch_radii (list[int]): The sizes of the patches.

    """

    def __init__(
        self,
        patch_radii: list[int],
    ):
        super().__init__()
        """
        Initializes the PatchCCA module.

        Args:
            patch_radii (list[int]): The sizes of the patches.
        """
        self.patch_radii = patch_radii

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the patch-wise CCA between two image tensors
        shaped (B, C, H, W). C dimension is treated as data entries
        within each patch.

        Args:
            x (torch.Tensor): The first images of shape (B, C, H, W).
            y (torch.Tensor): The second images of shape (B, C, H, W).

        Returns:
            torch.Tensor: The CCA mean correlations over patches of shape (B,)

        """
        # Compute the patch-wise CCA for sparse patches
        cca_values = [
            patch_cca_conv(x, y, patch_radius)
            for patch_radius in self.patch_radii
        ]

        # return shape (B, 1, H, W)
        return torch.stack(cca_values, dim=-1)


class PatchCCASparse(torch.nn.Module):
    """
    Computes the patch-wise CCA between two image tensors
    shaped (B, C, H, W). C dimension is treated as data entries
    within each patch. This version is meant for use with 
    only less than 50% (guessing) of the pixels in the image. 
    Otherwise, the dense version makes more sense. One way to 
    speed up optimization processes is to subsample the pixel
    locations used for metric computation.

    Args:
        patch_radii (list[int]): The sizes of the patches.

    """

    def __init__(
        self,
        patch_radii: list[int],
    ):
        super().__init__()
        """
        Initializes the PatchCCA module.

        Args:
            patch_radii (list[int]): The sizes of the patches.
        """
        self.patch_radii = patch_radii

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the patch-wise CCA between two image tensors
        shaped (B, C, H, W). C dimension is treated as data entries
        within each patch.

        Args:
            x (torch.Tensor): The first images of shape (B, C, H, W).
            y (torch.Tensor): The second images of shape (B, C, H, W).
            coordinates (torch.Tensor): The coordinates of the pixels to use
                in the CCA computation. The coordinates must be in the range
                [0, 1] with shape (B, N, 2) where N is the number of locations

        Returns:
            torch.Tensor: The CCA mean correlations over patches of shape (B,)

        """
        # Compute the patch-wise CCA for sparse patches
        cca_values = [
            patch_cca_sparse(x, y, coordinates, patch_radius)
            for patch_radius in self.patch_radii
        ]

        # Take the list of [(B, N)] tensors and stack them to (B, N, len(patch_radii))
        return torch.stack(cca_values, dim=-1)
