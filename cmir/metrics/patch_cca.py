from typing import Tuple
import torch
from torch import Tensor


@torch.jit.script
def batch_cca(
    x: Tensor,
    y: Tensor,
    cca_type: str = "cov",
    std_threshold: float = 0.0001,
    eps: float = 1e-4,
) -> Tensor:
    """
    Computes the batched CCA between two matrices of
    shape (..., n, m1) and (..., n, m2). n is the number of
    data points and m is the dimension of the data points.
    The CCA is computed batch-wise on the last two dimensions.

    Args:
        x (Tensor): The first input matrix of shape (..., n, m1).
        y (Tensor): The second input matrix of shape (..., n, m2).
        cca_type (str):The type of CCA to compute. Options are 'cov' and 'corr'. Defaults to 'cov'.
            The 'cov' option computes the CCA using the covariance matrices. The 'corr' option
            computes the CCA using the correlation matrices.
        std_threshold (float, optional): A threshold on the standard deviation to prevent division
            by zero. Defaults to 0.0001. If the standard deviation is less than this value, the
            standard deviation is set to 1.
        eps (float, optional): A small value to add to the covariance matrices to
            prevent singular matrices. Defaults to 1e-4.

    Returns:
        Tensor: The mean-across channel CCA correlations of shape (...,)


    The canonical correlation coefficients are the square roots of the eigenvalues of the
    correlation matrix between the two sets of variables. The function returns the mean of
    the first min(m_dim(x), m_dim(y)) absolute values of the correlations.

    """

    # Standardize the input matrices
    x = x - x.mean(dim=-2, keepdim=True)
    y = y - y.mean(dim=-2, keepdim=True)

    # if type is correlation, normalize by the standard deviation
    if cca_type == "corr":
        x_std = x.std(dim=-2, keepdim=True)
        y_std = y.std(dim=-2, keepdim=True)
        x = x / torch.where(x_std < std_threshold, torch.ones_like(x_std), x_std)
        y = y / torch.where(y_std < std_threshold, torch.ones_like(y_std), y_std)

    # Compute covariance matrices
    cov_xx = torch.matmul(x.transpose(-2, -1), x) / (x.shape[-2] - 1)
    cov_yy = torch.matmul(y.transpose(-2, -1), y) / (y.shape[-2] - 1)
    cov_xy = torch.matmul(x.transpose(-2, -1), y) / (x.shape[-2] - 1)

    # Compute the inverse square root of cov_xx and cov_yy
    inv_sqrt_xx = torch.linalg.inv(
        torch.linalg.cholesky(
            cov_xx + eps * torch.eye(cov_xx.shape[-1], device=x.device)
        )
    )
    inv_sqrt_yy = torch.linalg.inv(
        torch.linalg.cholesky(
            cov_yy + eps * torch.eye(cov_yy.shape[-1], device=y.device), upper=True
        )
    )

    # Compute the canonical correlation matrix
    cov_matrices = torch.matmul(torch.matmul(inv_sqrt_xx, cov_xy), inv_sqrt_yy)

    # return the trace over the max dimension (min rank)
    return cov_matrices.diagonal(dim1=-2, dim2=-1).abs().mean(dim=-1)


@torch.jit.script
def batch_cca_from_intermediates(
    x_vals_normed: Tensor,
    x_inv_sqrt: Tensor,
    y_vals_normed: Tensor,
    y_inv_sqrt: Tensor,
) -> Tensor:
    """
    Computes the non-dense patch-wise CCA between two image tensors from intermediate
    values computed by patch_offset_grid_cache.

    Args:
        x_vals_normed (Tensor): The first set of patches of shape (..., N, M1)
        x_inv_sqrt (Tensor): The inverse square root of the covariance matrix of x_vals_normed of shape (..., M1, M1)
        y_vals_normed (Tensor): The second set of patches of shape (..., N, M2)
        y_inv_sqrt (Tensor): The inverse square root of the covariance matrix of y_vals_normed of shape (..., M2, M2)

    Returns:
        Tensor: The patch-wise CCA of shape (...)

    """

    # calculate the cross covariance matrix
    cov_xy = torch.matmul(x_vals_normed.transpose(-2, -1), y_vals_normed) / (
        x_vals_normed.shape[-2] - 1
    )

    # compute the canonical correlation matrix
    cov_matrices = torch.matmul(torch.matmul(x_inv_sqrt, cov_xy), y_inv_sqrt)

    # return avg eigenvalue magnitude sum by averaging over the diagonal
    similarity = cov_matrices.diagonal(dim1=-2, dim2=-1).abs().mean(dim=-1)

    return similarity


@torch.jit.script
def batch_cca_to_intermediates(
    x: Tensor,
    upper: bool = False,
    cca_type: str = "cov",
    std_threshold: float = 0.0001,
    eps: float = 1e-4,
) -> Tuple[Tensor, Tensor]:
    """
    Computes the batched CCA between two matrices of
    shape (..., n, m1) and (..., n, m2). n is the number of
    data points and m is the dimension of the data points.
    The CCA is computed batch-wise on the last two dimensions.

    Args:
        x (Tensor): The first input matrix of shape (..., n, m).
        upper (bool): Whether to return the upper triangular matrix. Defaults to False.
        cca_type (str):The type of CCA to compute. Options are 'cov' and 'corr'. Defaults to 'cov'.
            The 'cov' option computes the CCA using the covariance matrices. The 'corr' option
            computes the CCA using the correlation matrices.
        std_threshold (float, optional): A threshold on the standard deviation to prevent division
            by zero. Defaults to 0.0001. If the standard deviation is less than this value, the
            standard deviation is set to 1.
        eps (float, optional): A small value to add to the covariance matrices to
            prevent singular matrices. Defaults to 1e-4.

    Returns:
        Tuple[Tensor, Tensor]: The normalized input matrices and the inverse square root of the
            covariance matrices. The normalized input matrices are of shape (..., n, m) and the
            inverse square root of the covariance matrices are of shape (..., m, m). If upper is
            True, the upper triangular matrices are returned.

    """

    # Standardize the input matrices
    x_norm = x - x.mean(dim=-2, keepdim=True)

    # if type is correlation, normalize by the standard deviation
    if cca_type == "corr":
        x_std = x_norm.std(dim=-2, keepdim=True)
        x_norm = x_norm / torch.where(
            x_std < std_threshold, torch.ones_like(x_std), x_std
        )

    # Compute covariance matrices
    cov_xx = torch.matmul(x_norm.transpose(-2, -1), x_norm) / (x_norm.shape[-2] - 1)

    # Compute the inverse square root of cov_xx
    inv_sqrt_xx = torch.linalg.inv(
        torch.linalg.cholesky(
            cov_xx + eps * torch.eye(cov_xx.shape[-1], device=x.device), upper=upper
        )
    )

    # x_norm is shape (..., n, m) and inv_sqrt_xx is shape (..., m, m)
    return x_norm, inv_sqrt_xx


@torch.jit.script
def image_cca_convolution(
    image1: Tensor, image2: Tensor, patch_radius: int, eps: float = 1e-4
) -> Tensor:
    """
    Densely computes the patch-wise CCA between two image tensors
    shaped (B, C, H, W) via signal processing principles. Operations
    between patches are combined. For larger patches, this is much
    faster than the unfold version, and it is also more memory
    efficient. C dimension is treated as data entries within each patch.

    **Only covariance CCA is supported.**

    Args:
        image1 (Tensor): The first image of shape (B, C, H, W).
        image2 (Tensor): The second image of shape (B, C, H, W).
        patch_radius (int): The radius of the patches.
        eps (float, optional): A small value to add to the covariance matrices to
            prevent singular matrices. Defaults to 1e-4.

    Returns:
        Tensor: The patch-wise CCA values of shape (B, 1, H, W)

    Reference:

    Heinrich, Mattias P., et al. "Multispectral image registration based on
    local canonical correlation analysis." Medical Image Computing and
    Computer-Assisted Intervention-MICCAI 2014: 17th International Conference,
    Boston, MA, USA, September 14-18, 2014, Proceedings, Part I 17. Springer
    International Publishing, 2014.

    """
    # get shapes
    B, C1, H, W = image1.shape
    _, C2, _, _ = image2.shape

    # reflection pad by patch_radius
    padding = (patch_radius, patch_radius, patch_radius, patch_radius)
    image1 = torch.nn.functional.pad(image1, padding, mode="reflect")
    image2 = torch.nn.functional.pad(image2, padding, mode="reflect")

    # xy, xx, yy are shape (B, C1*C2, H, W)
    xy = (image1[:, :, None, :, :] * image2[:, None, :, :, :]).flatten(1, 2)
    xx = (image1[:, :, None, :, :] * image1[:, None, :, :, :]).flatten(1, 2)
    yy = (image2[:, :, None, :, :] * image2[:, None, :, :, :]).flatten(1, 2)

    # compute the local means of outer products
    xx_mean = torch.nn.functional.avg_pool2d(
        xx, kernel_size=2 * patch_radius + 1, stride=1
    )
    yy_mean = torch.nn.functional.avg_pool2d(
        yy, kernel_size=2 * patch_radius + 1, stride=1
    )
    xy_mean = torch.nn.functional.avg_pool2d(
        xy, kernel_size=2 * patch_radius + 1, stride=1
    )

    # compute the local mean values using pooling
    xmean = torch.nn.functional.avg_pool2d(
        image1, kernel_size=2 * patch_radius + 1, stride=1
    )
    ymean = torch.nn.functional.avg_pool2d(
        image2, kernel_size=2 * patch_radius + 1, stride=1
    )

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

    cxx_inv_sqrt = torch.linalg.inv(
        torch.linalg.cholesky(
            cxx + eps * torch.eye(cxx.shape[-1], device=image1.device)
        )
    )
    cyy_inv_sqrt = torch.linalg.inv(
        torch.linalg.cholesky(
            cyy + eps * torch.eye(cyy.shape[-1], device=image2.device),
            upper=True,
        )
    )

    # compute the canonical correlation matrix and similarities
    corr_matrices = torch.matmul(torch.matmul(cxx_inv_sqrt, cxy), cyy_inv_sqrt)
    similarity = corr_matrices.diagonal(dim1=-2, dim2=-1).abs().mean(dim=-1)

    return similarity.reshape(B, 1, H, W)
