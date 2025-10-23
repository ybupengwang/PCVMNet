
import torch
import torch.nn as nn
import numpy as np
from typing import Callable, List, Optional, Tuple
from scipy import optimize
def coord_cov_from_gaussian_ls(
    heatmap: torch.Tensor,
    gamma: Optional[float] = None,
    ls_library: str = "scipy",
    spatial_dims: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the modal coordinates and covariance matrix from a heatmap through fitting the heatmap
    on Gaussian distribution with a specicic scaling factor gamma with help of least squares
    optimization.

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W)
        gamma (float): gamma parameter of the gaussian heatmap generator
        ls_library (str): library to use for least squares optimization. (scipy or pytorch)
        spatial_dims (int): number of spatial dimensions (2 or 3)

    Returns:
        (torch.Tensor): coordinates of shape (B, C, 2)
        (torch.Tensor): covariance matrix of shape (B, C, 2, 2)
    """
    # TODO: see if we can use a pytorch implementation (e.g, pytorch-minimize seems to be broken)
    if spatial_dims != 2:
        raise ValueError(f"Spatial dimensions must be 2: {spatial_dims}")
    if ls_library == "scipy":
        return coord_cov_from_gaussian_ls_scipy(heatmap, gamma=gamma)
    raise ValueError("Method not implemented.")

def coord_cov_from_gaussian_ls_scipy(
    heatmap: torch.Tensor, gamma: Optional[float] = None, spatial_dims: int = 2
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the modal coordinates and covariance matrix from a heatmap through fitting the heatmap
    on Gaussian distribution with a specicic scaling factor gamma with help of least squares
    optimization.

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W)
        gamma (float): gamma parameter of the gaussian heatmap generator
        spatial_dims (int): number of spatial dimensions (2 or 3)

    Returns:
        (torch.Tensor): coordinates of shape (B, C, 2)
        (torch.Tensor): covariance matrix of shape (B, C, 2, 2)
    """

    def generate_gaussian(heatmap: torch.Tensor) -> Callable:
        """
        Returns the gaussian function for the given landmarks, sigma and rotation.
        """
        gaussian_generator = GaussianHeatmapGenerator(
            1,
            gamma=gamma,
            heatmap_size=(heatmap.shape[-2], heatmap.shape[-1]),
            learnable=False,
        ).to(heatmap.device)

        def fun_to_minimize(x):
            gaussian_generator.set_sigmas(x[:2])
            gaussian_generator.set_rotation(x[2])
            return (
                (
                    heatmap
                    - gaussian_generator(torch.Tensor(x[3:]).view((1, 1, 2))).view(heatmap.shape)
                )
                .flatten()
                .detach()
                .cpu()
                .numpy()
            )

        return fun_to_minimize

    if spatial_dims != 2:
        raise ValueError(f"Spatial dimensions must be 2: {spatial_dims}")
    b, c, _, _ = heatmap.shape
    coords = torch.zeros((b, c, 2)).to(heatmap.device)
    covs = torch.zeros((b, c, 2, 2)).to(heatmap.device)
    for b1 in range(b):
        for c1 in range(c):
            init_coord = coord_argmax(heatmap[b1, c1].unsqueeze(0).unsqueeze(0))[0, 0]
            result = optimize.least_squares(
                generate_gaussian(heatmap[b1, c1]),
                np.array([1, 1, 0, init_coord[0].item(), init_coord[1].item()]),
                method="trf",
            )
            x = result.x
            coords[b1, c1] = torch.tensor([x[3], x[4]], dtype=torch.float)
            rotation = torch.tensor(
                [[np.cos(x[2]), -np.sin(x[2])], [np.sin(x[2]), np.cos(x[2])]], dtype=torch.float
            )
            diagonal = torch.diag(torch.tensor(x[:2] ** 2, dtype=torch.float))
            covs[b1, c1] = torch.mm(torch.mm(rotation, diagonal), rotation.t())
    return coords, covs
