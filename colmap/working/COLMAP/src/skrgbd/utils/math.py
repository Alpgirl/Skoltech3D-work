import numpy as np
import roma
import torch


def make_extrinsics(R, t):
    r"""Make the extrinsics matrix from a rotation matrix and a translation vector.
    
    Parameters
    ----------
    R : torch.Tensor
        of shape [batch_size, 3, 3]
    t : torch.Tensor
        of shape [batch_size, 3]
    
    Returns
    -------
    extrinsics : torch.Tensor
        of shape [batch_size, 4, 4]
    """
    batch_size = len(R)
    extrinsics = R.new_zeros(batch_size, 4, 4)
    extrinsics[:, 3, 3] = 1
    extrinsics[:, :3, :3] = R
    extrinsics[:, :3, 3] = t
    return extrinsics


def mat2trotq(mat):
    r"""Converts 4x4 transform matrix to translation vector and rotation quaternion.

    Parameters
    ----------
    mat: torch.Tensor
        of shape [..., 4, 4], transformation matrix.

    Returns
    -------
    t : torch.Tensor
        of shape [..., 3], translation vector.
    rotq : torch.Tensor
        of shape [..., 4], rotation quaternion in xyzw notation.
    """
    t = mat[..., :3, 3]
    rotm = mat[..., :3, :3]
    rotq = roma.rotmat_to_unitquat(rotm); del rotm
    return t, rotq


def sphere_arange_phi(r, phi_min, phi_max, step, flip=False, eps=1e-7):
    r"""For a given sphere radius r generates the values of the polar angle in the range [phi_min, phi_max] so that
    the respective points are placed on the sphere with the given step.

    Parameters
    ----------
    phi_min : float
    phi_max : float
    step : float
    flip : bool
        If True, start the values from phi_max.
    eps : float
        Small value added to the right end of the range to include the end point.

    Returns
    -------
    phi : torch.Tensor
        of shape [values_n]
    """
    phi_step = np.arccos(1 - .5 * (step / r) ** 2)
    if not flip:
        phi = torch.arange(phi_min, phi_max + eps, phi_step)
    else:
        phi = torch.arange(phi_max, phi_min - eps, -phi_step)
    return phi


def sphere_arange_theta(r, phi, theta_min, theta_max, step, flip=False, eps=1e-7):
    r"""For a given polar angle phi and a sphere radius r generates the values of the azimuth in the range
    [theta_min, theta_max] so that the respective points are placed on the sphere with the given step.

    Parameters
    ----------
    r : float
    phi : float
    theta_min : float
    theta_max : float
    step : float
    flip : bool
        If True, start the values from theta_max.
    eps : float
        Small value added to the right end of the range to include the end point.

    Returns
    -------
    theta : torch.Tensor
        of shape [values_n]
    """
    theta_step = np.arccos(1 - .5 * (step / (r * np.sin(phi))) ** 2)
    if not flip:
        theta = torch.arange(theta_min, theta_max + eps, theta_step)
    else:
        theta = torch.arange(theta_max, theta_min - eps, -theta_step)
    return theta


def spherical_to_cartesian(rho_theta_phi):
    r"""Converts spherical coordinates to cartesian.

    Parameters
    ----------
    rho_theta_phi : torch.Tensor
        of shape [**, 3].

    Returns
    -------
    xyz : torch.Tensor
        of shape [**, 3]
    """
    rho = rho_theta_phi[..., 0]
    theta = rho_theta_phi[..., 1]
    phi = rho_theta_phi[..., 2]
    x = rho * theta.cos() * phi.sin()
    y = rho * theta.sin() * phi.sin()
    z = rho * phi.cos()
    return torch.stack([x, y, z], -1)
