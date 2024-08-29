from collections import OrderedDict

import torch


class CubicBSpline(torch.nn.Module):
    r"""Implements interpolation with multivariate cardinal cubic B-spline.

    Parameters
    ----------
    min_pt : torch.Tensor
        of shape [dims_n].
    max_pt : torch.Tensor
        of shape [dims_n].
    step : torch.Tensor
        of shape [dims_n].
    channels_n : int
        Dimensionality of the interpolated function.
    eps : float

    Attributes
    ----------
    origin : torch.Tensor
        of shape [dims_n], origin of the grid of the control points.
    step : torch.Tensor
        of shape [dims_n], step of the grid of the control points.
    max_interp_grid_coord : torch.Tensor
        of shape [dims_n], max interpolated coordinate in control grid coordinates.
    channels_n : int
        Dimensionality of the interpolated function.
    control_pts : torch.Tensor
        of shape [s1, ..., s_n, channels_n], coordinates of the control points.
    """
    def __init__(self, min_pt, max_pt, step, channels_n=0, eps=1e-8):
        super().__init__()
        origin = min_pt - step * (1 + eps)
        control_grid_size = (max_pt - origin).div(step).add(2 + eps).ceil().long()
        max_interp_grid_coord = control_grid_size - 2
        self.register_buffer('origin', origin)
        self.register_buffer('step', step)
        self.register_buffer('max_interp_grid_coord', max_interp_grid_coord)
        control_grid_size = control_grid_size.tolist()
        self.channels_n = channels_n
        if channels_n != 0:
            control_grid_size = [channels_n] + control_grid_size
        self.control_pts = torch.nn.Parameter(torch.ones(control_grid_size), requires_grad=False)

    @classmethod
    def from_state_dict(cls, state_dict):
        r"""Makes an empty CubicBSpline to load_state_dict to.

        Parameters
        ----------
        state_dict : OrderedDict

        Returns
        -------
        spline : CubicBSpline
        """
        dims_n = len(state_dict['origin'])
        control_grid_size = torch.tensor(state_dict['control_pts'].shape)
        if (len(control_grid_size) - dims_n) != 0:
            channels_n = control_grid_size[0]
            control_grid_size = control_grid_size[1:]
        else:
            channels_n = 0
        eps = 0
        step = torch.ones_like(state_dict['step'])
        min_pt = torch.ones_like(step)
        max_pt = control_grid_size - 2
        spline = cls(min_pt, max_pt, step, channels_n, eps)
        return spline

    def forward(self, pts, padding_value=0):
        r"""Interpolates the spline at pts.

        Parameters
        ----------
        pts : torch.Tensor
            of shape [**, dims_n].
        padding_value : float

        Returns
        -------
        vals : torch.Tensor
            of shape [**], or [**, channels_n] if channels_n != 0.

        Notes
        -----
        An input point is interpolated using the four closest control points in each dimension around it,
        as illustrated below
           A   B   C   D
        0  o---o---o---o
           |   |   |   |
        1  o---o---o---o
           |   | * |   |
        2  o---o---o---o
           |   |   |   |
        3  o---o---o---o

        Each B-spline is defined on the region four-by-four-etc around its control point, as illustrated below
           0   1   2   3   4
        0  .---.---.---.---.
           |   |   |   |   |
        1  .---.---.---.---.
           |   |   |   |   |
        2  .---.---o---.---.
           |   |   |   |   |
        3  .---.---.---.---.
           |   |   |   |   |
        4  .---.---.---.---.

        See
          https://en.wikipedia.org/wiki/Irwin%E2%80%93Hall_distribution#Special_cases, "For n = 4"
        for the formulas of the interpolation coefficients.
        """
        pts = pts.sub(self.origin).div_(self.step)
        is_in_bounds = (pts.data >= 1).all(-1).logical_and_((pts.data < self.max_interp_grid_coord).all(-1))
        pts = pts.where(is_in_bounds.unsqueeze(-1), pts.new_tensor(1))
        out_shape, dims_n = pts.shape[:-1], pts.shape[-1]
        if self.channels_n != 0:
            out_shape = list(out_shape) + [self.channels_n]
        pts = pts.view(-1, dims_n)
        pts_n = len(pts)

        pts_int = pts.data.long()  # same as .floor().long()
        stencil = pts_int.new_tensor([-1, 0, 1, 2])
        ctrl_pt_ids = None
        for dim_i in range(dims_n - 1, -1, -1):
            dim_stride = self.control_pts.stride(dim_i + self.control_pts.ndim - dims_n)
            pts_int_i = pts_int[:, dim_i]
            shifts = pts_int_i.mul(dim_stride).unsqueeze(1).add(stencil * dim_stride); del pts_int_i
            if ctrl_pt_ids is None:
                ctrl_pt_ids = shifts
            else:
                ctrl_pt_ids = (ctrl_pt_ids.unsqueeze(1) + shifts.unsqueeze(2)).view(pts_n, -1)
            del shifts
        del stencil

        if self.channels_n == 0:
            ctrl_pts = self.control_pts.view(1, -1).expand(pts_n, -1).gather(1, ctrl_pt_ids)
            ctrl_pts = ctrl_pts.unsqueeze(1)
        else:
            ctrl_pts = self.control_pts.view(1, self.channels_n, -1).expand(pts_n, -1, -1)
            ctrl_pt_ids = ctrl_pt_ids.view(pts_n, 1, -1).expand(-1, self.channels_n, -1)
            ctrl_pts = ctrl_pts.gather(2, ctrl_pt_ids)
        del ctrl_pt_ids

        coefs = None
        for dim_i in range(dims_n - 1, -1, -1):
            pts_frac_i = pts[:, dim_i].sub(pts_int[:, dim_i])
            coefs_i = _coefs_inplace(pts_frac_i); del pts_frac_i
            if coefs is None:
                coefs = coefs_i
            else:
                coefs = (coefs.unsqueeze(1) * coefs_i.unsqueeze(2)).view(pts_n, -1)
            del coefs_i
        vals = ctrl_pts @ coefs.unsqueeze(2); del ctrl_pts, coefs
        vals = vals.view(out_shape)
        if self.channels_n != 0:
            is_in_bounds = is_in_bounds.unsqueeze(-1).expand_as(vals)
        vals = vals.where(is_in_bounds, vals.new_tensor(padding_value))
        return vals

    @torch.no_grad()
    def get_cell_ids(self, pts):
        r"""
        Parameters
        ----------
        pts : torch.Tensor
            of shape [**, dims_n].

        Returns
        -------
        cell_ids : torch.Tensor
            of shape [**, dims_n]. Cell id for out of bound points is -1.
        """
        cell_ids = pts.data.sub(self.origin).div_(self.step)
        is_in_bounds = (cell_ids >= 1).all(-1).logical_and_((cell_ids < self.max_interp_grid_coord).all(-1))
        cell_ids = cell_ids.long()
        cell_ids = cell_ids.where(is_in_bounds.unsqueeze(-1), cell_ids.new_tensor(-1))
        return cell_ids


def _coefs_inplace(x):
    r"""
    Parameters
    ----------
    x : torch.Tensor
        of shape [pts_n].

    Returns
    -------
    coefs : torch.Tensor
        of shape [pts_n, 4].
    """
    coefs = x.new_empty([len(x), 4])
    omx = 1 - x
    coefs[:, 0] = omx.pow(3)
    coefs[:, 2] = omx.pow_(2).mul_(x + 1).mul_(-3).add_(4); del omx
    coefs[:, 1] = (x - 2).mul_(x.pow(2)).mul_(3).add_(4)
    coefs[:, 3] = x.pow_(3); del x
    coefs = coefs.div_(6)
    return coefs
