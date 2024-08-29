from collections import OrderedDict

import torch

from skrgbd.calibration.camera_models.camera_model import CameraModel
from skrgbd.calibration.spline.cubic_bspline import CubicBSpline
from skrgbd.utils.torch import get_sub_dict


class UvDUndistortionModel(torch.nn.Module):
    r"""Implements S(u,v) * S(d) depth undistortion model.

    Parameters
    ----------
    d_spline : CubicBSpline
        with dims_n=1.
    uv_comp : torch.Tensor
        of shape [h, w].
    u_cell_ids : torch.LongTensor
        of shape [h, w].
    v_cell_ids : torch.LongTensor
        of shape [h, w].
    cell_is_calib : torch.BoolTensor
        of shape [un, vn, dn].
    """
    def __init__(self, d_spline, uv_comp, u_cell_ids, v_cell_ids, cell_is_calib):
        super().__init__()
        self.d_spline = d_spline
        self.register_buffer('uv_comp', uv_comp)
        self.register_buffer('u_cell_ids', u_cell_ids)
        self.register_buffer('v_cell_ids', v_cell_ids)
        self.register_buffer('cell_is_calib', cell_is_calib)

    @classmethod
    def from_pb_model(cls, pb_model, cam_model):
        r"""Makes the model from the pix-batch version.

        Parameters
        ----------
        pb_model : PixBatchUvDUndistortionModel
        cam_model : CameraModel

        Returns
        -------
        model : UvDUndistortionModel
        """
        uv = cam_model.get_pix_uvs(); del cam_model
        uv = uv.permute(1, 2, 0)
        uv_comp = pb_model.uv_spline(uv)
        u_ids, v_ids = pb_model.uv_spline.get_cell_ids(uv).unbind(-1); del uv
        model = cls(pb_model.d_spline, uv_comp, u_ids, v_ids, pb_model.cell_is_calib)
        return model

    @classmethod
    def from_state_dict(cls, state_dict):
        r"""Makes an empty model to load_state_dict to.

        Parameters
        ----------
        state_dict : OrderedDict

        Returns
        -------
        model : UvDUndistortionModel
        """
        d_spline_dict = get_sub_dict(state_dict, 'd_spline')
        d_spline = CubicBSpline.from_state_dict(d_spline_dict); del d_spline_dict
        pars = dict()
        for key in ['uv_comp', 'u_cell_ids', 'v_cell_ids', 'cell_is_calib']:
            pars[key] = torch.empty_like(state_dict[key])
        model = cls(d_spline, **pars)
        return model

    def forward(self, d_map, uncalib_val=float('nan')):
        r"""Undistorts a raw depthmap.

        Parameters
        ----------
        d_map : torch.Tensor
            of shape [**, h, w].
        uncalib_val : float
            Uncalibrated pixels will have this value.

        Returns
        -------
        undist_d_map : torch.Tensor
            of shape [**, h, w].
        """
        d_comp = self.d_spline(d_map.unsqueeze(-1)).squeeze(-1)
        undist_d_map = d_comp.mul_(self.uv_comp); del d_comp

        d_ids = self.d_spline.get_cell_ids(d_map.unsqueeze(-1)).squeeze(-1); del d_map
        u_ids, v_ids, d_ids = torch.broadcast_tensors(self.u_cell_ids, self.v_cell_ids, d_ids)
        is_calib = self.cell_is_calib[u_ids.reshape(-1), v_ids.reshape(-1), d_ids.reshape(-1)].view_as(d_ids)
        is_calib = is_calib.logical_and_(u_ids != -1); del u_ids
        is_calib = is_calib.logical_and_(v_ids != -1); del v_ids
        is_calib = is_calib.logical_and_(d_ids != -1); del d_ids
        undist_d_map = undist_d_map.where(is_calib, undist_d_map.new_tensor(uncalib_val)); del is_calib
        return undist_d_map


class PixBatchUvDUndistortionModel(torch.nn.Module):
    r"""Implements a variant of S(u,v) * S(d) depth undistortion model for training.

    Parameters
    ----------
    uv_spline : CubicBSpline
        with dims_n=2.
    d_spline : CubicBSpline
        with dims_n=1.
    """
    def __init__(self, uv_spline, d_spline):
        super().__init__()
        self.uv_spline = uv_spline
        self.d_spline = d_spline

        uvd_grid_size = list(uv_spline.control_pts.shape) + list(d_spline.control_pts.shape)
        uvd_grid_size = [s - 2 for s in uvd_grid_size]
        cell_is_calib = torch.zeros(uvd_grid_size, device=uv_spline.control_pts.device, dtype=torch.bool)
        self.register_buffer('cell_is_calib', cell_is_calib)

    @classmethod
    def from_sizes(cls, min_uv, max_uv, step_uv, min_d, max_d, step_d):
        r"""Makes an empty model from spline parameters.

        Parameters
        ----------
        min_uv : torch.Tensor
            of shape [2].
        max_uv : torch.Tensor
            of shape [2].
        step_uv : torch.Tensor
            of shape [2].
        min_d : torch.Tensor
            of shape [1].
        max_d : torch.Tensor
            of shape [1].
        step_d : torch.Tensor
            of shape [1].

        Returns
        -------
        model : PixBatchUvDUndistortionModel
        """
        uv_spline = CubicBSpline(min_uv, max_uv, step_uv)
        d_spline = CubicBSpline(min_d, max_d, step_d)
        model = cls(uv_spline, d_spline)
        return model

    @classmethod
    def from_state_dict(cls, state_dict):
        r"""Makes an empty model to load state dict to.

        Parameters
        ----------
        state_dict : OrderedDict

        Returns
        -------
        model : PixBatchUvDUndistortionModel
        """
        splines = []
        for name in ['uv_spline', 'd_spline']:
            spline_dict = get_sub_dict(state_dict, name)
            spline = CubicBSpline.from_state_dict(spline_dict); del spline_dict
            splines.append(spline); del spline
        model = cls(*splines); del splines
        return model

    def forward(self, uv, d, uncalib_val=float('nan')):
        r"""Undistorts a batch of raw depth pixels.

        Parameters
        ----------
        uv : torch.Tensor
            of shape [batch_size, 2].
        d : torch.Tensor
            of shape [batch_size, 1].
        uncalib_val : float
            Uncalibrated pixels will have this value.

        Returns
        -------
        undist_d : torch.Tensor
            of shape [batch_size].
        """
        undist_d = self.uv_spline(uv) * self.d_spline(d)

        u_ids, v_ids = self.uv_spline.get_cell_ids(uv).unbind(1); del uv
        d_ids = self.d_spline.get_cell_ids(d).squeeze(-1); del d
        is_calib = (u_ids != -1).logical_and_(v_ids != -1).logical_and_(d_ids != -1)
        is_calib = is_calib.logical_and_(self.cell_is_calib[u_ids, v_ids, d_ids]); del u_ids, v_ids, d_ids
        undist_d = undist_d.where(is_calib, undist_d.new_tensor(uncalib_val)); del is_calib
        return undist_d
