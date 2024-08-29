import numpy as np
import torch
import yaml

from skrgbd.optim.bfgs import BatchBFGS
from skrgbd.calibration.camera_models.camera_model import CameraModel
from skrgbd.utils.logging import DummyTqdm, tqdm
from skrgbd.calibration.camera_models.pinhole import PinholeCameraModel


class CentralGeneric(CameraModel):
    r"""Represents a central generic camera model, based on the implementation in
    https://github.com/puzzlepaint/camera_calibration/blob/master/applications/camera_calibration/generic_models/src/central_generic.h

    Parameters
    ----------
    config_yaml : str
        Path to intrinsics.yaml, used to unproject the points from the 2D image space into the 3D camera space.
    inverse_grid_pt : str
        Path to inverse_grid.pt, used to project the points from the 3D camera space into the 2D image space.
    """
    def __init__(self, config_yaml, inverse_grid_pt=None, dtype=None):
        with open(config_yaml, 'r') as f:
            config = yaml.load(f, yaml.SafeLoader)
        if config['type'] != 'CentralGenericModel':
            raise RuntimeError(f'Expected configuration for CentralGenericModel but got {config["type"]}')
        if dtype is None:
            dtype = torch.ones([]).dtype

        size_wh = torch.tensor([config['width'], config['height']])
        super().__init__(size_wh)

        # UV coordinates of the corners of the calibrated area:
        # top-left corner of top-left calibrated pixel
        calib_area_min = torch.tensor([config['calibration_min_x'], config['calibration_min_y']], dtype=dtype)
        self.register_buffer('calib_area_min', calib_area_min)
        # _top-left corner_ of bottom-right calibrated pixel.
        calib_area_max = torch.tensor([config['calibration_max_x'], config['calibration_max_y']], dtype=dtype)
        self.register_buffer('calib_area_max', calib_area_max)

        # Note that the control grid extends beyond the calibrated area by one cell,
        # so calib_area_min corresponds to grid[1, 1], and calib_area_max corresponds to grid[-2, -2].
        grid_wh = torch.tensor([config['grid_width'], config['grid_height']])
        self.register_buffer('grid_wh', grid_wh)
        control_points = torch.tensor(config['grid'], dtype=dtype).view(grid_wh[1], grid_wh[0], 3)
        control_points = control_points.permute(2, 0, 1).contiguous()
        self.control_points = torch.nn.Parameter(control_points, requires_grad=False)

        self._init_inverse_model()
        if inverse_grid_pt is not None:
            data = torch.load(inverse_grid_pt)
            self.inverse_control_points.copy_(data['points'])
            self.inverse_calib_area_min.copy_(data['min'])
            self.inverse_calib_area_max.copy_(data['max'])
            self.inverse_grid_wh.copy_(data['size'])

    # Point reprojection
    # ------------------
    def unproject(self, uv, nonan=False):
        r"""For points in the image space calculates the respective 3D directions in the camera space.

        Parameters
        ----------
        uv : torch.Tensor
            of shape [2, n], image coordinates of the points.
            The origin of the image space is in the top-left corner of the top-left pixel:
            the coordinates of the top-left corner are (0,0), the coordinates of the bottom-right corner are (w,h).

        nonan : bool
            If False, returns NaN for the points outside the calibrated area.
            If True, projects these points to the edge of the calibrated area.

        Returns
        -------
        direction : torch.Tensor
            of shape [3, n], directions in camera space, X to the right, Y down, Z from the camera.
        """
        grid_coord = self._uv_to_grid_coord(uv, nonan=nonan)
        direction = eval_uniform_cubic_bspline_surface(self.control_points, grid_coord); del grid_coord
        direction = torch.nn.functional.normalize(direction, dim=0)
        return direction

    def project(self, xyz, nonan=False, max_unproj_dev=None):
        r"""For points in the camera space calculates UV coordinates of their projections in the image space.

        Parameters
        ----------
        xyz : torch.Tensor
            of shape [3, points_n]
        nonan : bool
            If False, returns NaN for the points outside the calibrated area.
            If True, projects these points to the edge of the calibrated area.
        max_unproj_dev : float
            If not None, unproject the result and compare the unprojection to the input:
            if the deviation is greater than max_unproj_dev set the respective point to NaN.

        Returns
        -------
        uv : torch.Tensor
            of shape [2, n], image coordinates of the points.
            The origin of the image space is in the top-left corner of the top-left pixel:
            the coordinates of the top-left corner are (0,0), the coordinates of the bottom-right corner are (w,h).
        """
        xy = xyz[:2] / xyz[2]
        uv = self._project_xy(xy, nonan=nonan); del xy
        if nonan:
            uv = self._clip_uv(uv)
        else:
            if max_unproj_dev is None:
                uv = uv.where(self.uv_is_in_calibrated(uv), uv.new_full([], np.nan))
            else:
                xyz = torch.nn.functional.normalize(xyz, dim=0)
                unproj_dir = self.unproject(uv)
                dev = unproj_dir.sub_(xyz).norm(dim=0); del xyz, unproj_dir
                good_dev = dev <= max_unproj_dev; del dev
                good_dev = good_dev.unsqueeze(0).expand(2, -1)
                uv = uv.where(good_dev, uv.new_full([], np.nan))
        return uv

    def project_fine(self, xyz, max_iters_n=100, show_progress=True):
        r"""For points in the camera space calculates UV coordinates of their projections in the image space.
        After initial estimate with `project` fine tunes UVs w.r.t deviation of their unprojections from xyz directions.

        Parameters
        ----------
        xyz : torch.Tensor
            of shape [3, points_n]
        max_iters_n : int
            Maximum number of iterations to optimize for.
        show_progress : bool
        Returns
        -------
        uv : torch.Tensor
            of shape [2, n], image coordinates of the points.
            The origin of the image space is in the top-left corner of the top-left pixel:
            the coordinates of the top-left corner are (0,0), the coordinates of the bottom-right corner are (w,h).
        """
        show_progress = tqdm if show_progress else DummyTqdm

        uv = self.project(xyz, nonan=True)
        uv = uv.T.contiguous().requires_grad_()

        target_direction = torch.nn.functional.normalize(xyz.T, dim=1); del xyz

        optimizer = BatchBFGS([uv], line_search_fn='strong_wolfe')
        mean_losses = []
        max_losses = []

        def closure(not_converged_ids):
            direction = self.unproject(uv[not_converged_ids].T).T
            loss = (direction - target_direction[not_converged_ids]).pow(2).sum(1); del direction
            optimizer.zero_grad()
            loss.backward(torch.ones_like(loss))
            return loss

        progress = show_progress(range(max_iters_n))
        for _ in progress:
            loss = optimizer.step(closure)
            if loss is None:
                break
            loss = loss.data
            mean_losses.append(loss.mean().item())
            loss = loss.data.max().item()
            max_losses.append(loss)
            progress.set_description(f'Loss {loss:.2e}')
        del optimizer

        uv = uv.detach().T
        # stats = dict(mean_losses=mean_losses, max_losses=max_losses)
        return uv

    # Resize, crop
    # ------------
    def resize_(self, new_wh):
        r"""Resizes camera model inplace to a new resolution.

        Parameters
        ----------
        new_wh : torch.Tensor
        """
        old_wh = self.size_wh.clone()
        new_wh = new_wh.to(old_wh)
        self.size_wh.copy_(new_wh)
        self.calib_area_min.mul_(new_wh).div_(old_wh)
        self.calib_area_max.mul_(new_wh).div_(old_wh)
        self.inverse_control_points.data.mul_(new_wh.unsqueeze(1).unsqueeze(2)).div_(old_wh.unsqueeze(1).unsqueeze(2))
        return self

    # Dtype, device
    # -------------
    @property
    def dtype(self):
        return self.control_points.dtype

    # Fitting
    # -------
    def fit_pinhole(self, verbose=True, samples_n=None, subsamples_n=100, iters_n=1000):
        r"""Finds the pinhole camera model that best fits this camera model.

        The algorithm is:
        1. Sample N random points in the image space. Assign a random depth to each point.
        2. Unproject these points to the 3D camera space of this camera.
        3. Project these points to the image space of the pinhole camera.
        4. Estimate the pinhole parameters so that the projection from step 3 best fits the samples from step 1.

        Parameters
        ----------
        verbose : bool
            If True, show the progress and return the fitting curve.
        samples_n : int or None
            Number of UV samples to use for fitting.
            The default is the number of grid points times subsamples_n.
        subsamples_n : int
            Number of subsamples per grid point.
        iters_n : int
            Number iterations. Note that LBFGS does multiple iterations per optimization iterations,
            so the number of iterations in the returned losses will be different.

        Returns
        -------
        pinhole_model : PinholeCameraModel
        losses : torch.Tensor
            of shape [iters_n], MSE of sample projections calculated with the fitted inverse model.
        residuals : torch.Tensor
            of shape [samples_n], deviations of sample projections calculated with the fitted inverse model.
        uv : torch.Tensor
            of shape [2, samples_n], UV coordinates of samples.
        """
        device = self.device
        if samples_n is None:
            grid_knots_n = self.inverse_grid_wh.prod().item()
            samples_n = grid_knots_n * subsamples_n

        min_u, max_u = self.calib_area_min[0], (self.calib_area_max[0] + 1)
        min_v, max_v = self.calib_area_min[1], (self.calib_area_max[1] + 1)

        # 1. Sample UV points.
        u = torch.rand(samples_n, device=device) * (max_u - min_u) + min_u
        v = torch.rand(samples_n, device=device) * (max_v - min_v) + min_v
        uv = torch.stack([u, v]).reshape(2, -1); del u, v

        # 2. Unproject the UV points.
        dirs = self.unproject(uv)

        # 3. Fit the pinhole model.
        principal_point = [(min_u + max_u) / 2, (min_v + max_v) / 2]
        focal = self.size_wh.to(self.dtype, copy=True)
        pinhole = PinholeCameraModel(focal, principal_point, self.size_wh.cpu()).to(device)
        pinhole.requires_grad_(True)
        loss_scale = 1e15
        optim = torch.optim.LBFGS(pinhole.parameters(), lr=1)
        losses = []

        def closure():
            est_uv = pinhole.project(dirs)
            loss = torch.nn.functional.mse_loss(est_uv, uv); del est_uv
            losses.append(loss.item())
            optim.zero_grad()
            loss = loss * loss_scale
            loss.backward()
            return loss

        iters = range(iters_n)
        if verbose:
            t = tqdm(iters)
            for i in t:
                optim.step(closure)
                t.set_description(f'Loss {losses[-1]:.2e}')
        else:
            for i in iters:
                optim.step(closure)

        pinhole.requires_grad_(False)

        if verbose:
            losses = torch.tensor(losses)
            est_uv = pinhole.project(dirs)
            residuals = (est_uv - uv).norm(dim=0).cpu()
            return pinhole, losses, residuals, uv.cpu()
        else:
            return pinhole

    # Private
    # -------
    def uv_is_in_calibrated(self, uv):
        r"""For points in the image space checks if they are inside the calibrated area.

        Parameters
        ----------
        uv : torch.Tensor
            of shape [2, n], image coordinates of the points.
            The origin of the image space is in the top-left corner of the top-left pixel:
            the coordinates of the top-left corner are (0,0), the coordinates of the bottom-right corner are (w,h).
        Returns
        -------
        in_area : torch.Tensor
            of shape [n] of type bool
        """
        min_uv = self.calib_area_min.unsqueeze(1).expand_as(uv)
        max_uv = self.calib_area_max.add(1).unsqueeze(1).expand_as(uv)
        in_grid = (uv >= min_uv).all(0).bitwise_and_((uv < max_uv).all(0))
        return in_grid

    def _clip_uv(self, uv):
        eps = torch.finfo(uv.dtype).eps
        min_uv = self.calib_area_min.unsqueeze(1).expand_as(uv)
        max_uv = self.calib_area_max.add(1).mul(1 - eps).unsqueeze(1).expand_as(uv)
        return uv.max(min_uv).min(max_uv)

    def _project_xy(self, xy, nonan=False):
        r"""For points on the z=1 plane in the camera space
        calculates UV coordinates of their projections in the image space.

        Parameters
        ----------
        xy : torch.Tensor
            of shape [2, n]

        nonan : bool
            If False, returns NaN for the points outside the calibrated area.
            If True, projects these points to the edge of the calibrated area.

        Returns
        -------
        uv : torch.Tensor
            of shape [2, n], image coordinates of the points.
            The origin of the image space is in the top-left corner of the top-left pixel:
            the coordinates of the top-left corner are (0,0), the coordinates of the bottom-right corner are (w,h).
        """
        grid_coord = self._xy_to_grid_coord(xy, nonan=nonan)
        uv = eval_uniform_cubic_bspline_surface(self.inverse_control_points, grid_coord); del grid_coord
        return uv

    def _uv_to_grid_coord(self, uv, nonan=False):
        r"""For points in the image space finds their respective control grid coordinates.
        The integer part of the result equals to the ids of the grid cells that the points lie in.

        Based on https://github.com/puzzlepaint/camera_calibration/blob/master/applications/camera_calibration/generic_models/src/central_generic.h#L515.

        Parameters
        ----------
        uv : torch.Tensor
            of shape [2, n], image coordinates of the points.
            The origin of the image space is in the top-left corner of the top-left pixel:
            the coordinates of the top-left corner are (0,0), the coordinates of the bottom-right corner are (w,h).

        Returns
        -------
        grid_coordinate : torch.Tensor
            of shape [2, n]
        nonan : bool

        Notes
        -----
        The control grid extends beyond the calibration area by one cell,
        so calib_area_min corresponds to grid[1, 1], and calib_area_max corresponds to grid[-2, -2].
        """
        if nonan:
            uv = self._clip_uv(uv)
        else:
            in_grid = self.uv_is_in_calibrated(uv)
            uv = uv.where(in_grid.unsqueeze(0).expand_as(uv), uv.new_full([], np.nan)); del in_grid

        calib_area_size = self.calib_area_max + 1 - self.calib_area_min
        _ = (uv - self.calib_area_min.unsqueeze(1)).div_(calib_area_size.unsqueeze(1))
        _ = _.mul_((self.grid_wh - 3).unsqueeze(1)).add_(1)
        return _

    def _xy_to_grid_coord(self, xy, nonan=False):
        r"""For points on the z=1 plane in the camera space finds their respective control grid coordinates.
        The integer part of the result equals to the ids of the grid cells that the points lie in.

        Parameters
        ----------
        xy : torch.Tensor
            of shape [2, n]
        nonan : bool

        Returns
        -------
        grid_coordinate : torch.Tensor
            of shape [2, n]

        Notes
        -----
        The control grid extends beyond the calibration area by one cell,
        so calib_area_min corresponds to grid[1, 1], and calib_area_max corresponds to grid[-2, -2].
        """
        min_xy = self.inverse_calib_area_min.unsqueeze(1).expand_as(xy)
        max_xy = self.inverse_calib_area_max.unsqueeze(1).expand_as(xy)

        if nonan:
            xy = xy.max(min_xy).min(max_xy)
        else:
            in_grid = (xy >= min_xy).all(0).bitwise_and_((xy < max_xy).all(0))
            xy = xy.where(in_grid.unsqueeze(0).expand_as(xy), xy.new_full([], np.nan)); del in_grid

        inverse_calib_area_size = self.inverse_calib_area_max - self.inverse_calib_area_min
        _ = (xy - min_xy).div_(inverse_calib_area_size.unsqueeze(1))
        _ = _.mul_((self.inverse_grid_wh - 3).unsqueeze(1)).add_(1)
        return _

    def _init_inverse_model(self, grid_wh=None):
        r"""Initializes the B-spline surface used for projection from camera space to image space.

        Parameters
        ----------
        grid_wh : iterable of int or None
            Size of the grid. If None, the size of the "forward" grid is used.
        """
        # 1. Sample UV points on the edges of the calibrated area.
        min_u, max_u = self.calib_area_min[0], self.calib_area_max[0] + 1
        min_v, max_v = self.calib_area_min[1], self.calib_area_max[1] + 1

        subsamples_n = 10
        u = torch.linspace(min_u, max_u, int((max_u - min_u) * subsamples_n + 1), dtype=self.dtype)
        v = torch.linspace(min_v, max_v, int((max_v - min_v) * subsamples_n + 1), dtype=self.dtype)

        uv = torch.cat([
            torch.stack([u, torch.full_like(u, min_v)]),
            torch.stack([u, torch.full_like(u, max_v)]),
            torch.stack([torch.full_like(v, min_u), v]),
            torch.stack([torch.full_like(v, max_u), v]),
        ], 1)
        del u, v

        # 2. Unproject the UV points.
        dirs = self.unproject(uv, nonan=True)
        xy = dirs[:2] / dirs[2]; del dirs

        # 3. Initialize the inverse calibrated area.
        inverse_calib_area_min = xy.min(1)[0]
        inverse_calib_area_max = xy.max(1)[0]
        if grid_wh is None:
            inverse_grid_wh = self.grid_wh.clone()
        else:
            inverse_grid_wh = torch.tensor(grid_wh)
        w, h = inverse_grid_wh
        inverse_control_points = torch.empty(2, h, w, dtype=self.dtype)
        inverse_control_points[0] = self.size_wh[0] / 2
        inverse_control_points[1] = self.size_wh[1] / 2

        self.register_buffer('inverse_calib_area_min', inverse_calib_area_min)
        self.register_buffer('inverse_calib_area_max', inverse_calib_area_max)
        self.register_buffer('inverse_grid_wh', inverse_grid_wh)
        self.inverse_control_points = torch.nn.Parameter(inverse_control_points, requires_grad=False)

    def _fit_inverse_model(self, verbose=True, samples_n=None, subsamples_n=100, iters_n=1000):
        r"""Fits the control points of the B-spline surface used for projection from camera space to image space.

        Parameters
        ----------
        verbose : bool
            If True, show the progress and return the fitting curve.
        samples_n : int or None
            Number of UV samples to use for fitting.
            The default is the number of grid points times subsamples_n.
        subsamples_n : int
            Number of subsamples per grid point.
        iters_n : int
            Number iterations. Note that LBFGS does multiple iterations per optimization iterations,
            so the number of iterations in the returned losses will be different.

        Returns
        -------
        losses : torch.Tensor
            of shape [iters_n], MSE of sample projections calculated with the fitted inverse model.
        residuals : torch.Tensor
            of shape [samples_n], deviations of sample projections calculated with the fitted inverse model.
        uv : torch.Tensor
            of shape [2, samples_n], UV coordinates of samples.
        """
        device = self.device
        if samples_n is None:
            grid_knots_n = self.inverse_grid_wh.prod().item()
            samples_n = grid_knots_n * subsamples_n

        min_u, max_u = self.calib_area_min[0], (self.calib_area_max[0] + 1)
        min_v, max_v = self.calib_area_min[1], (self.calib_area_max[1] + 1)

        # 1. Sample UV points.
        u = torch.rand(samples_n, device=device) * (max_u - min_u) + min_u
        v = torch.rand(samples_n, device=device) * (max_v - min_v) + min_v
        uv = torch.stack([u, v]).reshape(2, -1); del u, v

        # 2. Unproject the UV points.
        dirs = self.unproject(uv)
        xy = dirs[:2] / dirs[2]; del dirs

        # 3. Transform UV from HxW to 1x1.
        uv = uv / self.size_wh.unsqueeze(1)

        # 4. Fit the inverse model.
        self.inverse_control_points.requires_grad_(True)
        loss_scale = 1e15
        optim = torch.optim.LBFGS([self.inverse_control_points], lr=1)
        losses = []

        def closure():
            est_uv = self._project_xy(xy, nonan=True).div_(self.size_wh.unsqueeze(1))
            loss = torch.nn.functional.mse_loss(est_uv, uv); del est_uv
            losses.append(loss.item())
            optim.zero_grad()
            loss = loss * loss_scale
            loss.backward()
            return loss

        iters = range(iters_n)
        if verbose:
            t = tqdm(iters)
            for i in t:
                optim.step(closure)
                t.set_description(f'Loss {losses[-1]:.2e}')
        else:
            for i in iters:
                optim.step(closure)

        self.inverse_control_points.requires_grad_(False)
        if verbose:
            losses = torch.tensor(losses)
            uv = uv * self.size_wh.unsqueeze(1)
            est_uv = self._project_xy(xy, nonan=True)
            residuals = (est_uv - uv).norm(dim=0).cpu()
            return losses, residuals, uv.cpu()

    def _save_inverse_grid(self, file):
        data = dict()
        data['points'] = self.inverse_control_points.detach().cpu()
        data['min'] = self.inverse_calib_area_min.cpu()
        data['max'] = self.inverse_calib_area_max.cpu()
        data['size'] = self.inverse_grid_wh.cpu()
        return torch.save(data, file)


def eval_uniform_cubic_bspline_surface(control_points, points):
    r"""Interpolates the values of control_points at points in grid coordinates with cubic B-Splines.

    The integer part of the point coordinates equal to the ids of the grid cells that the points lie in.
    The value at the point (x,y) is calculated from the values of the surrounding control points as illustrated,
       A   B       C   D
    0  o---o-------o---o
       |   |       |   |
    1  o---#-------o---o
       |   | (x,y) |   |
    2  o---o-------o---o
       |   |       |   |
    3  o---o-------o---o
    where the id of the point marked with '#' equals to the integer part of (x,y).
    The value is calculated using cubic B-Spline surface, as if the point was two cells down the bottom-right corner
       A   B   C   D
    0  o---o---o---o
       |   |   |   |
    1  o---o---o---o
       |   |   |   |
    2  o---o---o---o
       |   |   |   |
    3  o---o---o---.-------.
                   | (x,y) |
                   .-------.

    Based on https://github.com/puzzlepaint/camera_calibration/blob/master/applications/camera_calibration/generic_models/src/util.h#L54.

    Parameters
    ----------
    control_points : torch.Tensor
        of shape [dims_n, h, w]
    points : torch.Tensor
        of shape [2, points_n]

    Returns
    -------
    value : torch.Tensor
        of shape [dims_n, points_n]
    """
    dims_n, h, w = control_points.shape

    # Shift the point two cells down the bottom-right corner to use the surrounding points in the formulas
    x = points[0] + 2
    y = points[1] + 2; del points

    eps = torch.finfo(x.dtype).eps
    x = x.clamp_(max=w * (1 - eps))
    y = y.clamp_(max=h * (1 - eps))

    # Calculate the id of the shifted cell, reject invalid coordinates
    j = x.where(x.isfinite(), x.new_full([], 3)).long()
    i = y.where(y.isfinite(), y.new_full([], 3)).long()

    # Calculate the ids of the control points
    i0, i1, i2, i3 = i - 3, i - 2, i - 1, i; del i
    jA, jB, jC, jD = j - 3, j - 2, j - 1, j; del j

    # Recalculate (x,y) w.r.t the upper left control point
    x = x.sub_(jA)
    y = y.sub_(i0)

    # Calculate X coefficients
    cA, cB, cC, cD = _bspline_coefs_inplace(x); del x

    # Interpolate grid points in X and calculate the intermediate Y control points
    control_points = control_points.view(dims_n, -1)

    def calculate_point(i):
        i = i.mul_(w)
        _ = control_points.gather(1, (jA + i).unsqueeze(0).expand(dims_n, -1)).mul_(cA).add_(
            control_points.gather(1, (jB + i).unsqueeze(0).expand(dims_n, -1)).mul_(cB)).add_(
            control_points.gather(1, (jC + i).unsqueeze(0).expand(dims_n, -1)).mul_(cC)).add_(
            control_points.gather(1, (jD + i).unsqueeze(0).expand(dims_n, -1)).mul_(cD))
        return _

    p0 = calculate_point(i0); del i0
    p1 = calculate_point(i1); del i1
    p2 = calculate_point(i2); del i2
    p3 = calculate_point(i3); del i3, jA, jB, jC, jD, cA, cB, cC, cD

    # Calculate Y coefficients
    c0, c1, c2, c3 = _bspline_coefs_inplace(y); del y

    value = p0.mul_(c0).add_(p1.mul_(c1)).add_(p2.mul_(c2)).add_(p3.mul_(c3))
    return value


def _bspline_coefs_inplace(x):
    x2 = x.pow(2)
    x3 = x.pow(3)
    cA = (4 - x).pow_(3).div_(6)
    cB = x.mul(39).sub_(x2 * 11).add_(x3).sub_(131 / 3).div_(2)
    cC = x2.mul_(5).sub_(x3.div_(2)).sub_(x * 16).add_(50 / 3); del x2, x3
    cD = x.sub(3).pow_(3).div_(6)
    return cA, cB, cC, cD
