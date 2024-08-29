import numpy as np
from scipy.spatial.transform import Rotation
import scipy.optimize
import torch

from skrgbd.optim.bfgs import BatchBFGS
from skrgbd.calibration.camera_models.camera_model import CameraModel
from skrgbd.utils.logging import DummyTqdm, tqdm
from skrgbd.utils import ignore_warnings


class RVCameraModel(CameraModel):
    r"""Represents a camera model used in RangeVision ScanCenter.

    Parameters
    ----------
    focal_length : float
    pixel_size : array-like
        (x, y).
    principal_point : array-like
        (cx, cy).
    undist_coeffs : array-like
        (d0, d1, d2, d3, d4, d5).
    board_to_camera : array-like
        of shape [4, 4].
    """
    @ignore_warnings(['To copy construct from a tensor, it is recommended to use'])
    def __init__(self, focal_length, pixel_size, principal_point, undist_coeffs, board_to_camera):
        size_wh = [2048, 1536]
        super().__init__(size_wh)
        self.register_buffer('focal_length', torch.tensor(focal_length))
        self.register_buffer('pixel_size', torch.tensor(pixel_size))
        self.register_buffer('principal_point', torch.tensor(principal_point))
        self.register_buffer('undist_coeffs', torch.tensor(undist_coeffs))
        self.register_buffer('board_to_camera', torch.tensor(board_to_camera))

    @classmethod
    def from_impar_txt(cls, impar_txt):
        r"""Makes the model from impar.txt

        Parameters
        ----------
        impar_txt : str

        Returns
        -------
        cam_model : RVCameraModel
        """
        lines = open(impar_txt).readlines()

        # Intrinsics.
        # f: Focal length in mm is not estimated during calibration. Is taken into account in the size of the pixel.
        focal_length = float(lines[3])
        # m: Size of the pixel [x,y] in mm.
        pixel_size = torch.tensor([float(_) for _ in lines[12].split()])
        # b: Principal point [x,y] in pixels.
        principal_point = torch.tensor([float(_) for _ in lines[14].split()])

        # Undistortion coefficients.
        d0, d1, d2 = [float(_) for _ in lines[16].split()]
        d3, d4, d5 = [float(_) for _ in lines[17].split()]
        undist_coeffs = (d0, d1, d2, d3, d4, d5)

        # Camera extrinsics w.r.t the calibration board.
        # alf, om, kap: Euler angles of the camera.
        alf, om, kap = [float(_) for _ in lines[7].split()]
        board_to_camera_rotation = Rotation.from_euler('yxz', [alf, -om, -kap]).as_matrix()
        # X0: Camera position in mm.
        x0 = np.array([float(_) for _ in lines[9].split()])
        x0 = x0 / 1000  # mm to meters

        board_to_camera = torch.zeros(4, 4)
        board_to_camera[3, 3] = 1
        board_to_camera.numpy()[:3, :3] = board_to_camera_rotation
        board_to_camera.numpy()[:3, 3] = -board_to_camera_rotation @ x0

        cam_model = cls(focal_length, pixel_size, principal_point, undist_coeffs, board_to_camera)
        return cam_model

    @classmethod
    def from_state_dict(cls, state_dict):
        r"""Makes an empty model to load_state_dict to.

        Parameters
        ----------
        state_dict : OrderedDict

        Returns
        -------
        cam_model : RVCameraModel
        """
        cam_model = cls(0., [0., 0.], [0., 0.], [0.] * 6, [[0.] * 4] * 4)
        if 'uvn_to_uv_map' in state_dict:
            cam_model.register_buffer('uvn_to_uv_map', torch.empty_like(state_dict['uvn_to_uv_map']))
            cam_model.register_buffer('uvn_min', torch.empty_like(state_dict['uvn_min']))
            cam_model.register_buffer('uvn_max', torch.empty_like(state_dict['uvn_max']))
        return cam_model

    # Point reprojection
    # ------------------
    def unproject(self, uv):
        r"""For points in the image space calculates the respective 3D directions in the camera space.

        Parameters
        ----------
        uv : torch.Tensor
            of shape [2, n], image coordinates of the points.
            The origin of the image space is in the top-left corner of the top-left pixel:
            the coordinates of the top-left corner are (0,0),
            the coordinates of the bottom-right corner are (w,h).

        Returns
        -------
        direction : torch.Tensor
            of shape [3, n], directions in camera space (!!!), X to the right, Y up, Z to the camera.
        """
        uv = (uv - self.principal_point.unsqueeze(1))
        _ = self.pixel_size * uv.new_tensor([1, -1])  # flip y axis
        uv = uv.mul_(_.unsqueeze(1))

        # Undistort
        d0, d1, d2, d3, d4, d5 = self.undist_coeffs
        u, v = uv; del uv
        u2 = u.square()
        v2 = v.square()
        two_uv = u.mul(v).mul_(2)
        r2 = u2 + v2

        r4 = r2.square()
        r6 = r2.pow(3)
        radial = d1 * r2 + d2 * r4 + d3 * r6; del r4, r6

        # Note that `du = v * d0 ...` and `dv = u * d0 ...` is not a typo according to RV.
        du = v * d0 + u * radial + d4 * (r2 + 2 * u2) + d5 * two_uv; del u2
        dv = u * d0 + v * radial + d5 * (r2 + 2 * v2) + d4 * two_uv; del v2, two_uv, r2, radial

        u = u.add(du); del du
        v = v.add(dv); del dv

        dir_z = torch.full_like(u, -self.focal_length)
        direction = torch.stack([u, v, dir_z]); del u, v, dir_z
        direction = torch.nn.functional.normalize(direction, dim=0)
        return direction

    def project(self, xyz):
        r"""For points in the camera space calculates UV coordinates of their projections in the image space.

        Parameters
        ----------
        xyz : torch.Tensor
            of shape [3, points_n]

        Returns
        -------
        uv : torch.Tensor
            of shape [2, n], image coordinates of the points.
            The origin of the image space is in the top-left corner of the top-left pixel:
            the coordinates of the top-left corner are (0,0), the coordinates of the bottom-right corner are (w,h).
            Returns NaN for points projected outside the calibrated area.
        """
        uvn = xyz[:2] / xyz[2]
        uv = self._uvn_to_uv(uvn); del uvn
        return uv

    def project_fine(self, xyz, **kwargs):
        r"""For this model `project` is sufficiently precise, so simply calls `project`."""
        return self.project(xyz)

    # Resize, crop
    # ------------
    def resize_(self, new_wh):
        raise NotImplementedError

    # Dtype, device
    # -------------
    @property
    def dtype(self):
        return self.principal_point.dtype

    # Fitting
    # -------
    def fit_projection_parameters(self, grid_wh=(512, 384), max_iters_n=100, show_progress=True):
        r"""Fits the parameters for 3D-to-2D projection.

        Parameters
        ----------
        grid_wh : tuple of int
            (width, height) of the inverse warping grid.

        Returns
        -------
        losses : list
            of shape [iters_n]
        """
        show_progress = tqdm if show_progress else DummyTqdm

        pix_rays = self.get_pixel_rays().view(3, -1)
        uvn = pix_rays[:2] / pix_rays[2]; del pix_rays
        uvn_min = uvn.min(1)[0]
        uvn_max = uvn.max(1)[0]; del uvn

        w, h = self.size_wh.cpu().tolist()
        un_min, un_max, u_n, u_min, u_max = uvn_min[0], uvn_max[0], grid_wh[0], .5, w - .5
        vn_min, vn_max, v_n, v_min, v_max = uvn_min[1], uvn_max[1], grid_wh[1], .5, h - .5

        v, u = torch.meshgrid([torch.linspace(vn_min, vn_max, v_n), torch.linspace(un_min, un_max, u_n)])
        uvn = torch.stack([u, v]).view(2, -1); del u, v
        v, u = torch.meshgrid([torch.linspace(v_min, v_max, v_n), torch.linspace(u_min, u_max, u_n)])
        uv = torch.stack([u, v]).view(2, -1); del u, v

        uv = uv.T.to(self.device, self.dtype).requires_grad_()
        uvn = uvn.T.to(self.device, self.dtype)

        optimizer = BatchBFGS([uv], line_search_fn='strong_wolfe')
        losses = []

        def closure(not_converged_ids):
            directions = self.unproject(uv[not_converged_ids].T).T
            uvn_est = directions[:, :2] / directions[:, 2:3]; del directions
            loss = (uvn_est - uvn[not_converged_ids]).pow(2).sum(1); del uvn_est
            optimizer.zero_grad()
            loss.backward(torch.ones_like(loss))
            return loss

        progress = show_progress(range(max_iters_n))
        for _ in progress:
            loss = optimizer.step(closure)
            if loss is None:
                break
            # loss = loss.data.mean().item()
            loss = loss.data.max().item()
            progress.set_description(f'Loss {loss:.2e}')
            losses.append(loss)

        uvn_to_uv_map = uv.T.detach().view(2, grid_wh[1], grid_wh[0])
        uvn_min = uvn_min.to(uvn_to_uv_map)
        uvn_max = uvn_max.to(uvn_to_uv_map)

        self.register_buffer('uvn_to_uv_map', uvn_to_uv_map)
        self.register_buffer('uvn_min', uvn_min)
        self.register_buffer('uvn_max', uvn_max)
        return losses

    # Private
    # -------
    def _uvn_to_uv(self, uvn):
        r"""For points on the z=1 plane in the camera space
        calculates UV coordinates of their projections in the image space.

        Parameters
        ----------
        uvn : torch.Tensor
            of shape [2, n]

        Returns
        -------
        uv : torch.Tensor
            of shape [2, n], image coordinates of the points.
            The origin of the image space is in the top-left corner of the top-left pixel:
            the coordinates of the top-left corner are (0,0), the coordinates of the bottom-right corner are (w,h).
            Returns NaN for points projected outside the calibrated area.
        """
        uvn = (uvn.T - self.uvn_min) / (self.uvn_max - self.uvn_min) * 2 - 1
        is_in_bounds = (uvn >= -1).all(1).logical_and_((uvn <= 1).all(1))
        uvn = uvn.where(is_in_bounds.unsqueeze(1).expand_as(uvn), uvn.new_tensor(np.nan)); del is_in_bounds

        uvn = uvn.view(1, 1, -1, 2)
        uvn_to_uv_map = self.uvn_to_uv_map.unsqueeze(0)
        uv = torch.nn.functional.grid_sample(uvn_to_uv_map, uvn, mode='bilinear', align_corners=True)
        uv = uv.view(2, -1)
        return uv


def fit_camera_model(camera_model, rv_camera_model, samples_n=1_000_000, depth_range=(.1, 2), verbose=0):
    r"""Finds extrinsics of the camera, described by the first camera model, w.r.t the RV calibration board,
    given the RV camera model of that camera.

    The algorithm is:
    1. Sample N random points in the image space. Assign a random depth to each point.
    2. Unproject these points to the 3D camera space corresponding to the first camera model.
    3. Unproject these points to the 3D camera space corresponding to the RV camera model.
    4. Transform the points from the RV camera space to the RV calibration board space.
    5. Estimate the transform of these points from the board space to the first camera space,
       so that the transformed points and the points from Step 2 are close.
       Estimation is done via Least-Squares with Levenberg-Marquardt.

    Parameters
    ----------
    camera_model : CameraModel
    rv_camera_model : RVCameraModel
    samples_n : int
        Number of point samples.
    depth_range : tuple of float
        (min_depth, max_depth), the range to sample the depth for each point from.
    verbose : {0, 1, 2}
        Level of verbosity of least-squares solver.

    Returns
    -------
    rotation : torch.Tensor
        of shape [3, 3], rotation matrix of board-to-camera transform.
    translation : torch.Tensor
        of shape [3], translation vector of board-to-camera transform.
    residuals : np.ndarray
        of shape [samples_n], squared deviations of sampled points.
    """
    # 1. Sample N random points in the image space. Assign a random depth to each point.
    min_u, max_u = camera_model.calib_area_min[0], (camera_model.calib_area_max[0] + 1)
    min_v, max_v = camera_model.calib_area_min[1], (camera_model.calib_area_max[1] + 1)
    min_d, max_d = depth_range

    u = torch.rand(samples_n) * (max_u - min_u) + min_u
    v = torch.rand(samples_n) * (max_v - min_v) + min_v
    d = torch.rand(samples_n) * (max_d - min_d) + min_d
    uv = torch.stack([u, v]).reshape(2, -1)

    # 2. Unproject these points to the 3D camera space corresponding to the first camera model.
    cam_pts = camera_model.unproject(uv) * d

    # 3. Unproject these points to the 3D camera space corresponding to the RV camera model.
    # 4. Transform the points from the RV camera space to the RV calibration board space.
    rv_cam_pts = rv_camera_model.unproject(uv) * d
    rv_board_pts = rv_camera_model.camera_to_board[:3, :3] @ rv_cam_pts + rv_camera_model.camera_to_board[:3, 3:4]
    del rv_cam_pts

    # 5. Estimate the transform of these points from the board space to the first camera space,
    #    so that the transformed points and the points from Step 2 are close.
    def cost_fn(x):
        rotation, translation = x_to_extrinsics(x)
        rv_pts_reprojected = rotation @ rv_board_pts + translation.unsqueeze(1)
        cost = torch.nn.functional.mse_loss(rv_pts_reprojected, cam_pts, reduction='none').sum(0)
        return cost  # squared deviation of each point, of shape [pts_n]

    def x_to_extrinsics(x):
        translation = torch.from_numpy(x[:3])
        rotation = x[3:6]
        rotation = Rotation.from_euler('xyz', rotation).as_matrix()
        rotation = torch.from_numpy(rotation)
        return rotation, translation

    x0 = [0, 0, 0, 0, 0, 0]
    solution = scipy.optimize.least_squares(cost_fn, x0, method='lm', verbose=verbose)
    rotation, translation = x_to_extrinsics(solution.x)
    return rotation, translation, solution.fun
