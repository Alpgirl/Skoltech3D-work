from PIL import Image
import numpy as np
import open3d as o3d
import torch

from skrgbd.calibration.camera_models.camera_model import CameraModel
from skrgbd.utils import ignore_warnings
from skrgbd.data.io import imgio
from skrgbd.calibration.camera_models import load_from_colmap_txt, load_from_pt
from skrgbd.data.io.poses import load_poses
from skrgbd.data.dataset.scene_paths import ScenePaths


class Helper:
    r"""Encapsulates data-loading for image alignment in Sk3D.

    Parameters
    ----------
    scene_paths : ScenePaths
    """
    def __init__(self, scene_paths):
        self.scene_paths = scene_paths

    def load_cam_model(self, cam, mode):
        r"""Loads RV camera model for the SL right camera, and pinhole camera model for the other sensors.

        Parameters
        ----------
        cam : str
        mode : str

        Returns
        -------
        cam_model : CameraModel
        """
        if cam == 'stl_right':
            cam_model = self.scene_paths.cam_model(cam, 'rgb', 'pt')
            cam_model = load_from_pt(cam_model)
        else:
            cam_model = self.scene_paths.cam_model(cam, mode)
            cam_model = load_from_colmap_txt(cam_model)
        return cam_model

    def load_cam_poses(self, cam, mode, var='calib'):
        r"""Loads camera poses.
        For the SL right camera, loads the poses consistent with the RV camera model.

        Parameters
        ----------
        cam : str
        mode : str
        var : str

        Returns
        -------
        world_to_cam : torch.Tensor
            of shape [views_n, 4, 4].
        """
        if cam == 'stl_right':
            if var != 'ref':
                raise ValueError('Only refined cam poses for stl_right')
            s2w = self.scene_paths.sl_board_to_w_refined()
            s2w = torch.load(s2w)
            s2w = torch.stack(s2w, 0).double()
            w2s = s2w.inverse(); del s2w
            cam_model = self.load_cam_model(cam, mode)
            w2c = cam_model.board_to_camera.double() @ w2s; del cam_model, w2s
        else:
            w2c = self.scene_paths.cam_poses(cam, mode, var)
            w2c = load_poses(w2c)
        return w2c

    @ignore_warnings(['The given NumPy array is not writeable, and PyTorch'])
    def load_img(self, cam, mode, view_i, light=None):
        r"""Loads the undistorted image; for the SL right camera loads the raw image.

        Parameters
        ----------
        cam : str
        mode : str
        view_i : int
        light : str

        Returns
        -------
        img : torch.Tensor
            of shape [channels_n, h, w].
        """
        var = 'undist' if cam != 'stl_right' else 'raw'
        img = self.scene_paths.img(cam, mode, view_i, light, var)
        img = Image.open(img)
        img = np.asarray(img)
        img = torch.from_numpy(img)
        img = torch.atleast_3d(img).permute(2, 0, 1).div(255)
        img = img.expand(3, -1, -1)
        return img

    @ignore_warnings(['The given NumPy array is not writeable, and PyTorch'])
    def load_ir_img(self, cam, mode, view_i):
        r"""Loads the undistorted IR image; for RealSense loads the HDR IR image.

        Parameters
        ----------
        cam : str
        mode : str
        view_i : int

        Returns
        -------
        img : torch.Tensor
            of shape [h, w].
        """
        light = 'hdr' if (cam == 'real_sense') else None
        img = self.scene_paths.img(cam, mode, view_i, light)
        img = imgio.read[cam][mode](img)
        img = torch.from_numpy(img)
        return img

    def load_rec(self):
        r"""Loads full SL scan.

        Returns
        -------
        occ : o3d.geometry.TriangleMesh
        """
        rec = self.scene_paths.sl_full('cleaned')
        rec = o3d.io.read_triangle_mesh(rec)
        return rec

    def load_occ(self):
        r"""Loads the occluding surface.

        Returns
        -------
        occ : o3d.geometry.TriangleMesh
        """
        occ = self.scene_paths.sl_occ()
        occ = o3d.io.read_triangle_mesh(occ)
        return occ
