from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import torch

from skrgbd.calibration.camera_models.central_generic import CentralGeneric
from skrgbd.calibration.eth_tool.dataset import Dataset
from skrgbd.data.io import imgio
from skrgbd.utils.logging import logger, tqdm
from skrgbd.data.processing.depth_utils.mesh_rendering import MeshRenderer
from skrgbd.calibration.eth_tool.points import Points
from skrgbd.calibration.eth_tool.poses import Poses
from skrgbd.data.dataset.scene_paths import ScenePaths


def main():
    description = 'Calculates depth wall undistortion data.'
    parser = ArgumentParser(description=description)
    parser.add_argument('--calib-dir', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--cam', type=str, required=True)
    args = parser.parse_args()

    f'Calc wall undist data {args.cam}' >> logger.debug
    scene_paths = ScenePaths(None, data_dir=args.data_dir)
    calib_paths = CalibPaths(args.calib_dir, args.cam)
    calc_undist_data_wall(scene_paths, calib_paths, args.cam)


def calc_undist_data_wall(scene_paths, calib_paths, cam, device='cpu', dtype=torch.float32):
    r"""Calculates depth wall undistortion data.

    Parameters
    ----------
    scene_paths : ScenePaths
    calib_paths : CalibPaths
    cam : str
    device : torch.device
    dtype : torch.dtype
    """
    f'Load cam model for {cam}' >> logger.debug
    cam_model = scene_paths.cam_model(cam, 'ir', 'generic')
    cam_model = CentralGeneric(cam_model, dtype=dtype).to(device)

    'Load trajectory' >> logger.debug
    w2c, view_ids = load_traj(calib_paths)
    c2w = torch.from_numpy(w2c).inverse().to(device, dtype); del w2c

    'Load board model' >> logger.debug
    board = load_board(calib_paths.points())

    'Initialize renderer' >> logger.debug
    renderer = Renderer(board); del board
    renderer.set_rays_from_camera(cam_model); del cam_model

    'Render' >> logger.debug
    interm_data = {mode: [] for mode in ['i', 'j', 'd', 'ir', 'd_sl', 'cos']}
    for view_i, c2w_m in zip(tqdm(view_ids), c2w):
        # 'Render SL depth' >> logger.debug
        c2w_t, c2w_rot = c2w_m[:3, 3], c2w_m[:3, :3]; del c2w_m
        depthmap, cos, sl_is_valid = renderer.render(c2w_t, c2w_rot); del c2w_t, c2w_rot

        # 'Load sensor depth' >> logger.debug
        raw_depthmap = calib_paths.raw_depth(view_i)
        raw_depthmap = imgio.read[cam].raw_depth(raw_depthmap)
        raw_depthmap = torch.from_numpy(raw_depthmap).to(dtype).div_(1000)

        # 'Load sensor IR' >> logger.debug
        raw_ir = calib_paths.raw_ir(view_i)
        raw_ir = imgio.read[cam].ir(raw_ir)
        raw_ir = torch.from_numpy(raw_ir).to(dtype)

        raw_is_valid = raw_depthmap > .1
        is_valid = raw_is_valid.logical_and_(sl_is_valid); del raw_is_valid, sl_is_valid
        i, j = is_valid.nonzero(as_tuple=True); del is_valid

        interm_data['i'].append(i.numpy())
        interm_data['j'].append(j.numpy())
        interm_data['d'].append(raw_depthmap[i, j].numpy()); del raw_depthmap
        interm_data['ir'].append(raw_ir[i, j].numpy()); del raw_ir
        interm_data['d_sl'].append(depthmap[i, j]); del depthmap
        interm_data['cos'].append(cos[i, j]); del cos, i, j

    'Concatenate' >> logger.debug
    samples_n = sum(len(i) for i in interm_data['i'])
    undist_data = np.empty([samples_n], dtype=[('i', np.int64), ('j', np.int64), ('d', np.float32), ('ir', np.float32),
                                               ('d_sl', np.float32), ('cos', np.float32)])
    for mode in undist_data.dtype.fields:
        undist_data[mode] = np.concatenate(interm_data[mode]); del interm_data[mode]

    undist_data_npy = calib_paths.undist_data()
    f'Save undist data to {undist_data_npy}' >> logger.debug
    Path(undist_data_npy).parent.mkdir(exist_ok=True, parents=True)
    np.save(undist_data_npy, undist_data)
    'Done' >> logger.debug


class CalibPaths:
    def __init__(self, calib_dir, cam):
        self.calib_dir = calib_dir
        self.cam = cam
        self.wall_dir = f'{self.calib_dir}/raw_calibration/depth_wall_{cam}'
        self.raw_img_dir = f'{self.calib_dir}/images/depth_wall_{cam}/{cam}_raw'

    def dataset(self):
        return f'{self.wall_dir}/{self.cam}_ir/localization/dataset.bin'

    def points(self):
        return f'{self.wall_dir}/{self.cam}_ir/localization/points.yaml'

    def r2c(self):
        return f'{self.wall_dir}/{self.cam}_ir/localization/camera_tr_rig.yaml'

    def w2r(self):
        return f'{self.wall_dir}/{self.cam}_ir/localization/rig_tr_global.yaml'

    def raw_depth(self, view_i):
        return f'{self.raw_img_dir}/depth/{view_i:06}.png'

    def raw_ir(self, view_i):
        return f'{self.raw_img_dir}/ir/{view_i:06}.png'

    def undist_data(self):
        return f'{self.wall_dir}/{self.cam}_tensor.npy'


def load_traj(calib_paths, img_ext='.png'):
    r"""Loads localized wall camera poses and the respective image ids.

    Parameters
    ----------
    calib_paths : CalibPaths
    img_ext : str

    Returns
    -------
    w2c : np.ndarray
        of shape [views_n, 4, 4].
    view_ids : list of int
        of shape [views_n].
    """
    w2r, found_pose_ids = load_w2c(calib_paths.w2r())
    r2c, _ = load_w2c(calib_paths.r2c()); del _
    w2c = r2c[0] @ w2r; del r2c, w2r

    dataset = Dataset.fromfile(calib_paths.dataset())
    img_names = list(dataset.image_data.keys()); del dataset
    view_ids = [int(img_names[i][:-len(img_ext)]) for i in found_pose_ids]
    return w2c, view_ids


def load_w2c(poses_yaml):
    r"""Loads camera poses.

    Parameters
    ----------
    poses_yaml : str

    Returns
    -------
    w2c : np.ndarray
        of shape [cams_n, 4, 4].
    found_pose_ids : np.ndarray
        of shape [cams_n].
    """
    w2c_poses = Poses.from_yaml(poses_yaml)

    found_pose_ids = w2c_poses.is_def.nonzero()[0]
    w2c_t = w2c_poses.w2c_t[found_pose_ids]
    w2c_rotq = w2c_poses.w2c_rotq[found_pose_ids]; del w2c_poses
    w2c_rotm = Rotation.from_quat(w2c_rotq).as_matrix(); del w2c_rotq

    w2c = np.zeros([len(w2c_t), 4, 4], dtype=w2c_t.dtype)
    w2c[:, :3, 3] = w2c_t; del w2c_t
    w2c[:, :3, :3] = w2c_rotm; del w2c_rotm
    w2c[:, 3, 3] = 1
    return w2c, found_pose_ids


def load_board(pts_yaml, corner_pt_ids=(0, 39, 5), size_m=5):
    r"""Loads 3D model of the calibration board and makes a 3D square from it.

    Parameters
    ----------
    pts_yaml : str
    corner_pt_ids : list of int
        (bottom_left, bottom_right, top_left).
    size_m : float
        Side of the square, in meters.

    Returns
    -------
    mesh : o3d.geometry.TriangleMesh
    """
    pts = Points.from_yaml(pts_yaml).pts
    pts = pts[list(corner_pt_ids)]

    x_dir = pts[1] - pts[0]
    x_dir = x_dir / np.linalg.norm(x_dir)
    y_dir = pts[2] - pts[0]
    y_dir = y_dir / np.linalg.norm(y_dir)
    center = (pts[1] + pts[2]) / 2

    verts = np.array([- x_dir - y_dir, x_dir - y_dir, x_dir + y_dir, - x_dir + y_dir])
    verts = verts * size_m / 2 + center
    tris = [[0, 1, 2], [0, 2, 3]]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(tris, dtype=np.int32))
    return mesh


class Renderer(MeshRenderer):
    def render(self, c2w_t, c2w_rot):
        # 'Render' >> logger.debug
        c2w_rot = c2w_rot.to(self.rays)
        render = self.render_to_camera(c2w_t, c2w_rot,
                                       ['z_depth', 'world_normals', 'world_ray_dirs'], cull_back_faces=True)
        cos = torch.einsum('ijk,ijk->ij', render['world_ray_dirs'], render['world_normals'])
        depthmap = render['z_depth']; del render
        is_valid = depthmap.isfinite()
        depthmap = depthmap.where(is_valid, depthmap.new_tensor(float('nan')))
        return depthmap, cos, is_valid


if __name__ == '__main__':
    main()
