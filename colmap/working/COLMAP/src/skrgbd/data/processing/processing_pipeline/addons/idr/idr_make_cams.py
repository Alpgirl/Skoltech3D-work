from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from skrgbd.data.dataset.params import cam_pos_ids, sensor_to_cam_mode
from skrgbd.data.image_utils import get_trim
from skrgbd.data.io import imgio
from skrgbd.calibration.camera_models import load_from_colmap_txt
from skrgbd.data.io.poses import load_poses
from skrgbd.utils.logging import logger, tqdm
from skrgbd.data.dataset.scene_paths import ScenePaths


def main():
    description = 'Saves camera parameters in IDR format.'
    parser = ArgumentParser(description=description)
    parser.add_argument('--addons-dir', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--sensor', type=str, required=True)
    parser.add_argument('--scene-name', type=str, required=True)
    args = parser.parse_args()

    f'Make IDR cams {args.scene_name}' >> logger.debug
    scene_paths = ScenePaths(args.scene_name, addons_dir=args.addons_dir, data_dir=args.data_dir)
    make_cams(scene_paths, args.sensor)


def make_cams(scene_paths, sensor, dtype=torch.double):
    r"""Saves camera parameters in IDR format
        https://github.com/lioryariv/idr/blob/main/DATA_CONVENTION.md.

    Parameters
    ----------
    scene_paths : ScenePaths
    sensor : str
    dtype : torch.dtype

    Notes
    -----
    The cameras file in IDR format is a NumPy .npz file containing:
        camera_mat_{view_i}: nd.ndarray
            of shape [4, 4], camera model matrix, for view_i in [0..99].
        world_mat_{view_i}: nd.ndarray
            of shape [4, 4], camera projection matrix.
        scale_mat_{view_i}: nd.ndarray
            of shape [4, 4], normalization matrix.
        camera_mat_inv_{view_i}: nd.ndarray
        world_mat_inv_{view_i}: nd.ndarray
        scale_mat_inv_{view_i}: nd.ndarray
            Inverses of these matrices.
        roi_box_{view_i}: tuple
            (left, right, top, bottom) bounding box of the object (projection of the SL scan) in the image in pixels.
    """
    cam, mode = sensor_to_cam_mode[sensor]
    f'Load cam data for {cam}.{mode}' >> logger.debug
    cam_model = scene_paths.cam_model(cam, mode)
    cam_model = load_from_colmap_txt(cam_model).to(dtype)

    w2c = scene_paths.cam_poses(cam, mode)
    w2c = load_poses(w2c, dtype=dtype)

    'Compute cam matrices' >> logger.debug
    camera_mat = torch.eye(4, dtype=dtype)
    camera_mat[[0, 1], [0, 1]] = cam_model.focal
    camera_mat[[0, 1], [2, 2]] = cam_model.principal; del cam_model
    camera_mat_inv = camera_mat.inverse()

    world_mat = camera_mat @ w2c; del w2c
    world_mat_inv = world_mat.inverse()

    'Load SL data' >> logger.debug
    rec = scene_paths.sl_full()
    rec = o3d.io.read_triangle_mesh(rec)

    'Compute scaling' >> logger.debug
    center = rec.get_axis_aligned_bounding_box().get_center()
    verts = np.asarray(rec.vertices); del rec
    radius = np.linalg.norm(verts - center, axis=-1).max(); del verts
    scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32); del radius
    scale_mat[:3, 3] = center; del center
    scale_mat_inv = np.linalg.inv(scale_mat)

    cams_dict = dict()
    for view_i in cam_pos_ids:
        cams_dict[f'camera_mat_{view_i}'] = camera_mat.float().numpy()
        cams_dict[f'camera_mat_inv_{view_i}'] = camera_mat_inv.float().numpy()
        cams_dict[f'world_mat_{view_i}'] = world_mat[view_i].float().numpy()
        cams_dict[f'world_mat_inv_{view_i}'] = world_mat_inv[view_i].float().numpy()
        cams_dict[f'scale_mat_{view_i}'] = scale_mat
        cams_dict[f'scale_mat_inv_{view_i}'] = scale_mat_inv

    'Compute bounding boxes' >> logger.debug
    src, src_var = 'stl', 'clean_rec'
    for view_i in tqdm(cam_pos_ids):
        depthmap = scene_paths.proj_depth(src, src_var, sensor, 'undist', view_i)
        depthmap = imgio.read.stl.depth(depthmap)
        is_valid = torch.from_numpy(depthmap).isfinite(); del depthmap
        i_min, i_max, j_min, j_max = get_trim(is_valid); del is_valid
        cams_dict[f'roi_box_{view_i}'] = j_min, j_max, i_min, i_max

    cameras_npz = scene_paths.idr_cams(cam, mode)
    f'Save cams to {cameras_npz}' >> logger.debug
    Path(cameras_npz).parent.mkdir(exist_ok=True, parents=True)
    np.savez(cameras_npz, **cams_dict)
    'Done' >> logger.debug


if __name__ == '__main__':
    main()
