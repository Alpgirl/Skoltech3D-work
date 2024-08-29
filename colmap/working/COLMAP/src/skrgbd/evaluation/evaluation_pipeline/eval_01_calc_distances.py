from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import yaml

import skrgbd
from skrgbd.evaluation.distance.calculate_distances import calc_dists
from skrgbd.calibration.calibrations.small_scale_sphere import Calibration
from skrgbd.evaluation.evaluation_pipeline.configs import configs
from skrgbd.evaluation.pathfinder import eval_pathfinder
from skrgbd.utils.logging import get_git_revision, logger
from skrgbd.calibration.camera_models import load_from_colmap_txt
from skrgbd.data.dataset.pathfinder import pathfinder


def main():
    description = 'Calculates distances from reconstruction to reference and vice versa.'

    parser = ArgumentParser(description=description)
    parser.add_argument('--aux-dir', type=str, required=True)
    parser.add_argument('--calib-dir', type=str, required=True)
    parser.add_argument('--dataset-dir', type=str, required=True)
    parser.add_argument('--results-dir', type=str, required=True)

    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--camera', type=str, required=True)
    parser.add_argument('--light-setup', type=str)

    parser.add_argument('--scene-name', type=str, required=True)
    parser.add_argument('--progress', action='store_true')
    args = parser.parse_args()

    if args.light_setup == 'none':
        args.light_setup = None

    f'Eval 01 {args.scene_name}' >> logger.debug
    calibration = Calibration(args.calib_dir)
    pathfinder.set_dirs(data_root=args.dataset_dir, aux_root=args.aux_dir)
    eval_pathfinder.set_dirs(results_root=args.results_dir)
    calc_and_save_dists(
        args.method, args.version, args.camera, args.light_setup, args.scene_name, calibration, args.progress)


def calc_and_save_dists(method, version, cam, light_setup, scene_name, calibration, show_progress=True):
    r"""Loads reconstruction and reference data, calculates distance from reconstruction to reference mesh
    and occluded space, and distance from reference mesh to reconstruction, and saves the distribs to disk.

    Parameters
    ----------
    method : str
    version : str
    cam : str
    light_setup : str
    scene_name : str
    calibration : Calibration
    show_progress : bool
    """
    dtype = torch.float

    'Load camera model: stl_right.undist' >> logger.debug
    cam_model = pathfinder.stl_right.rgb.pinhole_intrinsics
    cam_model = load_from_colmap_txt(cam_model).to(dtype)

    'Load camera poses' >> logger.debug
    cam_to_world = get_stl_right_cam_to_world(calibration, pathfinder, scene_name)
    world_to_cam = cam_to_world.inverse().to(dtype); del cam_to_world

    'Load SL reconstruction' >> logger.debug
    ref = pathfinder[scene_name].stl.reconstruction.cleaned
    ref = o3d.io.read_triangle_mesh(ref)
    occ = pathfinder[scene_name].stl.occluded_space
    occ = o3d.io.read_triangle_mesh(occ)

    'Load method reconstruction' >> logger.debug
    rec_paths = eval_pathfinder.reconstructions[method](version, scene_name, cam, light_setup)
    if 'mesh' in rec_paths:
        rec = o3d.io.read_triangle_mesh(rec_paths.mesh)
    elif 'points' in rec_paths:
        rec = o3d.io.read_point_cloud(rec_paths.points)
        rec = np.asarray(rec.points).astype(np.float32)

    'Load config' >> logger.debug
    config = configs[method][cam, light_setup][version]
    eval_version = f'{version}_{config.config_ver}'

    'Calculate' >> logger.debug
    distribs = calc_dists(
        rec, ref, occ, cam_model, world_to_cam, config.occ_threshold, config.max_visible_depth, config.max_dist,
        config.max_edge_len, config.max_dist_to_sample, show_progress=show_progress
    )

    distrib_paths = eval_pathfinder.evaluation[method](eval_version, scene_name, cam, light_setup).distributions
    meta_yaml = distrib_paths.meta
    f'Save meta to {meta_yaml}' >> logger.debug
    Path(meta_yaml).parent.mkdir(exist_ok=True, parents=True)
    meta = dict(method=method, version=eval_version, cam=cam, light_setup=light_setup,
                scene_name=scene_name, config=dict(config))
    meta['skrgbd.rev'] = get_git_revision(Path(skrgbd.__file__).parent)
    with open(meta_yaml, 'w') as file:
        yaml.dump(meta, file)

    distribs_pt = distrib_paths.data
    f'Save distribs to {distribs_pt}' >> logger.debug
    torch.save(distribs, distribs_pt)


def get_stl_right_cam_to_world(calibration, pathfinder, scene_name, dtype=torch.double):
    r"""Calculates positions of the right SL camera during scanning.

    Parameters
    ----------
    calibration : Calibration
    pathfinder : Pathfinder
    scene_name : str
    dtype : torch.dtype

    Returns
    -------
    cam_to_world : torch.Tensor
        of shape [sl_scans_n, 4, 4].
    """
    cam_to_board = calibration.rv_calib_to_stl_right.to(dtype).inverse(); del calibration
    board_to_world = pathfinder[scene_name].stl.partial.aligned.refined_board_to_world
    board_to_world = torch.load(board_to_world).to(dtype)
    cam_to_world = board_to_world @ cam_to_board
    return cam_to_world


if __name__ == '__main__':
    main()
