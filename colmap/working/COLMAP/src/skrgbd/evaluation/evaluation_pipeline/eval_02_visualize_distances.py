from argparse import ArgumentParser
from pathlib import Path

from PIL import Image
import numpy as np
import open3d as o3d
import torch
import yaml

import skrgbd
from skrgbd.evaluation.evaluation_pipeline.configs import configs
from skrgbd.evaluation.pathfinder import eval_pathfinder
from skrgbd.utils.logging import get_git_revision, logger
from skrgbd.calibration.camera_models import load_from_colmap_txt
from skrgbd.data.io.poses import load_poses
from skrgbd.data.dataset.pathfinder import pathfinder
from skrgbd.evaluation.visualization.visualize_distances import setup_pov_cam, visualize_distances


def main():
    description = 'Visualizes distances from reconstruction to reference surface and vice versa.'

    parser = ArgumentParser(description=description)
    parser.add_argument('--dataset-dir', type=str, required=True)
    parser.add_argument('--results-dir', type=str, required=True)

    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--camera', type=str, required=True)
    parser.add_argument('--light-setup', type=str)

    parser.add_argument('--scene-name', type=str, required=True)
    args = parser.parse_args()

    if args.light_setup == 'none':
        args.light_setup = None

    f'Eval 02 {args.scene_name}' >> logger.debug
    pathfinder.set_dirs(data_root=args.dataset_dir)
    eval_pathfinder.set_dirs(results_root=args.results_dir)

    vis_and_save_dists(args.method, args.version, args.camera, args.light_setup, args.scene_name)


def vis_and_save_dists(method, version, cam, light_setup, scene_name):
    r"""Visualizes distances from reconstruction to reference surface and vice versa and saves visualizations to disk.

    Parameters
    ----------
    method : str
    version : str
    cam : str
    light_setup : str
    scene_name : str
    """
    dtype = torch.float

    'Load config' >> logger.debug
    config = configs[method][cam, light_setup][version]
    eval_version = f'{version}_{config.config_ver}'

    'Load camera poses' >> logger.debug
    world_to_cam = pathfinder[scene_name].tis_right.rgb.refined_extrinsics
    world_to_cam = load_poses(world_to_cam, dtype)[config.vis_view_i]

    'Load SL reconstruction' >> logger.debug
    ref = pathfinder[scene_name].stl.reconstruction.cleaned
    ref = o3d.io.read_triangle_mesh(ref)
    ref = ref.transform(world_to_cam)

    'Setup camera' >> logger.debug
    cam_model = pathfinder.tis_right.rgb.pinhole_intrinsics
    cam_model = load_from_colmap_txt(cam_model).to(dtype)
    cam_model, crop_left_top = setup_pov_cam(cam_model, torch.from_numpy(np.asarray(ref.vertices)).T.to(dtype),
                                             config.vis_pov_wh, config.vis_resolution_wh)

    'Load distance distributions' >> logger.debug
    distribs = eval_pathfinder.evaluation[method](eval_version, scene_name, cam, light_setup).distributions.data
    distribs = torch.load(distribs)

    'Load method reconstruction' >> logger.debug
    if 'rec_pts' in distribs:
        rec = distribs['rec_pts']
        rec_normals = distribs['rec_pt_normals']
    else:
        rec_paths = eval_pathfinder.reconstructions[method](version, scene_name, cam, light_setup)
        if 'mesh' in rec_paths:
            rec = o3d.io.read_triangle_mesh(rec_paths.mesh)
        elif 'points' in rec_paths:
            rec = o3d.io.read_point_cloud(rec_paths.points)
        rec_normals = torch.from_numpy(np.asarray(rec.normals)).to(dtype)
        rec = torch.from_numpy(np.asarray(rec.points)).to(dtype)
    rec = rec @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
    rec_normals = rec_normals @ world_to_cam[:3, :3].T; del world_to_cam

    'Visualize' >> logger.debug
    results = visualize_distances(
        ref, distribs['dist_from_ref'], rec, rec_normals, distribs['visible_rec_pt_ids'], distribs['dist_to_ref'],
        distribs['dist_to_occ'], cam_model, config.z_near, config.z_far,
        config.occ_eps, config.max_dist, config.dist_range, config.color_range, cmap=config.cmap,
        splat_size=config.splat_size, crop=config.crop, margin=config.margin,
        draw_ref=config.draw_ref, two_sided=config.two_sided,
    )

    crop_left_top = crop_left_top + results['crop_left_top']

    meta_yaml = eval_pathfinder.evaluation[method](eval_version, scene_name, cam, light_setup).visualizations.meta
    f'Save meta to {meta_yaml}' >> logger.debug
    Path(meta_yaml).parent.mkdir(exist_ok=True, parents=True)
    meta = dict(method=method, version=eval_version, cam=cam, light_setup=light_setup,
                scene_name=scene_name, config=dict(config), crop_left_top=crop_left_top.tolist())
    meta['skrgbd.rev'] = get_git_revision(Path(skrgbd.__file__).parent)
    with open(meta_yaml, 'w') as file:
        yaml.dump(meta, file)

    for vis_name in ['completeness', 'accuracy', 'surf_accuracy', 'reference', 'reconstruction']:
        if vis_name not in results:
            continue
        img_png = eval_pathfinder.evaluation[method](eval_version, scene_name, cam, light_setup).visualizations[vis_name]
        f'Save {vis_name} to {img_png}' >> logger.debug
        img = results[vis_name]
        img = img.mul(255).clamp(0, 255).byte().numpy()
        Image.fromarray(img).save(img_png)


if __name__ == '__main__':
    main()
