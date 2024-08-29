from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from skrgbd.data.dataset.params import cam_pos_ids
from skrgbd.calibration.camera_models import load_from_colmap_txt
from skrgbd.utils.logging import logger, tqdm
from skrgbd.data.processing.depth_utils.occluded_mesh_rendering import MeshRenderer
from skrgbd.data.dataset.pathfinder import Pathfinder, sensor_to_cam_mode
from skrgbd.data.io.poses import Poses
from skrgbd.data.dataset.dataset import wip_scene_name_by_id


def main():
    description = r"""This script creates pair.txt files for running MVSNets on the dataset."""
    name = 'Main'
    parser = ArgumentParser(description=description)
    parser.add_argument('--processed-scans-dir', type=str, required=True)
    parser.add_argument('--sensor', type=str, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--scene-name', type=str)
    group.add_argument('--scene-i', type=int)
    parser.add_argument('--progress', action='store_true')
    args = parser.parse_args()

    scene_name = args.scene_name if args.scene_name else (wip_scene_name_by_id[args.scene_i])
    logger.info(f'{name}: Scene name {scene_name}')
    pathfinder = Pathfinder(data_root=args.processed_scans_dir)
    create_pair_files(scene_name, args.sensor, pathfinder, show_progress=args.progress)


def create_pair_files(scene_name, sensor, pathfinder, occ_threshold=1e-3, samples_n=10_240, target_baseline_angle_deg=5,
                      pos_angle_sigma_deg=10, neg_angle_sigma_deg=1, src_views_n=10, show_progress=True):
    name = 'CreatePairFiles'
    show_progress = tqdm if show_progress else (lambda x: x)
    cam, mode = sensor_to_cam_mode[sensor]
    dtype = torch.float

    logger.info(f'{name}: Load camera model: {sensor}.undistorted')
    cam_model = pathfinder[scene_name][cam][mode].pinhole_intrinsics
    cam_model = load_from_colmap_txt(cam_model).to(dtype=dtype)

    logger.info(f'{name}: Load camera poses')
    poses = pathfinder[scene_name][cam][mode].refined_extrinsics
    poses = Poses.from_colmap(poses, dtype)
    world_to_cam = torch.empty(len(cam_pos_ids), 4, 4, dtype=dtype)
    for pos_i in cam_pos_ids:
        world_to_cam[pos_i].copy_(poses[pos_i + 1])  # COLMAP's image_id is one-based
    del poses

    logger.info(f'{name}: Load scan data')
    rec = pathfinder[scene_name].stl.reconstruction.cleaned
    rec = o3d.io.read_triangle_mesh(rec)
    occ = pathfinder[scene_name].stl.occluded_space
    occ = o3d.io.read_triangle_mesh(occ)
    renderer = MeshRenderer(rec, occ, occ_threshold); del occ

    logger.info(f'{name}: Sample points')
    samples = rec.sample_points_uniformly(samples_n); del rec
    samples = torch.from_numpy(np.asarray(samples.points)).to(dtype)

    logger.info(f'{name}: Calculate sample visibility')
    sample_is_visible = torch.empty(len(cam_pos_ids), samples_n, dtype=torch.bool)
    for view_i in show_progress(cam_pos_ids):
        sample_is_visible[view_i] = calculate_sample_visibility(samples, world_to_cam[view_i], cam_model, renderer)
    del renderer

    logger.info(f'{name}: Calculate scores')
    score_matrix = torch.zeros(len(cam_pos_ids), len(cam_pos_ids), dtype=dtype)
    for view_i, view_j in show_progress(list(combinations(cam_pos_ids, 2))):
        visible_samples = samples[sample_is_visible[view_i].logical_and(sample_is_visible[view_j])]
        cam_center_i = world_to_cam[view_i].inverse()[:3, 3]
        cam_center_j = world_to_cam[view_j].inverse()[:3, 3]
        score = calculate_score(visible_samples, cam_center_i, cam_center_j,
                                target_baseline_angle_deg, pos_angle_sigma_deg, neg_angle_sigma_deg)
        del visible_samples, cam_center_i, cam_center_j
        score_matrix[view_i, view_j] = score_matrix[view_j, view_i] = score; del score

    logger.info(f'{name}: Save pair file')
    pair_txt = pathfinder[scene_name][cam].mvsnet_input.pair_txt
    Path(pair_txt).parent.mkdir(exist_ok=True, parents=True)
    with open(pair_txt, 'w') as pair_txt:
        pair_txt.write(f'{len(cam_pos_ids)}\n')
        for view_i in show_progress(cam_pos_ids):
            pair_txt.write(f'{view_i}\n')
            top_scores, src_view_ids = score_matrix[view_i].topk(src_views_n)
            scores_str = f'{src_views_n} ' + ' '.join(
                f'{view_j} {score:f}' for (score, view_j) in zip(top_scores, src_view_ids)
            ) + '\n'
            pair_txt.write(scores_str)


def calculate_sample_visibility(samples, world_to_cam, cam_model, renderer):
    cam_center = world_to_cam.inverse()[:3, 3]
    rays_world = samples - cam_center

    casted_rays = torch.empty([len(samples), 6], device=renderer.device, dtype=renderer.dtype)
    casted_rays[:, :3] = cam_center; del cam_center
    casted_rays[:, 3:6] = rays_world; del rays_world
    result = renderer.render_rays(casted_rays, cull_back_faces=True); del casted_rays
    not_occluded = result['ray_hit_depth'].isfinite(); del result

    samples_cam = world_to_cam[:3, :3] @ samples.T + world_to_cam[:3, 3].unsqueeze(1); del world_to_cam, samples
    uv = cam_model.project(samples_cam); del samples_cam
    in_bounds = (uv >= 0).all(0).logical_and_((uv < cam_model.size_wh.unsqueeze(1)).all(0)); del uv

    sample_is_visible = not_occluded.logical_and_(in_bounds)
    return sample_is_visible


def calculate_score(visible_samples, cam_center_i, cam_center_j, target_baseline_angle_deg=5,
                    pos_angle_sigma_deg=10, neg_angle_sigma_deg=1):
    sample_to_i = torch.nn.functional.normalize(cam_center_i - visible_samples, dim=1); del cam_center_i
    sample_to_j = torch.nn.functional.normalize(cam_center_j - visible_samples, dim=1); del cam_center_j, visible_samples
    cosines = (sample_to_i.unsqueeze(1) @ sample_to_j.unsqueeze(2)).squeeze(); del sample_to_i, sample_to_j
    baseline_angles = cosines.acos_(); del cosines
    angle_devs = baseline_angles.sub_(np.deg2rad(target_baseline_angle_deg)); del baseline_angles
    pos_scaled_devs = angle_devs / np.deg2rad(pos_angle_sigma_deg)
    neg_scaled_devs = angle_devs / np.deg2rad(neg_angle_sigma_deg); del angle_devs
    scaled_devs = pos_scaled_devs.where(pos_scaled_devs >= 0, neg_scaled_devs); del pos_scaled_devs, neg_scaled_devs
    scores = scaled_devs.pow_(2).div_(2).neg_().exp_(); del scaled_devs
    tot_score = scores.sum(); del scores
    return tot_score


if __name__ == '__main__':
    main()
