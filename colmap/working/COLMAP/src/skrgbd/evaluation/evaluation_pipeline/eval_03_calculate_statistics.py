from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import yaml

import skrgbd
from skrgbd.evaluation.statistics.calculate_distance_stats import calc_rec_to_ref_stats, calc_ref_to_rec_stats
from skrgbd.evaluation.evaluation_pipeline.configs import configs
from skrgbd.evaluation.pathfinder import eval_pathfinder
from skrgbd.utils.logging import get_git_revision, logger
from skrgbd.data.dataset.pathfinder import pathfinder
from skrgbd.evaluation.stats_db import StatsDB


def main():
    description = 'Calculates statistics of distances from reconstruction to reference surface and vice versa.'

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

    f'Eval 03 {args.scene_name}' >> logger.debug
    pathfinder.set_dirs(data_root=args.dataset_dir)
    eval_pathfinder.set_dirs(results_root=args.results_dir)

    calc_stats(args.method, args.version, args.camera, args.light_setup, args.scene_name)


def calc_stats(method, version, cam, light_setup, scene_name):
    r"""Calculates statistics of distances from reconstruction to reference surface and vice versa and saves them to disk.

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

    'Load SL reconstruction' >> logger.debug
    ref = pathfinder[scene_name].stl.reconstruction.cleaned
    ref = o3d.io.read_triangle_mesh(ref)
    ref_verts = torch.from_numpy(np.asarray(ref.vertices)).T.to(dtype); del ref

    'Load distance distributions' >> logger.debug
    distribs = eval_pathfinder.evaluation[method](eval_version, scene_name, cam, light_setup).distributions.data
    distribs = torch.load(distribs)
    if 'rec_pt_normals' in distribs:
        del distribs['rec_pt_normals']

    'Calculate ref_to_rec stats' >> logger.debug
    thresholds = torch.linspace(*config.thres_range, dtype=dtype)
    comp, mean_ref_to_rec = calc_ref_to_rec_stats(ref_verts, distribs['dist_from_ref'],
                                                  thresholds, config.stat_cell_size, config.max_dist)
    del ref_verts, distribs['dist_from_ref']

    'Load method reconstruction' >> logger.debug
    if 'rec_pts' in distribs:
        rec_pts = distribs['rec_pts'].T; del distribs['rec_pts']
    else:
        rec_paths = eval_pathfinder.reconstructions[method](version, scene_name, cam, light_setup)
        if 'mesh' in rec_paths:
            rec = o3d.io.read_triangle_mesh(rec_paths.mesh)
        elif 'points' in rec_paths:
            rec = o3d.io.read_point_cloud(rec_paths.points)
        rec_pts = torch.from_numpy(np.asarray(rec.points)).T.to(dtype); del rec
    rec_pts = rec_pts[:, distribs['visible_rec_pt_ids']]; del distribs['visible_rec_pt_ids']

    'Calculate rec_to_ref stats' >> logger.debug
    acc, mean_rec_to_ref = calc_rec_to_ref_stats(rec_pts, distribs['dist_to_ref'], distribs['dist_to_occ'], thresholds,
                                                 config.stat_cell_size, config.max_dist, occ_eps=config.occ_eps)
    del rec_pts, distribs['dist_to_ref'], distribs['dist_to_occ']

    meta_yaml = eval_pathfinder.evaluation[method](eval_version, scene_name, cam, light_setup).stats.meta
    f'Save meta to {meta_yaml}' >> logger.debug
    Path(meta_yaml).parent.mkdir(exist_ok=True, parents=True)
    meta = dict(method=method, version=eval_version, cam=cam, light_setup=light_setup,
                scene_name=scene_name, config=dict(config))
    meta['skrgbd.rev'] = get_git_revision(Path(skrgbd.__file__).parent)
    with open(meta_yaml, 'w') as file:
        yaml.dump(meta, file)

    stats_db = eval_pathfinder.evaluation[method](eval_version, scene_name, cam, light_setup).stats.data
    f'Save stats to {stats_db}' >> logger.debug
    stats_db = StatsDB(stats_db, mode='w')
    for measure, value in [('mean_ref_to_rec', mean_ref_to_rec), ('mean_rec_to_ref', mean_rec_to_ref)]:
        stats_db.set_measure(measure, method, eval_version, scene_name, cam, light_setup, value)
    thresholds = thresholds.tolist()
    comp = comp.tolist()
    acc = acc.tolist()
    for measure, values in [('completeness', comp), ('accuracy', acc)]:
        stats_db.set_measure(measure, method, eval_version, scene_name, cam, light_setup, values, thresholds)


if __name__ == '__main__':
    main()
