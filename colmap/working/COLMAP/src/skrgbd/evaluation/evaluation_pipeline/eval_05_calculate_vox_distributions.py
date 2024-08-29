from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import yaml

import skrgbd
from skrgbd.evaluation.statistics.calculate_vox_dists import calc_vox_accuracy, calc_vox_completeness
from skrgbd.evaluation.evaluation_pipeline.configs import configs
from skrgbd.evaluation.pathfinder import Pathfinder as EvalPathfinder
from skrgbd.utils.logging import get_git_revision, logger as glob_logger
from skrgbd.data.dataset.pathfinder import Pathfinder
from skrgbd.evaluation.io import read_rec_for_vis
from skrgbd.data.dataset.dataset import wip_scene_name_by_id


def main():
    description = r"""FIXME"""
    logger = glob_logger.get_context_logger('Main')

    parser = ArgumentParser(description=description)
    parser.add_argument('--processed-scans-dir', type=str, required=True)
    parser.add_argument('--results-dir', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--camera', type=str, required=True)
    parser.add_argument('--light-setup', type=str)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--scene-name', type=str)
    group.add_argument('--scene-i', type=int)
    args = parser.parse_args()

    scene_name = args.scene_name if args.scene_name else (wip_scene_name_by_id[args.scene_i])

    f'Scene name is {scene_name}' >> logger.info
    pathfinder = Pathfinder(data_root=args.processed_scans_dir)
    eval_pathfinder = EvalPathfinder(args.results_dir)

    calc_vox_dists(args.method, args.version, args.camera, args.light_setup, scene_name, pathfinder, eval_pathfinder)


def calc_vox_dists(method, version, cam, light_setup, scene_name, pathfinder, eval_pathfinder):
    r"""FIXME add desc

    Parameters
    ----------
    method : str
    version : str
    cam : str
    light_setup : str
    scene_name : str
    pathfinder : Pathfinder
    eval_pathfinder : EvalPathfinder
    """
    logger = glob_logger.get_context_logger('CalcAndSaveStats')
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

    'Calculate vox completeness' >> logger.debug
    comp = calc_vox_completeness(ref_verts, distribs['dist_from_ref'], config['vox_dists_thres'],
                                 config.stat_cell_size, config.max_dist)
    del ref_verts, distribs['dist_from_ref']

    'Load method reconstruction' >> logger.debug
    if 'rec_pts' in distribs:
        rec_pts = distribs['rec_pts'].T; del distribs['rec_pts']
    else:
        rec = eval_pathfinder.reconstructions[method](version, scene_name, cam, light_setup).reconstruction
        rec = read_rec_for_vis[method](rec)
        rec_pts = torch.from_numpy(np.asarray(rec.points)).T.to(dtype); del rec
    rec_pts = rec_pts[:, distribs['visible_rec_pt_ids']]; del distribs['visible_rec_pt_ids']

    'Calculate vox accuracy' >> logger.debug
    acc = calc_vox_accuracy(rec_pts, distribs['dist_to_ref'], distribs['dist_to_occ'], config['vox_dists_thres'],
                            config.stat_cell_size, config.max_dist, occ_eps=config.occ_eps)
    del rec_pts, distribs['dist_to_ref'], distribs['dist_to_occ']

    meta_yaml = eval_pathfinder.evaluation[method](eval_version, scene_name, cam, light_setup).vox_distributions.meta
    f'Save meta to {meta_yaml}' >> logger.debug
    Path(meta_yaml).parent.mkdir(exist_ok=True, parents=True)
    meta = dict(method=method, version=eval_version, cam=cam, light_setup=light_setup,
                scene_name=scene_name, config=dict(config))
    meta['skrgbd.rev'] = get_git_revision(Path(skrgbd.__file__).parent)
    with open(meta_yaml, 'w') as file:
        yaml.dump(meta, file)

    vox_distribs = dict(completeness=comp, accuracy=acc)
    vox_distribs_pt = eval_pathfinder.evaluation[method](eval_version, scene_name, cam, light_setup).vox_distributions.data
    f'Save vox distribs to {vox_distribs_pt}' >> logger.debug
    torch.save(vox_distribs, vox_distribs_pt)


if __name__ == '__main__':
    main()
