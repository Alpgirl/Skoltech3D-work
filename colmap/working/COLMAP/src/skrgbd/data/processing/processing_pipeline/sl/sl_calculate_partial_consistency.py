from argparse import ArgumentParser
from pathlib import Path

import meshio
import numpy as np
import matplotlib.pyplot as plt

from skrgbd.data.processing.alignment.scan_alignment import Scan
from skrgbd.utils.logging import logger, tqdm
from skrgbd.data.dataset.scene_paths import ScenePaths


def main():
    description = 'Calculates consistency metrics for reference partial SL scans'

    parser = ArgumentParser(description=description)
    parser.add_argument('--aux-dir', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--scene-name', type=str, required=True)
    args = parser.parse_args()

    f'SL calc stats {args.scene_name}' >> logger.debug
    scene_paths = ScenePaths(args.scene_name, data_dir=args.data_dir, aux_dir=args.aux_dir)
    calc_stats(scene_paths)


def calc_stats(scene_paths, max_dist=3e-3, max_dist_vis=3e-4, scan_ids=None, cmap=plt.cm.jet):
    r"""Calculates consistency metrics for reference partial SL scans.

    Parameters
    ----------
    scene_paths : ScenePaths
    max_dist_vis : float
    scan_ids : iterable of int
    cmap
    """
    if scan_ids is None:
        scan_ids = range(27)

    'Load scans' >> logger.debug
    scan_datas = dict()
    scans = dict()
    for scan_i in tqdm(scan_ids):
        scan = scene_paths.sl_part(scan_i)
        scan = meshio.read(scan)
        scan_datas[scan_i] = scan
        scan = Scan(scan.points, scan.cells[0].data, None, None, None)
        scans[scan_i] = scan; del scan

    for scan_i in tqdm(scan_ids):
        scan = scans[scan_i]
        pts = scan.verts.to(scan.normals)

        'Find closest pts' >> logger.debug
        dists = pts.new_empty(len(scan_ids), len(pts))
        for j, scan_j in tqdm(list(enumerate(scan_ids)), leave=False):
            closest_pts, _ = scans[scan_j].find_closest(pts); del _
            dists[j] = (pts - closest_pts).norm(dim=1); del closest_pts
        del pts

        'Calculate stats' >> logger.debug
        is_close = dists <= max_dist
        dists = dists.where(is_close, dists.new_tensor(float('nan'))); del is_close
        mean_dists = dists.nanmedian(dim=0)[0]; del dists

        cols = (mean_dists / max_dist_vis).clamp(0, 1); del mean_dists
        cols = cmap(cols.numpy())[:, :3]
        cols = np.clip(cols * 255, 0, 255).astype(np.uint8)

        'Save stats' >> logger.debug
        scan = scan_datas[scan_i]
        scan.point_data = dict(red=cols[:, 0], green=cols[:, 1], blue=cols[:, 2])
        stats_ply = scene_paths.sl_part_stats(scan_i)
        Path(stats_ply).parent.mkdir(parents=True, exist_ok=True)
        meshio.write(stats_ply, scan)
    'Finished' >> logger.debug


if __name__ == '__main__':
    main()
