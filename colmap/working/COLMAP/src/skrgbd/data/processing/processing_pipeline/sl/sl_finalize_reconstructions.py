from argparse import ArgumentParser

import numpy as np
import open3d as o3d

from skrgbd.utils.logging import logger
from skrgbd.data.io.ply import save_ply
from skrgbd.data.dataset.scene_paths import ScenePaths


def main():
    description = 'Cleans SL reconstructions.'
    parser = ArgumentParser(description=description)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--scene-name', type=str)
    args = parser.parse_args()

    f'SL clean reconstructions {args.scene_name}' >> logger.debug
    scene_paths = ScenePaths(args.scene_name, data_dir=args.data_dir)
    clean_rec(scene_paths)


def clean_rec(scene_paths):
    r"""Cleans SL reconstruction.

    Parameters
    ----------
    scene_paths : ScenePaths
    """
    'Load rec' >> logger.debug
    cleaned_ply = scene_paths.sl_full('cleaned')

    'Clean' >> logger.debug
    rec = o3d.io.read_triangle_mesh(cleaned_ply)
    rec = rec.remove_duplicated_vertices()
    rec = rec.remove_degenerate_triangles()
    rec = rec.remove_duplicated_triangles()
    rec = rec.remove_unreferenced_vertices()

    'Save' >> logger.debug
    verts = np.asarray(rec.vertices)
    tris = np.asarray(rec.triangles)
    save_ply(cleaned_ply, verts, tris)

    'Done' >> logger.debug


if __name__ == '__main__':
    main()
