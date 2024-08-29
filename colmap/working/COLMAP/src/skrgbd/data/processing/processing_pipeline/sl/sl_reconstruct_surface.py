from argparse import ArgumentParser
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import torch

from skrgbd.utils.logging import logger, tqdm
from skrgbd.data.io.ply import save_ply
from skrgbd.data.dataset.scene_paths import ScenePaths


def main():
    description = 'Reconstructs surface from partial SL scans using Screened Poisson Reconstruction.'
    parser = ArgumentParser(description=description)
    parser.add_argument('--aux-dir', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--poisson-recon-bin', type=str, required=True)
    parser.add_argument('--threads-n', type=int)
    parser.add_argument('--scene-name', type=str)
    args = parser.parse_args()

    f'SL reconstruct surface {args.scene_name}' >> logger.debug
    scene_paths = ScenePaths(args.scene_name, aux_dir=args.aux_dir, data_dir=args.data_dir)
    reconstruct_surface(scene_paths, args.poisson_recon_bin, threads_n=args.threads_n)


def reconstruct_surface(
        scene_paths, poisson_rec_bin, cell_width=3e-4, scan_ids=None, tmpdir='/tmp', threads_n=None, max_dist=3e-3,
        trim_dist=6e-4, min_comp_area=1e-2 ** 2, cmap=plt.cm.jet, dtype=torch.float32,
):
    r"""Reconstructs surface from partial SL scans using Screened Poisson Reconstruction.

    Parameters
    ----------
    scene_paths : ScenePaths
    poisson_rec_bin : str
    cell_width : float
    scan_ids : iterable of int
    tmpdir : str
    threads_n : int
    max_dist : float
    trim_dist : float
    min_comp_area : float
    cmap
    dtype : torch.dtype

    Notes
    -----
    The code was tested with Version 13.72
        https://github.com/mkazhdan/PoissonRecon/tree/65cd5fcadd403c69a15484436016de7177a00c91
    The default parameters were picked experimentally: cell width 0.3 mm and defaults from the official repo.
    """
    if scan_ids is None:
        scan_ids = range(27)

    'Load scans' >> logger.debug
    scans = []
    union_of_scans = o3d.geometry.TriangleMesh()
    for scan_i in tqdm(scan_ids):
        scan = scene_paths.sl_part(scan_i)
        scan = o3d.io.read_triangle_mesh(scan)
        scans.append(scan)
        union_of_scans = union_of_scans + scan; del scan
    union_of_scans.compute_vertex_normals()

    with tempfile.TemporaryDirectory(dir=tmpdir, suffix='_' + scene_paths.scene_name) as tmpdir:
        union_of_scans_ply = f'{tmpdir}/src.ply'
        f'Save the union of scans to {union_of_scans_ply}' >> logger.debug
        verts = np.asarray(union_of_scans.vertices)
        normals = np.asarray(union_of_scans.vertex_normals); del union_of_scans
        save_ply(union_of_scans_ply, verts, vert_normals=normals); del verts, normals

        raw_rec = f'{tmpdir}/raw_rec.ply'
        command = (f'{poisson_rec_bin} --in {union_of_scans_ply} --out {raw_rec} --width {cell_width}'
                   f' --tempDir {tmpdir} --verbose')
        if threads_n is not None:
            command += f' --threads {threads_n}'
        f'Run Poisson Reconstruction: {command}' >> logger.debug
        subprocess.run(command.split(), check=True)

        'Load reconstruction'
        raw_rec = o3d.io.read_triangle_mesh(raw_rec)

    'Compute distance from reconstruction to each source scan' >> logger.debug
    raw_rec.compute_vertex_normals()
    verts = torch.from_numpy(np.asarray(raw_rec.vertices)).to(dtype)
    all_dists = verts.new_full([len(scans), len(verts)], float('nan'))
    for scan_i, scan in tqdm(list(enumerate(scans))):
        raycasting = o3d.t.geometry.RaycastingScene()
        scan = o3d.t.geometry.TriangleMesh.from_legacy(scan)
        raycasting.add_triangles(scan); del scan
        closest_pts = raycasting.compute_closest_points(verts.numpy())['points'].numpy(); del raycasting
        closest_pts = torch.from_numpy(closest_pts).to(dtype)

        to_verts = verts - closest_pts; del closest_pts
        dists = to_verts.norm(dim=1)
        is_close = dists <= max_dist
        all_dists[scan_i, is_close] = dists[is_close]; del dists, is_close

    'Compute distance statistics' >> logger.debug
    mean_dists = all_dists.nanmedian(dim=0)[0]
    mean_dists = mean_dists.where(mean_dists.isfinite(), mean_dists.new_tensor(max_dist))
    mean_dists = mean_dists.numpy()

    all_dists = all_dists.where(all_dists.isfinite(), all_dists.new_tensor(float('inf')))
    min_dists = all_dists.min(dim=0)[0]; del all_dists
    min_dists = min_dists.where(min_dists.isfinite(), min_dists.new_tensor(max_dist))
    dists = min_dists.numpy(); del min_dists

    raw_ply = scene_paths.sl_full('raw')
    f'Save raw reconstruction {raw_ply}' >> logger.debug
    verts = verts.float().numpy()
    tris = np.asarray(raw_rec.triangles)
    cols = np.clip(mean_dists / trim_dist, 0, 1); del mean_dists
    cols = cmap(cols)[:, :3]
    cols = np.clip(cols * 255, 0, 255).astype(np.uint8)
    Path(raw_ply).parent.mkdir(parents=True, exist_ok=True)
    save_ply(raw_ply, verts, tris, vert_colors=cols); del cols, verts, tris

    'Trim reconstruction' >> logger.debug
    vert_is_close = dists <= trim_dist; del dists
    raw_rec.remove_vertices_by_mask(~vert_is_close)

    'Keep only large components' >> logger.debug
    tri_comp_ids, _, comp_areas = raw_rec.cluster_connected_triangles(); del _
    tri_comp_ids = np.asarray(tri_comp_ids)
    comp_areas = np.asarray(comp_areas)
    large_comp_ids = np.nonzero(comp_areas >= min_comp_area)[0]; del comp_areas
    kept_tri_mask = np.zeros_like(tri_comp_ids, dtype=bool)
    for comp_i in large_comp_ids:
        kept_tri_mask |= tri_comp_ids == comp_i
    del large_comp_ids
    rem_tri_mask = ~kept_tri_mask; del kept_tri_mask
    raw_rec.remove_triangles_by_mask(rem_tri_mask); del rem_tri_mask
    raw_rec = raw_rec.remove_unreferenced_vertices()

    pre_cleaned_ply = scene_paths.sl_full('pre_cleaned')
    f'Save pre-cleaned reconstruction {pre_cleaned_ply}' >> logger.debug
    verts = np.asarray(raw_rec.vertices).astype(np.float32)
    tris = np.asarray(raw_rec.triangles)
    Path(pre_cleaned_ply).parent.mkdir(parents=True, exist_ok=True)
    save_ply(pre_cleaned_ply, verts, tris)
    'Done' >> logger.debug


if __name__ == '__main__':
    main()
