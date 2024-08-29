from argparse import ArgumentParser
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch
import open3d as o3d

from skrgbd.utils.logging import logger, tqdm
from skrgbd.data.processing.occlusion.occlusion_rv import OcclusionHelper
from skrgbd.data.dataset.pathfinder import Pathfinder
from skrgbd.data.io.ply import save_ply
from skrgbd.data.dataset.dataset import wip_scene_name_by_id



# def main():
#     # TODO
#     description = r"""This script
#     (1) Reconstructs the surface from all partial structured light scans and saves it,
#     (2) And also saves the reconstruction trimmed by the distance to the source scans.
#     """
#     name = 'Main'
#     parser = ArgumentParser(description=description)
#     parser.add_argument('--raw-scans-dir', type=str, required=True)
#     parser.add_argument('--processed-scans-dir', type=str, required=True)
#     parser.add_argument('--aux-dir', type=str, required=True)
#     parser.add_argument('--poisson-recon-bin', type=str, required=True)
#     group = parser.add_mutually_exclusive_group(required=True)
#     group.add_argument('--scene-name', type=str)
#     group.add_argument('--scene-i', type=int)
#     parser.add_argument('--progress', action='store_true')
#     args = parser.parse_args()
#
#     scene_name = args.scene_name if args.scene_name else (wip_scene_name_by_id[args.scene_i])
#     logger.info(f'{name}: Scene name {scene_name}')
#     pathfinder = Pathfinder(
#         data_root=args.processed_scans_dir, aux_root=args.aux_dir, raw_scans_root=args.raw_scans_dir)
#     reconstruct_surface(scene_name, pathfinder, args.poisson_recon_bin, show_progress=args.progress)
#

def make_occlusion_mesh(
        scene_name, pathfinder, poisson_rec_bin, cell_width=3e-4, tmpdir='/tmp', device='cuda', dtype=torch.float,
        batch_size=10_240_000, dilation=3e-3, subdivs_n=1, rec_fidelity_threshold=1e-3, free_space_depth=1e-4,
        grid_wh=(2048 * 2, 1536 * 2), sample_size=1e-4, resample_rec=True, show_progress=True,
):
    r"""Calculates visibility occlusion mesh for a structured-light reconstruction
    given partial scans and the respective camera positions.

    Parameters
    ----------
    scene_name : str
    pathfinder : Pathfinder
    poisson_rec_bin : str
    cell_width : float
        Cell size in Poisson surface reconstruction.
    tmpdir : str
    device : torch.device
        Device used for carving.
    dtype : torch.dtype
        Data type used for carving.
    batch_size : int
        Size of the batch of the points during carving.
    dilation : float
        The initial envelope is dilated by this value.
    subdivs_n : int
        The initial envelope is calculated as the convex hull of the points of the reconstruction,
        each replaced with iso-sphere with the number of subdivisions `subdivs_n` and of radius `dilation`.
    rec_fidelity_threshold : float
        The partial depth maps are calculated from the reconstruction.
        If the depth value calculated from the reconstruction differs from the value calculated from the partial scan
        by more than `rec_fidelity_threshold`, then this value is rejected.
    free_space_depth : float
        A point is carved out if `depth_point < depth_reconstruction + free_space_depth`.
    grid_wh : tuple of int
        Resolution of partial depth maps.
    sample_size : float
        Approximate inter-sample distance.
    resample_rec : bool
        The samples for carving are drawn from the surface of the envelope and from boundary triangles
        of partial surfaces, i.e triangles between pixels with and without value.
        After carving, the remaining samples are concatenated with the samples from the reconstruction, to which
        the carving is not applied since reconstruction is guaranteed to lie on the surface of the occlusion mesh.
        If False, use vertices of the reconstruction as samples. If True, resample the reconstruction
        with the same density as the other parts, which is useful if the original sampling density of the reconstruction
        is lower than the sampling density of the other parts.
    show_progress : bool

    Notes
    -----
    The idea is to start from some envelope of the object, i.e the surface fully containing the object,
    and make this envelope tighter via carving out its portions which were in the free space
    during structured-light scanning, i.e which are located between the camera and the surface of the partial scan.
    The resulting occlusion surface can then be used during rendering of the structured-light reconstruction
    for occlusion of the holes appearing due to incompleteness of the structured-light scans.
    If the distance along the pixel ray to the occlusion surface is less than the distance to the reconstruction
    (minus some threshold to account for numerical errors),
    then the reconstruction is occluded along this ray and the rendered value, such as depth, is undefined.

    The occlusion mesh is the intersection of the envelope and the surfaces between the free and the occluded space
    in each partial scan, i.e triangulated depth maps with pixels with missing values protruding into the camera.
    Such surfaces contain long narrow triangles, corresponding to the boundary between pixels with and without value,
    so their explicit intersection is computationally intractable. Instead, we intersect them implicitly.
    For this, we sample points on these surfaces, and carve out the points lying outside of the occlusion mesh.
    A point lies outside of the occlusion mesh if and only if there is a partial scan where the depth along the ray
    from the respective camera to this point is greater than the depth of this point w.r.t the camera,
    which is trivial to test.
    Finally, we reconstruct the occlusion surface from point samples using Poisson surface reconstruction.
    Since both carving of the point cloud and Poisson reconstruction are extremely efficient,
    the points can be sampled with an excessive density (tested up to 10um inter-sample distance),
    and the occlusion mesh can be reconstructed with high accuracy.
    """
    name = 'MakeOcclusionMesh'
    show_progress = tqdm if show_progress else None

    occlusion_maker = OcclusionHelper(device, dtype)

    logger.info(f'{name}: Load reconstruction')
    rec = pathfinder[scene_name].stl.reconstruction.cleaned
    rec = o3d.io.read_triangle_mesh(rec)
    occlusion_maker.set_reconstruction(rec); del rec

    logger.info(f'{name}: Compute envelope')
    occlusion_maker.set_envelope(dilation, subdivs_n)

    logger.info(f'{name}: Initialize partial scans')
    occlusion_maker.init_partial_scans(pathfinder, scene_name)

    logger.info(f'{name}: Initialize carving rays')
    occlusion_maker.init_carving_rays(grid_wh)

    logger.info(f'{name}: Initialize carving depth maps')
    occlusion_maker.init_partial_carving_depths(rec_fidelity_threshold, free_space_depth, show_progress)

    logger.info(f'{name}: Sample occlusion surface')
    occlusion_maker.sample_occlusion_surface(rec_fidelity_threshold, sample_size, batch_size, resample_rec, show_progress)

    with tempfile.TemporaryDirectory(dir=tmpdir, suffix='_' + scene_name) as tmpdir:
        logger.info(f'{name}: Save samples to {tmpdir}')
        samples = np.concatenate(occlusion_maker.samples, 0, dtype=np.float64)
        normals = np.concatenate(occlusion_maker.normals, 0, dtype=np.float64)
        del occlusion_maker

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(samples); del samples
        pc.normals = o3d.utility.Vector3dVector(normals); del normals

        pc_ply = f'{tmpdir}/samples.ply'
        o3d.io.write_point_cloud(pc_ply, pc); del pc

        rec_ply = f'{tmpdir}/rec.ply'
        command = (
            f'{poisson_rec_bin} --in {pc_ply} --out {rec_ply} --width {cell_width}'
            f' --tempDir {tmpdir} --verbose'
        )
        logger.info(f'{name}: Run Poisson Reconstruction: {command}')
        subprocess.run(command.split(), check=True)

        logger.info(f'{name}: Load reconstruction')
        rec = o3d.io.read_triangle_mesh(rec_ply)

        logger.info(f'{name}: Clean')
        tri_cluster_i, _, areas = rec.cluster_connected_triangles(); del _
        max_cluster_i = np.asarray(areas).argmax(); del areas
        tri_cluster_i = np.asarray(tri_cluster_i)
        tri_ids_not_in_max_cluster = np.argwhere(tri_cluster_i != max_cluster_i)[:, 0]; del tri_cluster_i, max_cluster_i

        rec.remove_triangles_by_index(tri_ids_not_in_max_cluster); del tri_ids_not_in_max_cluster
        rec.remove_unreferenced_vertices()

        occluded_space_ply = pathfinder[scene_name].stl.occluded_space
        logger.info(f'{name}: Save to {occluded_space_ply}')
        Path(occluded_space_ply).parent.mkdir(parents=True, exist_ok=True)
        save_ply(occluded_space_ply, np.asarray(rec.vertices), np.asarray(rec.triangles))


# if __name__ == '__main__':
#     main()
