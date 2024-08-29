import numpy as np
import open3d as o3d
import torch

from skrgbd.evaluation.distance import dist_to_surface, dist_to_pts
from skrgbd.utils.logging import logger, tqdm
from skrgbd.data.processing.depth_utils.occluded_mesh_rendering import MeshRenderer
from skrgbd.calibration.camera_models.pinhole import PinholeCameraModel
from skrgbd.data.processing.mesh_utils.subdivide import split_long_edges


def calc_dists(rec, ref, occ, cam_model, world_to_cam, occ_threshold=1e-3, max_visible_depth=3e-3, max_dist=1e-2,
               max_edge_len=1e-4, max_dist_to_sample=1e-1, show_progress=True):
    r"""Calculates distance from reconstruction to reference mesh and occluded space,
    and distance from reference mesh to reconstruction.

    Parameters
    ----------
    rec : o3d.geometry.TriangleMesh or np.ndarray
        of shape [pts_n, 3] and dtype float32.
        Reconstruction produced by a method, in the form of a triangle mesh or points.
    ref : o3d.geometry.TriangleMesh
        SL reconstruction.
    occ : o3d.geometry.TriangleMesh
        Boundary of the occluded space.
    cam_model : PinholeCameraModel
        Pinhole camera model of the SL camera. Used for calculation of the occluded space.
    world_to_cam : torch.FloatTensor
        of shape [scans_n, 4, 4]. Positions of the SL camera. Used for calculation of the occluded space.
    occ_threshold : float
        Points of the reference surface below the boundary of the occluded region deeper than this are occluded.
    max_visible_depth : float
        For reconstructed points below the reference surface deeper than this
        the distance to reference surface is unknown.
    max_dist : float
        For point-to-point calculation of distance from reference to reconstruction,
        distances larger than this are set to inf.
    max_edge_len : float
        For reconstruction in form of triangle mesh, the maximal edge length for resampling of the reconstructed surface.
    max_dist_to_sample : float
        For reconstruction in form of triangle mesh, only triangles closer than this are resampled.
    show_progress : bool

    Returns
    -------
    results : dict
        visible_rec_pt_ids : torch.LongTensor
            of shape [vis_pts_n]. Ids of reconstruction points in visible space.
        dist_to_ref : torch.FloatTensor
            of shape [vis_pts_n]. Distance to the reference surface from the reconstruction points in visible space.
            The distance is unbounded.
        dist_to_occ : torch.FloatTensor
            of shape [vis_pts_n].
            Distance to the occluded space boundary from the reconstruction points in visible space.
            The distance is unbounded.
        dist_from_ref : torch.FloatTensor
            of shape [ref_verts_n]. Distance from the reference surface to the reconstruction points.
            If reconstruction is given as points, the distance is bounded by max_dist,
            otherwise the distance is unbounded.
        rec_pts : torch.FloatTensor
            of shape [pts_n, 3]. Samples on the reconstructed surface, for which the distance is calculated.
        rec_pt_normals : torch.FloatTensor
            of shape [pts_n, 3]. Normals of the samples.
    """
    if isinstance(rec, o3d.geometry.TriangleMesh):
        'Resample reconstructed mesh' >> logger.debug
        resampled_rec = resample_rec(rec, ref, max_edge_len, max_dist_to_sample)
        resampled_rec.compute_vertex_normals()
        pts = np.asarray(resampled_rec.vertices)
        pts = torch.from_numpy(pts).float()
        normals = np.asarray(resampled_rec.vertex_normals); del resampled_rec
        normals = torch.from_numpy(normals).float()
    elif isinstance(rec, np.ndarray):
        pts = torch.from_numpy(rec)

    'Calculate dist from rec to ref' >> logger.debug
    visible_rec_pt_ids, dist_to_ref, dist_to_occ = calculate_dist_from_rec_to_ref(
        pts, ref, occ, cam_model, world_to_cam, max_visible_depth, occ_threshold, show_progress)
    del occ, cam_model, world_to_cam

    'Calculate dist from ref to rec' >> logger.debug
    dist_from_ref = calculate_dist_from_ref_to_rec(ref, rec, max_dist); del ref
    dist_from_ref = torch.from_numpy(dist_from_ref).float()

    results = dict(visible_rec_pt_ids=visible_rec_pt_ids, dist_to_ref=dist_to_ref,
                   dist_to_occ=dist_to_occ, dist_from_ref=dist_from_ref)
    if isinstance(rec, o3d.geometry.TriangleMesh):
        results['rec_pts'] = pts
        results['rec_pt_normals'] = normals
    return results


def calculate_dist_from_ref_to_rec(ref, rec, max_dist=float('inf')):
    r"""Calculates distance to the reconstruction for vertices of the reference mesh.
    If reconstruction is a mesh, calculate point-to-mesh distance.
    If reconstruction is a pointcloud, calculate point-to-point distance.

    Parameters
    ----------
    ref : o3d.geometry.TriangleMesh
        Reference surface.
    rec : o3d.geometry.TriangleMesh or np.ndarray
        of shape [pts_n, 3].
    max_dist : float
        For point-to-point calculation, distances larger than this are set to inf.

    Returns
    -------
    dists : np.ndarray
        of shape [ref_verts_n] and dtype float32 for surface rec, and float64 for points rec.
        If reconstruction is given as points, the distance is bounded by max_dist,
        otherwise the distance is unbounded.
    """
    ref_pts = np.asarray(ref.vertices); del ref
    if isinstance(rec, o3d.geometry.TriangleMesh):
        'Calculate dist to surface' >> logger.debug
        ref_pts = ref_pts.astype(np.float32)
        dists, _ = dist_to_surface(ref_pts, rec); del _
    elif isinstance(rec, np.ndarray):
        'Calculate dist to points' >> logger.debug
        dists, _ = dist_to_pts(ref_pts, rec, max_dist=max_dist); del _
    else:
        raise ValueError
    return dists


def resample_rec(rec, ref, max_edge_len, max_dist_to_sample):
    r"""Subdivides reconstructed mesh to make all edges not longer than a threshold,
    and keeping only the triangles that are closer to the reference mesh than a max distance.

    Parameters
    ----------
    rec : o3d.geometry.TriangleMesh
        Reconstructed surface.
    ref : o3d.geometry.TriangleMesh
        Reference surface.
    max_edge_len : float
        Max edge length for resampling of the reconstructed surface.
    max_dist_to_sample : float
        Only triangles closer than this are resampled.

    Returns
    -------
    resampled_rec : o3d.geometry.TriangleMesh
    """
    'Calculate dists from rec verts to ref' >> logger.debug
    verts = np.asarray(rec.vertices).astype(np.float32)
    dist, _ = dist_to_surface(verts, ref); del _, verts, ref

    f'Remove verts far than {max_dist_to_sample}' >> logger.debug
    vert_is_too_far = dist > max_dist_to_sample; del dist
    rec = o3d.geometry.TriangleMesh(rec)
    rec.remove_vertices_by_mask(vert_is_too_far); del vert_is_too_far

    'Resample mesh' >> logger.debug
    verts = torch.from_numpy(np.asarray(rec.vertices)).float()
    tris = torch.from_numpy(np.asarray(rec.triangles)).long(); del rec
    resampled_rec = split_long_edges(verts, tris, max_edge_len, dtype=verts.dtype)

    return resampled_rec


def calculate_dist_from_rec_to_ref(rec, ref, occ, cam_model, world_to_cam, max_visible_depth=3e-3, occ_threshold=1e-3,
                                   show_progress=True):
    r"""Calculates distance from reconstruction to reference surface and to occluded space boundary.

    Parameters
    ----------
    rec : torch.FloatTensor
        of shape [pts_n, 3]. Points of the reconstruction.
    ref : o3d.geometry.TriangleMesh
        Reference surface.
    occ : o3d.geometry.TriangleMesh
        Surface of the occluded space.
    cam_model : PinholeCameraModel
        Pinhole camera model of the SL camera. Used for calculation of the occluded space.
    world_to_cam : torch.FloatTensor
        of shape [scans_n, 4, 4]. Positions of the SL camera. Used for calculation of the occluded space.
    max_visible_depth : float
        For reconstructed points below the reference surface deeper than this
        the distance to reference surface is unknown.
    occ_threshold : float
        Points of the reference surface below the boundary of the occluded region deeper than this are occluded.
    show_progress : bool

    Returns
    -------
    visible_pt_ids : torch.LongTensor
        of shape [vis_pts_n].
    dist_to_ref : torch.FloatTensor
        of shape [vis_pts_n]. The distance is unbounded.
    dist_to_occ : torch.FloatTensor
        of shape [vis_pts_n]. The distance is unbounded.
    """
    show_progress = tqdm if show_progress else (lambda x: x)

    # Remove reconstruction points in occluded space
    'Initialize renderer' >> logger.debug
    renderer = MeshRenderer(ref, occ, occ_threshold)
    renderer.set_rays_from_camera(cam_model)

    is_in_occluded_space_ids = torch.arange(len(rec))
    for scan_i in show_progress(range(len(world_to_cam))):
        f'Render to view {scan_i:04}' >> logger.debug
        c2w = world_to_cam[scan_i].inverse()
        render = renderer.render_to_camera(c2w[:3, 3], c2w[:3, :3], ['ray_depth'], cull_back_faces=True); del c2w
        depthmap = render['ray_depth']; del render
        depthmap = depthmap.where(depthmap.isfinite(), depthmap.new_tensor(float('-inf')))

        'Project pts to camera' >> logger.debug
        pts_cam = rec[is_in_occluded_space_ids] @ world_to_cam[scan_i, :3, :3].T + world_to_cam[scan_i, :3, 3]
        pix_i = cam_model.get_pix_i(pts_cam.T)
        pt_depths = pts_cam.norm(dim=1); del pts_cam

        'Eliminate visible points' >> logger.debug
        ref_depths = depthmap.view(-1)[pix_i]; del depthmap
        is_in_occluded_space = pt_depths.sub_(max_visible_depth) > ref_depths; del ref_depths, pt_depths
        is_in_occluded_space = is_in_occluded_space.logical_or_(pix_i == -1); del pix_i
        is_in_occluded_space_ids = is_in_occluded_space_ids[is_in_occluded_space]; del is_in_occluded_space
    del renderer, world_to_cam, cam_model

    is_visible = torch.ones(len(rec), dtype=torch.bool)
    is_visible[is_in_occluded_space_ids] = False; del is_in_occluded_space_ids
    visible_pt_ids = torch.arange(len(rec))[is_visible]; del is_visible

    # Calculate distances
    'Calculate distances to GT and to the boundary of occluded space' >> logger.debug
    rec = rec[visible_pt_ids].float().numpy()
    dist_to_ref, _ = dist_to_surface(rec, ref); del _, ref
    dist_to_occ, _ = dist_to_surface(rec, occ); del _, occ, rec
    dist_to_ref = torch.from_numpy(dist_to_ref)
    dist_to_occ = torch.from_numpy(dist_to_occ)

    return visible_pt_ids, dist_to_ref, dist_to_occ


