import numpy as np
import open3d as o3d
import scipy.spatial

from skrgbd.utils.logging import logger


def dist_to_pts(pts_from, pts_to, max_dist=float('inf'), leafsize=16, workers_n=-1):
    r"""Computes unsigned distance from points to points.

    Parameters
    ----------
    pts_from : array_like
        of shape [pts_n, 3].
    pts_to : array_like
        of shape [ref_pts_n, 3].
    max_dist : float
    leafsize : int
    workers_n : int

    Returns
    -------
    dists : np.ndarray
        of shape [pts_n] and dtype float64.
    closest_ids : np.ndarray
        of shape [pts_n] and dtype int64.
    """
    'Build KDTree' >> logger.debug
    tree = scipy.spatial.cKDTree(pts_to, leafsize=leafsize)

    'Query' >> logger.debug
    dists, closest_ids = tree.query(pts_from, distance_upper_bound=max_dist, workers=workers_n); del tree
    return dists, closest_ids


def dist_to_surface(pts_from, mesh_to):
    r"""Computes unsigned distance from points to mesh.

    Parameters
    ----------
    pts_from : np.ndarray
        of shape [pts_n, 3] and dtype float32.
    mesh_to : o3d.geometry.TriangleMesh

    Returns
    -------
    dists : np.ndarray
        of shape [pts_n] and dtype float32.
    closest_pts : np.ndarray
        of shape [pts_n, 3] and dtype float32.
    """
    'Build raycasting' >> logger.debug
    raycasting = o3d.t.geometry.RaycastingScene()
    raycasting.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_to)); del mesh_to

    'Calculate distance' >> logger.debug
    closest_pts = raycasting.compute_closest_points(pts_from)['points'].numpy(); del raycasting
    dists = np.linalg.norm(closest_pts - pts_from, axis=1); del pts_from
    return dists, closest_pts
