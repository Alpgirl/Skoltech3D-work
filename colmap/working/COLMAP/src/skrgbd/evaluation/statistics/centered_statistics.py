import torch
import torch_scatter

from skrgbd.evaluation.distance import dist_to_pts
from skrgbd.utils.logging import logger


def centered_stat(pts, attrs, centers, stat, max_dist=float('inf')):
    r"""Reduces distribution of values in Nd-space into closest centers.

    Parameters
    ----------
    pts : torch.Tensor
        of shape [dims_n, pts_n]. Coordinates of the points.
    attrs : torch.Tensor
        of shape [attrs_n, pts_n].
    centers : np.ndarray
        of shape [centers_n, 3]. Coordinates of the centers.
    stat : {'mean', 'min', 'max'}
        Statistic to calculate.
    max_dist : float
        Points farther from this from all centers are filtered out.

    Returns
    -------
    center_ids : torch.LongTensor
        of shape [filled_centers_n]. Ids of centers with points near them.
    center_stats : torch.Tensor
        of shape [attrs_n, filled_centers_n]. Statistics of each center.
    """
    centers_n = len(centers)

    'Calculate center ids' >> logger.debug
    _, center_ids = dist_to_pts(pts.T.numpy(), centers, max_dist=max_dist); del _, centers, pts
    center_ids = torch.from_numpy(center_ids)

    'Filter out points far from center' >> logger.debug
    is_close_to_centers = center_ids != centers_n
    attrs = attrs[:, is_close_to_centers]
    center_ids = center_ids[is_close_to_centers]; del is_close_to_centers

    'Find nonempty centers' >> logger.debug
    center_ids, center_ids_compr = center_ids.unique(sorted=False, return_inverse=True)

    'Calculate statistic' >> logger.debug
    if stat in {'mean', 'min', 'max'}:
        center_stats = torch_scatter.scatter(attrs, center_ids_compr, dim=1, reduce=stat)
    del attrs, center_ids_compr

    return center_ids, center_stats
