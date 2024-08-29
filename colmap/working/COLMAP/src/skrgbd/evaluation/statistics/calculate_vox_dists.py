import torch

from skrgbd.utils.logging import logger as glob_logger
from skrgbd.evaluation.statistics import VoxelizedStatistics


def calc_vox_completeness(ref_pts, dist_from_ref, threshold, cell_size=1e-3, max_dist=1e-2):
    r"""FIXME add desc

    Parameters
    ----------
    ref_pts : torch.Tensor
        of shape [3, pts_n]. Points of the reference surface.
    dist_from_ref : torch.Tensor
        of shape [pts_n]. Distance from the reference surface to the reconstruction.
    threshold : float
        FIXME add desc
    cell_size : float
        Size of the cell for intermediate averaging.
    max_dist : float
        Distances are max-clipped by this value.

    Returns
    -------
    completeness : torch.Tensor
        of shape [vox_n]. FIXME add desc
    """
    logger = glob_logger.get_context_logger('VoxComp')

    'Init voxelized stats' >> logger.debug
    stat_calculator = VoxelizedStatistics(ref_pts, cell_size); del ref_pts

    'Calculate stats' >> logger.debug
    dists = dist_from_ref.clamp(max=max_dist); del dist_from_ref
    comp = dists.le(threshold).float()
    comp = stat_calculator.reduce(comp.unsqueeze(0), 'mean').squeeze(0)
    return comp


def calc_vox_accuracy(rec_pts, dist_to_ref, dist_to_occ, threshold, cell_size=1e-3, max_dist=1e-2, occ_eps=1e-4):
    r"""FIXME

    Parameters
    ----------
    rec_pts : torch.Tensor
        of shape [3, vis_pts_n]. Points of the reconstruction, only in the visible space.
    dist_to_ref : torch.Tensor
        of shape [vis_pts_n]. Distance to the reference surface from the reconstruction points in visible space.
    dist_to_occ : torch.Tensor
        of shape [vis_pts_n].
        Distance to the occluded space boundary from the reconstruction points in visible space.
    threshold : float
        FIXME
    cell_size : float
        Size of the cell for intermediate averaging.
    max_dist : float
        Distances are max-clipped by this value.
    occ_eps : float
        If the distance from a reconstructed point to the reference surface is greater than the distance
        to the boundary of the occluded space plus occ_eps, then the distance from the point to the true surface
        is considered unknown.

    Returns
    -------
    accuracy : torch.Tensor
        of shape [vox_n]. FIXME
    """
    logger = glob_logger.get_context_logger('VoxAcc')

    'Init voxelized stats' >> logger.debug
    stat_calculator = VoxelizedStatistics(rec_pts, cell_size); del rec_pts

    'Calculate stats' >> logger.debug
    dist_to_occ = dist_to_occ + occ_eps
    dist_to_ref = dist_to_ref.clamp(max=max_dist)

    ref_is_closer = dist_to_ref <= threshold
    occ_is_farther = dist_to_occ > threshold
    dist_vs_thres_is_certain = occ_is_farther.logical_or_(ref_is_closer); del occ_is_farther
    ref_is_closer = ref_is_closer.float()
    dist_vs_thres_is_certain = dist_vs_thres_is_certain.float()

    certain_pts_n_per_cell = stat_calculator.reduce(dist_vs_thres_is_certain.unsqueeze(0), 'sum').squeeze(0)
    del dist_vs_thres_is_certain
    acc_per_cell = stat_calculator.reduce(ref_is_closer.unsqueeze(0), 'sum').squeeze(0); del ref_is_closer
    is_not_empty = certain_pts_n_per_cell.ne(0)
    acc = acc_per_cell[is_not_empty].div(certain_pts_n_per_cell[is_not_empty])
    del acc_per_cell, certain_pts_n_per_cell, is_not_empty
    return acc
