import torch

from skrgbd.utils.logging import logger
from skrgbd.evaluation.statistics import VoxelizedStatistics


def calc_ref_to_rec_stats(ref_pts, dist_from_ref, thresholds, cell_size=1e-3, max_dist=1e-2, batch_size=1):
    r"""Calculates mean distance from the reference surface to the reconstruction,
    and completeness of the reconstruction for different values of the threshold.

    Parameters
    ----------
    ref_pts : torch.Tensor
        of shape [3, pts_n]. Points of the reference surface.
    dist_from_ref : torch.Tensor
        of shape [pts_n]. Distance from the reference surface to the reconstruction.
    thresholds : torch.Tensor
        of shape [thres_n]. Thresholds for which to calculate the metric.
    cell_size : float
        Size of the cell for intermediate averaging.
    max_dist : float
        Distances are max-clipped by this value.
    batch_size : int
        Number of thresholds to calculate the metrics at once.

    Returns
    -------
    completeness : torch.Tensor
        of shape [thres_n]. Completeness of the reconstruction for each value of the threshold.
    mean_ref_to_rec_distance : float
        Mean distance from the reference surface to the reconstruction.
    """
    'Init voxelized stats' >> logger.debug
    stat_calculator = VoxelizedStatistics(ref_pts, cell_size); del ref_pts

    'Calculate stats' >> logger.debug
    dists = dist_from_ref.clamp(max=max_dist); del dist_from_ref

    def reduce_batch(thres_batch):
        closer_than_thres = dists <= thres_batch.unsqueeze(1)
        closer_than_thres = closer_than_thres.float()
        comp_batch = stat_calculator.reduce(closer_than_thres, 'mean').mean(1)
        return comp_batch

    comp = []
    for batch_start in range(0, len(thresholds), batch_size):
        batch_end = min(batch_start + batch_size, len(thresholds))
        comp.append(reduce_batch(thresholds[batch_start: batch_end]))
    comp = torch.cat(comp); del thresholds

    mean_ref_to_rec = stat_calculator.reduce(dists.unsqueeze(0), 'mean').squeeze(0).mean(); del dists
    mean_ref_to_rec = mean_ref_to_rec.item()

    return comp, mean_ref_to_rec


def calc_rec_to_ref_stats(rec_pts, dist_to_ref, dist_to_occ, thresholds,
                          cell_size=1e-3, max_dist=1e-2, batch_size=1, occ_eps=1e-4):
    r"""Calculates mean distance from the reconstruction to the reference surface,
    and accuracy of the reconstruction for different values of the threshold.

    Parameters
    ----------
    rec_pts : torch.Tensor
        of shape [3, vis_pts_n]. Points of the reconstruction, only in the visible space.
    dist_to_ref : torch.Tensor
        of shape [vis_pts_n]. Distance to the reference surface from the reconstruction points in visible space.
    dist_to_occ : torch.Tensor
        of shape [vis_pts_n].
        Distance to the occluded space boundary from the reconstruction points in visible space.

    thresholds : torch.Tensor
        of shape [thres_n]. Thresholds for which to calculate the metric.
    cell_size : float
        Size of the cell for intermediate averaging.
    max_dist : float
        Distances are max-clipped by this value.
    batch_size : int
        Number of thresholds to calculate the metrics at once.
    occ_eps : float
        If the distance from a reconstructed point to the reference surface is greater than the distance
        to the boundary of the occluded space plus occ_eps, then the distance from the point to the true surface
        is considered unknown.

    Returns
    -------
    accuracy : torch.Tensor
        of shape [thres_n]. Accuracy of the reconstruction for each value of the threshold.
    mean_rec_to_ref_distance : float
        Mean distance from the reconstruction to the reference surface.
    """
    'Init voxelized stats' >> logger.debug
    stat_calculator = VoxelizedStatistics(rec_pts, cell_size); del rec_pts

    'Calculate stats' >> logger.debug
    dist_to_occ = dist_to_occ + occ_eps
    dist_to_ref = dist_to_ref.clamp(max=max_dist)

    def reduce_batch(thres_batch, div_eps=1e-12):
        ref_is_closer = dist_to_ref <= thres_batch.unsqueeze(1)
        occ_is_farther = dist_to_occ > thres_batch.unsqueeze(1)
        dist_vs_thres_is_certain = occ_is_farther.logical_or_(ref_is_closer); del occ_is_farther

        ref_is_closer = ref_is_closer.float()
        dist_vs_thres_is_certain = dist_vs_thres_is_certain.float()

        certain_pts_n_per_cell = stat_calculator.reduce(dist_vs_thres_is_certain, 'sum'); del dist_vs_thres_is_certain
        acc_per_cell = stat_calculator.reduce(ref_is_closer, 'sum'); del ref_is_closer
        acc_per_cell = acc_per_cell.div_(certain_pts_n_per_cell.add(div_eps))
        acc_batch = acc_per_cell.sum(1).div_(certain_pts_n_per_cell.ne(0).sum(1))
        return acc_batch

    acc = []
    for batch_start in range(0, len(thresholds), batch_size):
        batch_end = min(batch_start + batch_size, len(thresholds))
        acc.append(reduce_batch(thresholds[batch_start: batch_end]))
    acc = torch.cat(acc)

    dists_is_certain = dist_to_ref <= dist_to_occ; del dist_to_occ
    dists_is_certain = dists_is_certain.float()
    certain_pts_n_per_cell = stat_calculator.reduce(dists_is_certain.unsqueeze(0), 'sum').squeeze(0)
    del dists_is_certain
    mean_rec_to_ref = stat_calculator.reduce(dist_to_ref.unsqueeze(0), 'sum').squeeze(0); del dist_to_ref
    mean_rec_to_ref = mean_rec_to_ref.div_(certain_pts_n_per_cell)
    mean_rec_to_ref = mean_rec_to_ref[certain_pts_n_per_cell != 0].mean(); del certain_pts_n_per_cell
    mean_rec_to_ref = mean_rec_to_ref.item()

    return acc, mean_rec_to_ref
