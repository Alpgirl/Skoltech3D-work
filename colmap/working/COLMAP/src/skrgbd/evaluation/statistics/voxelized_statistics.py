import torch
import torch_scatter

from skrgbd.utils.logging import logger


class VoxelizedStatistics:
    r"""Reduces distribution of values in Nd-space into cells.

    Parameters
    ----------
    pts : torch.Tensor
        of shape [dims_n, pts_n]. Coordinates of the points.
    cell_size : float
        Size of the cell.
    """

    def __init__(self, pts, cell_size):
        self.device = pts.device
        self.dtype = pts.dtype
        self.dims_n = len(pts)
        self.cell_size = cell_size

        'Put points to cells' >> logger.debug
        cell_ijk = pts.div(cell_size).floor_().long(); del pts
        self.grid_origin = cell_ijk.min(1)[0]
        cell_ijk = cell_ijk.sub_(self.grid_origin.unsqueeze(1))
        self.grid_size = cell_ijk.max(1)[0] + 1

        'Calculate cell ids' >> logger.debug
        cell_ids = cell_ijk[-1].clone()
        stride = 1
        for dim_i in range(self.dims_n - 2, -1, -1):
            stride *= self.grid_size[dim_i + 1]
            cell_ids = cell_ids.add_(cell_ijk[dim_i] * stride)
        del cell_ijk, stride

        'Find nonempty cells' >> logger.debug
        self.filled_cell_ids, self.cell_ids_compr = cell_ids.unique(sorted=False, return_inverse=True); del cell_ids

    def reduce(self, vals, stat):
        r"""Reduce the values.

        Parameters
        ----------
        vals : torch.Tensor
            of shape [attrs_n, pts_n].
        stat : {'sum', 'mean', 'min', 'max'}
            Statistic to calculate.

        Returns
        -------
        cell_stats : torch.Tensor
            of shape [attrs_n, cells_n]. Statistics of each cell.
        """
        cell_stats = torch_scatter.scatter(vals, self.cell_ids_compr, dim=1, reduce=stat)
        return cell_stats

    def calc_cell_coords(self):
        r"""Calculates nonempty cell coordinates.

        Returns
        -------
        cell_coords : torch.Tensor
            of shape [dims_n, cells_n]. Coordinates of the cell centers.
        """
        cell_coords = torch.empty([self.dims_n, len(self.filled_cell_ids)], device=self.device, dtype=self.dtype)
        cell_coords[-1] = self.filled_cell_ids.remainder(self.grid_size[-1])
        for dim_i in range(self.dims_n - 2, -1, -1):
            filled_cell_ids = filled_cell_ids.div_(self.grid_size[dim_i + 1], rounding_mode='floor')
            cell_coords[dim_i] = filled_cell_ids.remainder(self.grid_size[dim_i])
        del filled_cell_ids
        cell_coords = cell_coords.add_(self.grid_origin.unsqueeze(1) + .5).mul_(self.cell_size)

        return cell_coords
