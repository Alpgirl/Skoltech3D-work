import torch


def get_pseudo_normals(depthmap, scale=5):
    r"""

    Parameters
    ----------
    depthmap : torch.Tensor
        of shape [batch_size, 1, height, width]
    scale : float

    Returns
    -------
    normals : torch.Tensor
        of shape [batch_size, 3, height, width], with coordinates in range [0, 1].
    """
    shape = list(depthmap.shape)
    shape[1] = 3
    normals = depthmap.new_empty(shape)

    depthmap = torch.nn.functional.pad(depthmap, (1, 1, 1, 1), 'replicate').squeeze(1)
    normals[..., 0, :, :] = depthmap[..., 1:-1, 2:] - depthmap[..., 1:-1, 1:-1]
    normals[..., 1, :, :] = depthmap[..., 1:-1, 1:-1] - depthmap[..., 2:, 1:-1]
    normals[..., 2, :, :] = 1 / scale
    normals = torch.nn.functional.normalize(normals, dim=-3)
    normals[..., :2, :, :] = (normals[..., :2, :, :] + 1) / 2
    return normals
