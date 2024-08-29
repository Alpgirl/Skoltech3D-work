import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
import torch


def add_custom_cmaps():
    # Cold
    cmap = plt.cm.hot
    vals = np.linspace(0, 1, 100)
    colors = cmap(vals)[:, :3]

    hsv = matplotlib.colors.rgb_to_hsv(colors)
    hsv[:, 0] = np.remainder(1 - hsv[:, 0] + 250 / 360, 1)
    colors = matplotlib.colors.hsv_to_rgb(hsv)

    name = 'cold'
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(name, colors)
    plt.cm.register_cmap(name, cmap)

    name = 'cold_r'
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(name, colors[::-1])
    plt.cm.register_cmap(name, cmap)


add_custom_cmaps()


def colorize_dists(dists, dist_range=(0, 3e-3), color_range=(.02, 1), no_dist_color=(.75, .75, .75), cmap='hot_r'):
    r"""Maps a range of distances to a range of colors in a colormap.

    Parameters
    ----------
    dists : torch.Tensor
        of shape [pts_n]. Nonfinite distances are mapped to no_dist_color.
    dist_range : tuple of float
        (min_dist, max_dist). min_dist is mapped to min_color, max_dist is mapped to max_color.
    color_range : tuple of float
        (min_color, max_color).
    no_dist_color : tuple of float
        (r, g, b), in range [0, 1].
    cmap : str

    Returns
    -------
    colors : torch.Tensor
        of shape [pts_n], in range [0, 1].
    """
    min_dist, max_dist = dist_range
    min_color, max_color = color_range
    cmap = plt.cm.get_cmap(cmap)

    colors = dists.sub(min_dist).mul_((max_color - min_color) / (max_dist - min_dist)).add_(min_color)
    colors = colors.clamp_(min_color, max_color)
    colors = cmap(colors.numpy())[..., :3]
    colors = torch.from_numpy(colors)
    colors = colors.where(dists.isfinite().unsqueeze(1), colors.new_tensor(no_dist_color))
    return colors
