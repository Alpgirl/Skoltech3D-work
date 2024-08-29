from PIL import Image
import numpy as np


def make_im_slice(pos, size, step=None):
    return (slice(pos[0] - size // 2, pos[0] + size // 2, step),
            slice(pos[1] - size // 2, pos[1] + size // 2, step))


def get_luma(img):
    r"""Extracts the ITU-R 601-2 luma:
        L = R * 299/1000 + G * 587/1000 + B * 114/1000

    Parameters
    ----------
    img : np.ndarray
        of shape [h, w, 3].

    Returns
    -------
    luma : np.ndarray
        of shape [h, w].
    """
    img = np.asarray(Image.fromarray(img).convert('L'))
    return img
