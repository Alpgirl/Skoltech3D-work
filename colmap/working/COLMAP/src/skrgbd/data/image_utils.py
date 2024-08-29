import numpy as np
import torch


def pack_float32(ar):
    r"""Packs an array of float32 values to the array of uint8 quadruplets.

    Parameters
    ----------
    ar : np.naddary
        of shape [**].

    Returns
    -------
    ar : np.naddary
        of shape [**, 4]
    """
    shape = ar.shape
    return ar.ravel().view(np.uint8).reshape(*shape, 4)


def unpack_float32(ar):
    r"""Unpacks an array of uint8 quadruplets back to the array of float32 values.

    Parameters
    ----------
    ar : np.naddary
        of shape [**, 4].

    Returns
    -------
    ar : np.naddary
        of shape [**]
    """
    shape = ar.shape[:-1]
    return ar.ravel().view(np.float32).reshape(shape)


def equalize_image(im, kernel_size):
    r"""Performs adaptive histogram equalization, similar to
    https://scikit-image.org/docs/dev/api/skimage.filters.rank.html#skimage.filters.rank.equalize.
    Based on the theory from https://www.math.uci.edu/icamp/courses/math77c/demos/hist_eq.pdf.

    Parameters
    ----------
    im : torch.Tensor or np.ndarray
        of shape [B, C, H, W] or [H, W] in case of np array. Must be in [0, 1] range.
    kernel_size : int
        Size of the local neighbourhood. Must by odd.

    Returns
    ----------
    im : torch.Tensor or np.ndarray
        Equalized image in [0, 1] range of the same shape and class as input.
    """
    if isinstance(im, np.ndarray):
        np_input = True
        im = torch.from_numpy(np.ascontiguousarray(im))[None, None]
    else:
        np_input = False

    pad = (kernel_size - 1) // 2
    h, w = im.shape[-2:]
    _ = torch.nn.functional.pad(im, [pad, pad, pad, pad], mode='reflect')
    neighborhood = torch.stack([
        _[..., pad + i: pad + i + h, pad + j: pad + j + w]
        for i in range(-pad, pad + 1) for j in range(-pad, pad + 1)
    ]); del _
    equalized = (neighborhood <= im).double().mean(0)
    if np_input:
        equalized = equalized.numpy()[0, 0]
    return equalized


def get_trim(mask):
    h, w = mask.shape

    i_vals, i_mins = mask.max(0)
    i_min = i_mins[i_vals].min()
    j_vals, j_mins = mask.max(1)
    j_min = j_mins[j_vals].min()

    mask = mask.flip([0, 1])
    i_vals, i_mins = mask.max(0)
    i_max = h - i_mins[i_vals].min()
    j_vals, j_mins = mask.max(1)
    j_max = w - j_mins[j_vals].min()

    return i_min, i_max, j_min, j_max
