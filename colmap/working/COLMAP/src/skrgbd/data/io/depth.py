from PIL import Image
import numpy as np

from skrgbd.data.image_utils import pack_float32, unpack_float32


def save_depthmap(file, depthmap):
    r"""Saves a depthmap as float32 array packed into RGBA png.

    Parameters
    ----------
    file : str
    depthmap : np.ndarray
        of shape [height, width], np.float32
    """
    depthmap = pack_float32(depthmap)
    depthmap = Image.fromarray(depthmap)
    depthmap.save(file)


def load_depthmap(file):
    r"""Loads a float32 depthmap packed into RGBA png.

    Parameters
    ----------
    file : str

    Returns
    -------
    depthmap : np.ndarray
        of shape [height, width], np.float32
    """
    depthmap = Image.open(file)
    depthmap = np.asarray(depthmap)
    depthmap = unpack_float32(depthmap)
    return depthmap
