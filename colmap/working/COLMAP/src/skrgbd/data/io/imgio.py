import numpy as np
from PIL import Image

from skrgbd.data.image_utils import pack_float32, unpack_float32
from skrgbd.utils import SimpleNamespace


# RGB, IR
# -------

def read_img(file):
    r"""Read RGB, IR, or depth image from PNG or JPEG.

    Parameters
    ----------
    file : str

    Returns
    -------
    img : np.ndarray
        of shape [width, height] and type uint8 or int32 (converted from uint16),
        or of shape [width, height, 3] or [width, height, 4] and type uint8.
    """
    return np.asarray(Image.open(file))


def write_png(img_png, img, optimize=True, **kwargs):
    r"""Saves RGB, IR, or depth image to lossless PNG.

    Parameters
    ----------
    img_png : str
    img : np.ndarray
        of shape [width, height] and type uint8 or uint16,
        or of shape [width, height, 3] or [width, height, 4] and type uint8.
    """
    return Image.fromarray(img).save(img_png, optimize=optimize, **kwargs)


def write_jpg(img_jpg, img, quality=95, optimize=True, **kwargs):
    r"""Saves RGB, IR, or depth image to lossless JPEG.

    Parameters
    ----------
    img_jpg : str
    img : np.ndarray
        of shape [width, height, 3] and type uint8.
    """
    return Image.fromarray(img).save(img_jpg, quality=quality, optimize=optimize, **kwargs)


def read_f32(img_png):
    r"""Reads a float32 image packed into PNG.

    Parameters
    ----------
    img_png : str

    Returns
    -------
    img : np.ndarray
        of shape [height, width], float32
    """
    img = read_img(img_png)
    img = unpack_float32(img)
    return img


def write_f32(img_png, img):
    r"""Packs a float32 image into lossless PNG.

    Parameters
    ----------
    img_png : str
    img : np.ndarray
        of shape [height, width], float32
    """
    img = pack_float32(img)
    return write_png(img_png, img)


read = SimpleNamespace()
write = SimpleNamespace()
for rw in read, write:
    rw.tis_left = rw.tis_right = SimpleNamespace()
    rw.real_sense = SimpleNamespace()
    rw.kinect_v2 = SimpleNamespace()
    rw.phone_left = rw.phone_right = SimpleNamespace()
    rw.stl = SimpleNamespace()
    rw.stl_right = rw.stl_left = SimpleNamespace()

read.tis_left.rgb = read_img
write.tis_left.rgb = write_png

read.real_sense.rgb = read_img
write.real_sense.rgb = write_png
read.real_sense.ir = read.real_sense.ir_right = read_img
write.real_sense.ir = write.real_sense.ir_right = write_png

read.kinect_v2.rgb = read_img
write.kinect_v2.rgb = write_png
read.kinect_v2.ir = read_f32
write.kinect_v2.ir = write_f32

read.phone_left.rgb = read_img
write.phone_left.rgb = write_jpg
read.phone_left.ir = read_img
write.phone_left.ir = write_png

read.stl_left.partial = read.stl_left.validation = read_img
write.stl_left.partial = write.stl_left.validation = write_png


# Depth
# -----

def read_real_sense_raw_depth(img_png, dtype=np.float32, missing_val=0):
    r"""Read raw RealSense depth map.

    Parameters
    ----------
    img_png : str
    dtype : np.dtype
    missing_val : float

    Returns
    -------
    img : np.ndarray
        of shape [width, height] and type `dtype`, in mm, with missing values set to `missing_val`.
    """
    img = read_img(img_png)
    img = img.astype(dtype)
    if missing_val != 0:
        img = np.where(img == 0, missing_val, img)
    return img


def read_kinect_v2_raw_depth(img_png, dtype=np.float32, missing_val=0):
    r"""Read raw Kinect depth map.

    Parameters
    ----------
    img_png : str
    dtype : np.dtype
    missing_val : float

    Returns
    -------
    img : np.ndarray
        of shape [width, height] and type `dtype`, in mm, with missing values set to `missing_val`.
    """
    img = read_f32(img_png)
    img = img.astype(dtype)
    if missing_val != 0:
        img = np.where(img == 0, missing_val, img)
    return img


def read_phone_raw_depth(img_png, dtype=np.float32, missing_val=None):
    r"""Read raw RealSense depth map.

    Parameters
    ----------
    img_png : str
    dtype : np.dtype
    missing_val : float

    Returns
    -------
    img : np.ndarray
        of shape [width, height] and type `dtype`, in mm, with missing values set to `missing_val`.
        If `missing_val` is None, the respective pixels have arbitrary values below 100.
    """
    img = read_img(img_png)
    img = img.astype(dtype)
    if missing_val is not None:
        img = np.where(img <= 100, missing_val, img)
    return img


read.real_sense.raw_depth = read_real_sense_raw_depth
read.kinect_v2.raw_depth = read_kinect_v2_raw_depth
read.phone_left.raw_depth = read_phone_raw_depth

read.stl.depth = read.real_sense.undist_depth = read.kinect_v2.undist_depth = read.phone_left.undist_depth = read_f32
write.stl.depth = write.real_sense.undist_depth = write.kinect_v2.undist_depth = write.phone_left.undist_depth = write_f32
