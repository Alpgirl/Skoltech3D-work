from argparse import ArgumentParser
import gc
from pathlib import Path

import torch

from skrgbd.data.dataset.params import (cam_pos_ids, stl_view_ids, stl_val_view_ids,
                                        light_setups, kinect_light_setups, stl_light_setups, sensor_to_cam_mode)
from skrgbd.utils import ignore_warnings
from skrgbd.data.io import imgio
from skrgbd.utils.logging import logger, tqdm
from skrgbd.data.processing.undistortion.redistorter import Redistorter
from skrgbd.data.dataset.scene_paths import ScenePaths


def main():
    description = 'Undistorts raw RGB or IR images.'
    parser = ArgumentParser(description=description)
    parser.add_argument('--addons-dir', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--raw-dir', type=str, required=True)
    parser.add_argument('--sensor', type=str, required=True)
    parser.add_argument('--var', type=str)
    parser.add_argument('--scene-name', type=str, required=True)
    args = parser.parse_args()

    if args.var == 'none':
        args.var = None

    f'RGB undistort {args.scene_name}' >> logger.debug
    scene_paths = ScenePaths(args.scene_name, data_dir=args.data_dir, raw_dir=args.raw_dir, addons_dir=args.addons_dir)
    undist(scene_paths, args.sensor, args.var)


def undist(scene_paths, sensor, var=None, device='cpu', dtype=torch.double):
    r"""Undistorts raw RGB or IR images.

    Parameters
    ----------
    scene_paths : ScenePaths
    sensor : str
    var : str
    device : torch.device
    dtype : torch.dtype
    """
    cam, mode = sensor_to_cam_mode[sensor]

    undistorter = scene_paths.undist_model(cam, mode)
    f'Load undist params from {undistorter}' >> logger.debug
    undistorter = Redistorter.from_file(undistorter)
    undistorter = undistorter.to(device, dtype)

    if sensor in {'tis_left', 'tis_right', 'real_sense_rgb', 'phone_left_rgb',
                  'phone_right_rgb', 'real_sense_ir', 'real_sense_ir_right'}:
        pairs = [(light, view_i) for light in light_setups for view_i in cam_pos_ids]
    elif sensor == 'kinect_v2_rgb':
        pairs = [(light, view_i) for light in kinect_light_setups for view_i in cam_pos_ids]
    elif sensor in {'kinect_v2_ir', 'phone_left_ir', 'phone_right_ir'}:
        pairs = list(cam_pos_ids)
    elif cam in {'stl_left', 'stl_right'}:
        mode = var
        if var == 'partial':
            pairs = [(light, view_i) for light in stl_light_setups for view_i in stl_view_ids]
        elif var == 'validation':
            pairs = [(light, view_i) for light in stl_light_setups for view_i in stl_val_view_ids]

    for (light, view_i) in tqdm(pairs):
        undist_img(scene_paths, cam, mode, view_i, light, undistorter, device, dtype)
        gc.collect()  # collect garbage manually to prevent accumulation of allocated memory
    'Done' >> logger.debug


@ignore_warnings(['The given NumPy array is not writeable, and PyTorch'])
def undist_img(scene_paths, cam, mode, view_i, light, undistorter, device, dtype):
    r"""Undistorts a raw RGB or IR image.

    Parameters
    ----------
    scene_paths : ScenePaths
    cam : str
    mode : str
    view_i : int
    light : str
    undistorter : Redistorter
    device : torch.device
    dtype : torch.dtype
    """
    img = scene_paths.img(cam, mode, view_i, light, 'raw')
    f'Load {img}' >> logger.debug
    img = imgio.read[cam][mode](img)

    'Pre-process' >> logger.debug
    img = torch.from_numpy(img)
    src_ndim = img.ndim
    src_dtype = img.dtype
    if src_ndim == 2:
        img = img.unsqueeze(2)
    img = img.permute(2, 0, 1)
    img = img.to(device, dtype)

    'Undistort' >> logger.debug
    img_u = undistorter.redistort(img.unsqueeze(0)).squeeze(0); del img

    'Post-process' >> logger.debug
    img_u = img_u.permute(1, 2, 0)
    if src_ndim == 2:
        img_u = img_u.squeeze(2)

    if src_dtype is torch.uint8:
        img_u = img_u.round_().clamp_(0, 255).to('cpu', torch.uint8)
    elif src_dtype is torch.int:
        img_u = img_u.round_().clamp_(0, 65535).to('cpu', torch.int)  # is actually uint16, hence the max value
    elif src_dtype is torch.float:
        img_u = img_u.to('cpu', torch.float)
    img_u = img_u.numpy()

    dst = scene_paths.img(cam, mode, view_i, light, 'undist')
    f'Save {dst}' >> logger.debug
    Path(dst).parent.mkdir(exist_ok=True, parents=True)
    imgio.write[cam][mode](dst, img_u); del img_u


if __name__ == '__main__':
    main()
