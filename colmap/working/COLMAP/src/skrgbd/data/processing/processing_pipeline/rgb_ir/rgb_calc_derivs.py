from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np

from skrgbd.data.dataset.params import cam_pos_ids, light_setups, kinect_light_setups, sensor_to_cam_mode
from skrgbd.data.io import imgio
from skrgbd.utils.logging import logger, tqdm
from skrgbd.data.dataset.scene_paths import ScenePaths


def main():
    description = 'Calculates HDR images using the images captured under all lighting except flash and ambient_low.'
    parser = ArgumentParser(description=description)
    parser.add_argument('--addons-dir', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--sensor', type=str, required=True)
    parser.add_argument('--scene-name', type=str, required=True)
    args = parser.parse_args()

    f'RGB calc derivatives {args.scene_name}' >> logger.debug
    scene_paths = ScenePaths(args.scene_name, addons_dir=args.addons_dir, data_dir=args.data_dir)
    calc_derivs(scene_paths, args.sensor)


def calc_derivs(scene_paths, sensor, var='undist', dtype=np.float32):
    r"""Calculates HDR images using the images captured under all lighting except flash and ambient_low.

    Parameters
    ----------
    scene_paths : ScenePaths
    sensor : str
    var : str
    dtype : np.dtype
    """
    cam, mode = sensor_to_cam_mode[sensor]

    if cam == 'kinect_v2':
        excluded_lights = {'flash', 'ambient_low'}
        hdr_src_lights = list(filter(lambda l: (l not in excluded_lights), kinect_light_setups))
    else:
        excluded_lights = {'flash@best', 'flash@fast', 'ambient_low@fast'}
        hdr_src_lights = list(filter(lambda l: (l not in excluded_lights), light_setups))

    'Init MergeMertens' >> logger.debug
    merge_mertens = cv2.createMergeMertens()

    for view_i in tqdm(cam_pos_ids):
        imgs = []
        for light in hdr_src_lights:
            img = scene_paths.img(cam, mode, view_i, light, var)
            # f'Load {img}' >> logger.debug
            img = imgio.read[cam][mode](img)
            img = img.astype(dtype)
            imgs.append(img); del img

        'Calc HDR' >> logger.debug
        hdr = merge_mertens.process(imgs); del imgs
        hdr = np.clip(np.round(hdr * 255), 0, 255).astype(np.uint8)

        dst = scene_paths.img(cam, mode, view_i, 'hdr', var)
        f'Save HDR {dst}' >> logger.debug
        Path(dst).parent.mkdir(exist_ok=True, parents=True)
        imgio.write[cam][mode](dst, hdr); del hdr
    'Done' >> logger.debug


if __name__ == '__main__':
    main()
