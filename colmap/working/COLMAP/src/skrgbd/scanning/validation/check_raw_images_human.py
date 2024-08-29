from argparse import ArgumentParser
from pathlib import Path

import matplotlib.cm as cm
from PIL import Image
import numpy as np
from einops import rearrange
import torch

from skrgbd.data.io.imgio import read as imread
from skrgbd.data.dataset.params import light_setups as all_light_setups
from skrgbd.utils.logging import logger, tqdm
from skrgbd.data.dataset.pathfinder import Pathfinder, sensor_to_cam_mode


imread.real_sense.depth = imread.real_sense.raw_depth
imread.kinect_v2.depth = imread.kinect_v2.raw_depth
imread.phone_right.depth = imread.phone_right.raw_depth
imread.phone_left.depth = imread.phone_left.raw_depth

cam_pos_ids = list(range(30))


def prepare_previews(scene_pathfinder, scene_previews_dir, dmin=500, dmax=1200, cmap=None, sensors=None, modes=None,
                     light_setups=None):
    name = 'CheckRawImgs'
    logger.info(f'{name}: Prepare previews for {scene_pathfinder.scene_name}')
    if cmap is None:
        cmap = cm.plasma_r
    if sensors is None:
        sensors = ['tis_left', 'tis_right', 'real_sense_rgb', 'phone_left_rgb', 'phone_right_rgb', 'kinect_v2_rgb',
                   'real_sense_ir', 'real_sense_ir_right', 'kinect_v2_ir', 'phone_left_ir', 'phone_right_ir']
    sensors = set(sensors)
    if modes is None:
        modes = ['rgb', 'ir', 'depth']
    modes = set(modes)
    if light_setups is None:
        light_setups = all_light_setups

    scene_previews_dir = Path(scene_previews_dir)
    scene_previews_dir.mkdir(parents=True, exist_ok=True)

    if 'rgb' in modes:
        rgb_sensors = {'tis_left', 'tis_right', 'real_sense_rgb', 'phone_left_rgb', 'phone_right_rgb', 'kinect_v2_rgb'}
        rgb_sensors = rgb_sensors.intersection(sensors)
        for sensor in rgb_sensors:
            cam, mode = sensor_to_cam_mode[sensor]
            logger.debug(f'{name}: {cam}.{mode}')
            for light_setup in tqdm(light_setups, desc=f'{cam}.{mode}'):
                if cam == 'kinect_v2':
                    light_setup = light_setup.split('@')[0]
                preview_jpg = f'{scene_previews_dir}/{mode}.{cam}.{light_setup}.jpg'
                if Path(preview_jpg).exists():
                    continue
                grid = load_grid(scene_pathfinder, cam, mode, light_setup)
                grid = Image.fromarray(grid)
                grid.save(preview_jpg); del grid
    
    if 'ir' in modes:
        ir_sensors = {'real_sense_ir', 'real_sense_ir_right'}
        ir_sensors = ir_sensors.intersection(sensors)
        for sensor in ir_sensors:
            cam, mode = sensor_to_cam_mode[sensor]
            logger.debug(f'{name}: {cam}.{mode}')
            for light_setup in tqdm(light_setups, desc=f'{cam}.{mode}'):
                preview_jpg = f'{scene_previews_dir}/{mode}.{cam}.{light_setup}.jpg'
                if Path(preview_jpg).exists():
                    continue
                grid = load_grid(scene_pathfinder, cam, mode, light_setup)
                grid = Image.fromarray(grid)
                grid.save(preview_jpg); del grid

    if 'ir' in modes:
        ir_sensors = {'kinect_v2_ir', 'phone_left_ir', 'phone_right_ir'}
        ir_sensors = ir_sensors.intersection(sensors)
        for sensor in ir_sensors:
            cam, mode = sensor_to_cam_mode[sensor]
            logger.debug(f'{name}: {cam}.{mode}')
            preview_jpg = f'{scene_previews_dir}/{mode}.{cam}.jpg'
            if Path(preview_jpg).exists():
                continue
            grid = load_grid(scene_pathfinder, cam, mode)
            vmax = np.percentile(grid, 95)
            grid = (np.clip(grid / vmax * 255, 0, 255)).astype(np.uint8)
            grid = Image.fromarray(grid)
            grid.save(preview_jpg); del grid

    if 'depth' in modes:
        depth_sensors = {'real_sense_ir',}
        depth_sensors = depth_sensors.intersection(sensors)
        for sensor in depth_sensors:
            cam, mode = sensor_to_cam_mode[sensor]
            mode = 'depth'
            logger.debug(f'{name}: {cam}.{mode}')
            for light_setup in tqdm(light_setups, desc=f'{cam}.{mode}'):
                preview_jpg = f'{scene_previews_dir}/{mode}.{cam}.{light_setup}.jpg'
                if Path(preview_jpg).exists():
                    continue
                grid = load_grid(scene_pathfinder, cam, mode, light_setup)
                missing = grid < 100
                grid = (grid - dmin) / (dmax - dmin)
                grid = cmap(grid)[..., :3]
                grid[missing] = 0
                grid = (np.clip(grid * 255, 0, 255)).astype(np.uint8)
                grid = Image.fromarray(grid)
                grid.save(preview_jpg); del grid

    if 'depth' in modes:
        depth_sensors = {'kinect_v2_ir', 'phone_left_ir', 'phone_right_ir'}
        depth_sensors = depth_sensors.intersection(sensors)
        for sensor in depth_sensors:
            cam, mode = sensor_to_cam_mode[sensor]
            mode = 'depth'
            logger.debug(f'{name}: {cam}.{mode}')
            preview_jpg = f'{scene_previews_dir}/{mode}.{cam}.jpg'
            if Path(preview_jpg).exists():
                continue
            grid = load_grid(scene_pathfinder, cam, mode)
            missing = grid < 100
            grid = (grid - dmin) / (dmax - dmin)
            grid = cmap(grid)[..., :3]
            grid[missing] = 0
            grid = (np.clip(grid * 255, 0, 255)).astype(np.uint8)
            grid = Image.fromarray(grid)
            grid.save(preview_jpg); del grid


def load_grid(scene_pathfinder, cam, mode, light_setup=None, preview_size=(1080, 1920)):
    if light_setup is not None:
        imgs = [scene_pathfinder[cam][mode].raw[light_setup, pos_i] for pos_i in cam_pos_ids]
    else:
        imgs = [scene_pathfinder[cam][mode].raw[pos_i] for pos_i in cam_pos_ids]
    imgs = list(tqdm((imread[cam][mode](img) for img in imgs), desc=f'{light_setup}', total=len(imgs)))
    if imgs[0].ndim == 2:
        imgs = [img[..., None] for img in imgs]
    if not (cam.startswith('phone') and (mode == 'rgb')):
        imgs = [np.rot90(img) for img in imgs]

    grid = rearrange(imgs, '(hh ww) h w c -> (hh h) (ww w) c', ww=10); del imgs
    grid = torch.from_numpy(grid)
    grid = grid.permute(2, 0, 1)
    if grid.dtype is torch.int32:
        grid = grid.float()
    grid = torch.nn.functional.interpolate(grid.unsqueeze(0), size=preview_size).squeeze(0)
    grid = grid.permute(1, 2, 0).numpy()
    if grid.shape[2] == 1:
        grid = grid[..., 0]
    return grid


def main():
    description = r"""This script prepares preview grids for all raw images collected for the specified scenes."""

    parser = ArgumentParser(description=description)
    parser.add_argument('--raw-scans-dir', type=str, required=True)
    parser.add_argument('--previews-dir', type=str, required=True)
    parser.add_argument('--scene-names', type=str, nargs='+')
    parser.add_argument('--sensors', type=str, nargs='+')
    parser.add_argument('--modes', type=str, nargs='+')
    parser.add_argument('--light-setups', type=str, nargs='+')
    parser.add_argument('--trajectory', type=str, default='spheres')
    args = parser.parse_args()

    assert args.trajectory == 'human_sphere'

    pathfinder = Pathfinder(raw_scans_root=args.raw_scans_dir, trajectory=args.trajectory)
    previews_dir = Path(args.previews_dir)

    for scene_name in args.scene_names:
        prepare_previews(pathfinder[scene_name], previews_dir / scene_name,
                         sensors=args.sensors, modes=args.modes, light_setups=args.light_setups)


if __name__ == '__main__':
    main()
