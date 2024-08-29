from argparse import ArgumentParser
from pathlib import Path

import torch

from skrgbd.data.dataset.params import cam_pos_ids, sensor_to_cam_mode
from skrgbd.data.processing.processing_pipeline.addons.mvsnet.configs import configs
from skrgbd.calibration.camera_models import load_from_colmap_txt
from skrgbd.data.io.poses import load_poses
from skrgbd.utils.logging import logger, tqdm
from skrgbd.data.dataset.scene_paths import ScenePaths


def main():
    description = 'Saves camera parameters in MVSNet format.'
    parser = ArgumentParser(description=description)
    parser.add_argument('--addons-dir', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--sensor', type=str, required=True)
    parser.add_argument('--scene-name', type=str, required=True)
    args = parser.parse_args()

    f'Make MVSNet cams {args.scene_name}' >> logger.debug
    config = configs[args.sensor]
    scene_paths = ScenePaths(args.scene_name, addons_dir=args.addons_dir, data_dir=args.data_dir)
    make_cams(scene_paths, args.sensor, **config)


def make_cams(scene_paths, sensor, d_min=.473, d_interval=.002, d_planes_n=256, dtype=torch.double):
    r"""Saves camera parameters in MVSNet format
        https://github.com/YoYo000/MVSNet#camera-files.

    Parameters
    ----------
    scene_paths : ScenePaths
    sensor : str
    d_min : float
    d_interval : float
    d_planes_n : int
    dtype : torch.dtype

    Notes
    -----
    The camera file in MVSNet format contains the camera pose and camera model matrices, and the depth hypotheses range,
    as:

    extrinsic
    E00 E01 E02 E03
    E10 E11 E12 E13
    E20 E21 E22 E23
    E30 E31 E32 E33

    intrinsic
    K00 K01 K02
    K10 K11 K12
    K20 K21 K22

    DEPTH_MIN DEPTH_INTERVAL (DEPTH_NUM DEPTH_MAX)
    """
    cam, mode = sensor_to_cam_mode[sensor]
    f'Load cam data for {cam}.{mode}' >> logger.debug
    cam_model = scene_paths.cam_model(cam, mode)
    cam_model = load_from_colmap_txt(cam_model).to(dtype)

    w2c = scene_paths.cam_poses(cam, mode)
    w2c = load_poses(w2c, dtype=dtype)

    calib_m = torch.eye(3, dtype=dtype)
    calib_m[0, 0] = cam_model.focal[0]
    calib_m[1, 1] = cam_model.focal[1]
    calib_m[0, 2] = cam_model.principal[0]
    calib_m[1, 2] = cam_model.principal[1]; del cam_model

    d_max = d_min + d_interval * (d_planes_n - 1)

    'Save cams' >> logger.debug
    for view_i in tqdm(cam_pos_ids):
        cam_txt = scene_paths.mvsnet_cam(cam, mode, view_i)
        Path(cam_txt).parent.mkdir(exist_ok=True, parents=True)
        data = cam_txt_format.format(w2c_m=mat_to_str(w2c[view_i]), calib_m=mat_to_str(calib_m),
                                     d_min=d_min, d_interval=d_interval, d_planes_n=d_planes_n, d_max=d_max)
        with open(cam_txt, 'w') as f:
            f.write(data)
    'Done' >> logger.debug


def mat_to_str(mat):
    return '\n'.join(' '.join(f'{x:f}' for x in row) for row in mat)


cam_txt_format = r"""extrinsic
{w2c_m}

intrinsic
{calib_m}

{d_min} {d_interval} {d_planes_n} {d_max}"""


if __name__ == '__main__':
    main()
