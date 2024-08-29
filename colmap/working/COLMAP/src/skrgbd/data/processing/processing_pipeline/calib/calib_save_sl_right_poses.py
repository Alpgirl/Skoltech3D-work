from argparse import ArgumentParser
from pathlib import Path

import torch

from skrgbd.utils.logging import logger
from skrgbd.data.dataset.scene_paths import ScenePaths
from skrgbd.data.io.poses import save_poses


def main():
    description = 'Saves the poses of SL right camera in COLMAP format.'
    parser = ArgumentParser(description=description)
    parser.add_argument('--addons-dir', type=str, required=True)
    parser.add_argument('--aux-dir', type=str, required=True)
    parser.add_argument('--raw-calib-dir', type=str, required=True)
    parser.add_argument('--cam', type=str, required=True)
    parser.add_argument('--var', type=str)
    parser.add_argument('--scene-name', type=str, required=True)
    args = parser.parse_args()

    f'Calib save SL poses {args.scene_name}' >> logger.debug
    scene_paths = ScenePaths(args.scene_name, aux_dir=args.aux_dir,
                             raw_calib_dir=args.raw_calib_dir, addons_dir=args.addons_dir)
    save_sl_poses(scene_paths, args.cam, args.var)


def save_sl_poses(scene_paths, cam='stl_right', var='ref'):
    r"""Saves the refined poses of the SL right camera in COLMAP TXT format.

    Parameters
    ----------
    scene_paths : ScenePaths
    cam : {'stl_right'}
    var : {'ref', 'val'}
    """
    'Read poses' >> logger.debug
    s2w = scene_paths.sl_board_to_w_refined(var)  # refined partial scan-to-world transform
    s2w = torch.load(s2w)
    s2w = torch.stack(s2w, 0).double()
    w2s = s2w.inverse(); del s2w

    s2c = scene_paths.raw_calib.scan0_to_world  # scan-to-stl_right transform
    s2c = torch.load(s2c).double()
    w2c = s2c @ w2s; del s2c, w2s

    poses = []
    img_mode = {'ref': 'partial', 'val': 'validation'}[var]
    for view_i in range(len(w2c)):
        pose = dict(img_i=view_i + 1)  # COLMAP uses one-based indices
        pose['w2c'] = w2c[view_i]
        pose['filename'] = Path(scene_paths.img(cam, img_mode, view_i, var='undist')).name
        poses.append(pose)

    'Save poses' >> logger.debug
    images_txt = scene_paths.cam_poses(cam, img_mode, 'ref')
    save_poses(images_txt, poses)
    'Done' >> logger.debug


if __name__ == '__main__':
    main()
