from argparse import ArgumentParser
from pathlib import Path

from skrgbd.data.io.poses import load_poses
from skrgbd.utils.logging import logger
from skrgbd.data.io.poses import save_poses
from skrgbd.data.dataset.scene_paths import ScenePaths


def main():
    description = 'Computes refined IR sensor poses from the refined poses of RGB sensor of the same device.'
    parser = ArgumentParser(description=description)
    parser.add_argument('--aux-dir', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--cam', type=str, required=True)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--scene-name', type=str, required=True)
    args = parser.parse_args()

    f'Refine IR poses {args.scene_name}' >> logger.debug
    scene_paths = ScenePaths(args.scene_name, aux_dir=args.aux_dir, data_dir=args.data_dir)
    refine_ir_poses(scene_paths, args.cam, args.mode)


def refine_ir_poses(scene_paths, cam, mode='ir', proxy_view_i=0):
    r"""Computes refined IR sensor poses from the refined poses of RGB sensor of the same device.

    Parameters
    ----------
    scene_paths : ScenePaths
    cam : str
    mode : {'ir', 'ir_right'}
    proxy_view_i : int
    """
    'Compute RGB-to-IR transform' >> logger.debug
    w2c = scene_paths.cam_poses(cam, mode, 'calib')
    w2c = load_poses(w2c)
    w2c = w2c[proxy_view_i]

    w2r = scene_paths.cam_poses(cam, 'rgb', 'calib')
    w2r = load_poses(w2r)
    r2w = w2r[proxy_view_i].inverse(); del w2r

    r2c = w2c @ r2w; del w2c, r2w

    'Compute refined IR poses' >> logger.debug
    w2r = scene_paths.cam_poses(cam, 'rgb', 'ref')
    w2r = load_poses(w2r)

    w2c = r2c @ w2r; del r2c, w2r

    poses = []
    for view_i in range(len(w2c)):
        pose = dict(img_i=view_i + 1)  # COLMAP uses one-based indices
        pose['w2c'] = w2c[view_i]
        pose['filename'] = Path(scene_paths.img(cam, mode, view_i)).name
        poses.append(pose)

    images_txt = scene_paths.cam_poses(cam, mode, 'ref')
    f'Save poses to {images_txt}' >> logger.debug
    save_poses(images_txt, poses)
    'Done' >> logger.debug


if __name__ == '__main__':
    main()
