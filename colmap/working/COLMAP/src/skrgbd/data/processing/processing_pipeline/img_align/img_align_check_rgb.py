from argparse import ArgumentParser

from skrgbd.data.processing.processing_pipeline.img_align.img_align_check_sl import check_img_alignment
from skrgbd.data.processing.processing_pipeline.img_align.configs import configs
from skrgbd.utils.logging import logger
from skrgbd.data.dataset.scene_paths import ScenePaths


def main():
    description = ('Compares each image from a camera with the average projection from this camera '
                   'and with the average reference projection from the reference camera.')
    parser = ArgumentParser(description=description)
    parser.add_argument('--addons-dir', type=str, required=True)
    parser.add_argument('--aux-dir', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--raw-dir', type=str, required=True)
    parser.add_argument('--cam', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--light', type=str, required=True)
    parser.add_argument('--scene-name', type=str, required=True)
    args = parser.parse_args()

    f'RGB check alignment {args.scene_name}' >> logger.debug
    scene_paths = ScenePaths(args.scene_name, addons_dir=args.addons_dir,
                             aux_dir=args.aux_dir, data_dir=args.data_dir, raw_dir=args.raw_dir)
    check_rgb_img_alignment(scene_paths, f'{args.log_dir}/{args.scene_name}',
                            cam_name=args.cam, mode=args.mode, light=args.light)


def check_rgb_img_alignment(scene_paths, log_dir, cam_name='tis_right', mode='rgb', light='hdr', view_ids=range(100)):
    r"""Compares each image from a camera with the average projection from this camera
        and with the average reference projection from the SL camera.

    Parameters
    ----------
    scene_paths : ScenePaths
    log_dir : str
    cam_name : str
    mode : str
    light : str
    view_ids : iterable of int
    """
    f'Load config for {cam_name}.{mode}' >> logger.debug
    config = configs[cam_name, mode]
    ref_cam_name = config.load_cam_data_ref.cam_name

    check_img_alignment(
        scene_paths, log_dir, dst_dev_id=cam_name, src_dev_ids=[ref_cam_name, cam_name],
        devs_params={
            ref_cam_name: dict(**config.load_cam_data_ref),
            cam_name: dict(cam_name=cam_name, mode=mode, light=light, view_ids=view_ids, **config.load_cam_data),
        },
        **config.check_img_alignment,
    )


if __name__ == '__main__':
    main()
