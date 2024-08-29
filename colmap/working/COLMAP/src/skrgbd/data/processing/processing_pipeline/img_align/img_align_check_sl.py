from argparse import ArgumentParser
from pathlib import Path

from PIL import Image
import torch

from skrgbd.data.processing.alignment.image_alignment import Camera, Reprojector
from skrgbd.data.processing.processing_pipeline.img_align.configs import configs
from skrgbd.data.image_utils import get_trim
from skrgbd.data.processing.processing_pipeline.img_align.helper import Helper
from skrgbd.data.processing.processing_pipeline.img_align.img_align_align_rgb import load_cam_data as _load_cam_data
from skrgbd.utils.logging import logger, tqdm
from skrgbd.data.processing.depth_utils.mesh_rendering import MeshRenderer
from skrgbd.data.processing.depth_utils.occluded_mesh_rendering import MeshRenderer as OccMeshRenderer
from skrgbd.data.dataset.scene_paths import ScenePaths


def main():
    description = 'Reprojects the average SL texture from all SL scans to each SL scan and saves to images files'
    parser = ArgumentParser(description=description)
    parser.add_argument('--aux-dir', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--raw-dir', type=str, required=True)
    parser.add_argument('--scene-name', type=str)
    args = parser.parse_args()

    f'Img align check SL alignment {args.scene_name}' >> logger.debug
    scene_paths = ScenePaths(args.scene_name, data_dir=args.data_dir, raw_dir=args.raw_dir, aux_dir=args.aux_dir)
    check_sl_img_alignment(scene_paths, f'{args.log_dir}/{args.scene_name}')


def check_sl_img_alignment(scene_paths, log_dir, mode='partial', scan_ids=range(27)):
    r"""Reprojects the average SL texture from all SL scans to each SL scan and saves to image files.

    Parameters
    ----------
    scene_paths : ScenePaths
    log_dir : str
    mode : {'partial', 'validation'}
    scan_ids : iterable of int
    occ_thres : float
    device : torch.device
    dtype : torch.dype
    """
    f'Load config for stl_right' >> logger.debug
    config = configs['stl_right', mode]

    check_img_alignment(
        scene_paths, log_dir, dst_dev_id='stl_right', src_dev_ids=['stl_right'],
        devs_params=dict(
            stl_right=dict(cam_name='stl_right', mode=mode, light='maxwhite_00_0000', view_ids=scan_ids)
        ),
        **config.check_img_alignment,
    )


def check_img_alignment(scene_paths, log_dir, dst_dev_id, src_dev_ids, devs_params, device='cpu', dtype=torch.float32,
                        angle_gamma=4, max_angle_deg=70, occ_thres=1e-3):
    r"""Reprojects the average of images from a device to another device.

    Parameters
    ----------
    scene_paths : ScenePaths
    log_dir : str
    dst_dev_id : str
    src_dev_ids : iterable of str
    devs_params : dict
        dev_id: dict
            cam_name : str
            mode : str
            light : str
            view_ids : iterable of int
                of shape [views_n].
    device : torch.device
    dtype : torch.dype
    angle_gamma : float
    max_angle_deg : float
        Parameters of projection angle-based weighting, see `calc_angle_weight`.
    occ_thres : float
        Scan points below the occluding surface deeper than this are occluded.
    """
    helper = Helper(scene_paths)

    'Init renderers' >> logger.debug
    rec = helper.load_rec()
    rec = rec.compute_triangle_normals()
    occ = helper.load_occ()
    renderer = OccMeshRenderer(rec, occ, occ_thres); del rec
    occ_renderer = MeshRenderer(occ); del occ
    reprojector = Reprojector(renderer, occ_renderer, occ_thres); del renderer, occ_renderer

    'Load cam data' >> logger.debug
    devs_data = dict()
    for dev_id, params in tqdm(devs_params.items()):
        devs_data[dev_id] = load_cam_data(helper, device=device, dtype=dtype, **params)

    'Project src cams to scan' >> logger.debug
    src_cam_id_lists = dict()
    for dev_id in tqdm(src_dev_ids):
        cams = {f'{dev_id}_{view_i:04}': cam for (view_i, cam) in devs_data[dev_id].items()}
        reprojector.calc_src_uvs(cams, angle_gamma, max_angle_deg)
        src_cam_id_lists[dev_id] = cams.keys()

    'Render images' >> logger.debug
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    crop_stats = dict()
    for view_i, dst_cam in tqdm(devs_data[dst_dev_id].items()):
        i_min, i_max, j_min, j_max = None, None, None, None
        for src_dev_id, src_cam_ids in src_cam_id_lists.items():
            repr_img, vis_pix_ids = reprojector.reproject(dst_cam, src_cam_ids)
            if i_min is None:
                h, w = repr_img.shape[-2:]
                is_valid = repr_img.new_zeros(h, w, dtype=torch.bool)
                is_valid.view(-1)[vis_pix_ids] = True
                i_min, i_max, j_min, j_max = get_trim(is_valid); del is_valid
            del vis_pix_ids
            dst_img = dst_cam.feat_maps['img']
            dst_img = dst_img.where(repr_img.isfinite(), dst_img.new_tensor(float('nan')))
            for img, label in [(repr_img, 'repr'), (dst_img, 'dst')]:
                img = img[:, i_min:i_max, j_min:j_max].permute(1, 2, 0).mul(255).round().clamp(0, 255).byte()  # nan -> 0
                Image.fromarray(img.numpy()).save(f'{log_dir}/{view_i:04}_{src_dev_id}_{label}.jpg')
        crop_stats[view_i] = dict(i_min=i_min, j_min=j_min)
    torch.save(crop_stats, f'{log_dir}/crop_stats.pt')
    'Done' >> logger.debug


def load_cam_data(helper, cam_name, mode, light, view_ids, device, dtype, scale_factor=None, scale_mode='bilinear'):
    r"""Loads camera model, poses, and images, and initializes the Camera objects.

    Parameters
    ----------
    helper : Helper
    cam_name : str
    mode : str
    light : str
    view_ids : iterable of int
        of shape [views_n].
    device : torch.device
    dtype : torch.dtype
    scale_factor : float
    scale_mode : str

    Returns
    -------
    cams : dict
        view_i: Camera
    """
    cams_list, imgs = _load_cam_data(helper, cam_name, mode, light, view_ids, 'ref', device, dtype, scale_factor, scale_mode)
    cams = dict()
    for view_i, cam, img in zip(view_ids, cams_list, imgs):
        cam.feat_maps['img'] = img.to(device, dtype)
        cams[view_i] = cam
    del cams_list, imgs
    return cams


if __name__ == '__main__':
    main()
