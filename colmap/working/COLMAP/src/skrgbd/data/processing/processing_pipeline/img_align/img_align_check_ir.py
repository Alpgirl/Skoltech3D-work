from argparse import ArgumentParser
from pathlib import Path

from PIL import Image
import torch

from skrgbd.data.processing.processing_pipeline.img_align.configs import configs
from skrgbd.data.image_utils import get_trim
from skrgbd.data.processing.processing_pipeline.img_align.helper import Helper
from skrgbd.utils.logging import logger, tqdm
from skrgbd.data.processing.depth_utils.occluded_mesh_rendering import MeshRenderer
from skrgbd.data.dataset.scene_paths import ScenePaths


def main():
    description = 'Renders SL scan to the viewpoints of IR cameras and saves the renders to files'
    parser = ArgumentParser(description=description)
    parser.add_argument('--addons-dir', type=str, required=True)
    parser.add_argument('--aux-dir', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--cam', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--scene-name', type=str, required=True)
    args = parser.parse_args()

    f'IR check alignment {args.scene_name}' >> logger.debug
    scene_paths = ScenePaths(args.scene_name, addons_dir=args.addons_dir, aux_dir=args.aux_dir, data_dir=args.data_dir)
    check_ir_img_alignment(scene_paths, f'{args.log_dir}/{args.scene_name}', args.cam, args.mode)


def check_ir_img_alignment(scene_paths, log_dir, cam_name, mode='ir', view_ids=range(100), device='cpu', dtype=torch.float):
    r"""Renders SL scan to the viewpoints of IR cameras and saves the renders to files.

    Parameters
    ----------
    scene_paths : ScenePaths
    log_dir : str
    cam_name : str
    mode : {'ir', 'ir_right'}
    view_ids : iterable of int
    device : torch.device
    dtype : torch.dtype
    """
    helper = Helper(scene_paths)

    f'Load config for {cam_name}.{mode}' >> logger.debug
    config = configs[cam_name, mode]

    'Init renderers' >> logger.debug
    rec = helper.load_rec()
    occ = helper.load_occ()
    renderer = MeshRenderer(rec, occ, config.occ_thres); del rec, occ

    'Load cam data' >> logger.debug
    cam_model = helper.load_cam_model(cam_name, mode).to(device, dtype)
    w2c = helper.load_cam_poses(cam_name, mode, 'ref').to(device, dtype)

    ss = config.ss
    if ss is not None:
        f'Supersample camera model: {ss}' >> logger.debug
        cam_model = cam_model.resize_(cam_model.size_wh * ss)

    renderer.set_rays_from_camera(cam_model)

    'Render images' >> logger.debug
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    crop_stats = dict()
    for view_i in tqdm(view_ids):
        # f'Render scan to view {view_i}' >> logger.debug
        c2w = w2c[view_i].inverse()
        c2w_t = c2w[:3, 3]
        c2w_rot = c2w[:3, :3]; del c2w
        render = renderer.render_to_camera(c2w_t, c2w_rot, ['world_ray_dirs', 'world_normals'], cull_back_faces=True)
        del c2w_t, c2w_rot

        # 'Compute diffuse shading with point light at camera' >> logger.debug
        cos = torch.einsum('ijk,ijk->ij', render['world_ray_dirs'], render['world_normals']); del render
        shading = cos.neg_().clamp_(0, 1); del cos
        if ss is not None:
            shading = (torch.nn.functional.avg_pool2d(shading.unsqueeze(0).unsqueeze(1), ss, ss) /
                       torch.nn.functional.avg_pool2d((shading > 0).to(dtype).unsqueeze(0).unsqueeze(1), ss, ss)
                       ).squeeze(1).squeeze(0)
            shading = shading.where(shading.isfinite(), shading.new_tensor(0))
        is_valid = shading > 0

        # 'Load IR img' >> logger.debug
        img = helper.load_ir_img(cam_name, mode, view_i)
        if cam_name in {'kinect_v2', 'phone_left', 'phone_right'}:
            # 'Rescale IR img' >> logger.debug
            img = img.to(dtype) / img[is_valid].max()
            img = img.mul_(255).round_().clamp_(0, 255).byte()

        # 'Crop' >> logger.debug
        i_min, i_max, j_min, j_max = get_trim(is_valid); del is_valid
        img = img[i_min:i_max, j_min:j_max]
        shading = shading[i_min:i_max, j_min:j_max]

        # 'Save' >> logger.debug
        Image.fromarray(img.numpy()).save(f'{log_dir}/{view_i:04}_ir.jpg'); del img
        shading = shading.mul_(255).round_().clamp_(0, 255).byte()
        Image.fromarray(shading.numpy()).save(f'{log_dir}/{view_i:04}_shading.jpg'); del shading
        crop_stats[view_i] = dict(i_min=i_min, j_min=j_min)
    torch.save(crop_stats, f'{log_dir}/crop_stats.pt')
    'Done' >> logger.debug


if __name__ == '__main__':
    main()
