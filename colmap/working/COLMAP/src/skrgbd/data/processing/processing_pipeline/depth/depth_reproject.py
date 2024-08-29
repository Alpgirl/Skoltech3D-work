from argparse import ArgumentParser
from pathlib import Path

import torch

from skrgbd.data.dataset.params import cam_pos_ids, sensor_to_cam_mode
from skrgbd.calibration.camera_models.camera_model import CameraModel
from skrgbd.calibration.camera_models.central_generic import CentralGeneric
from skrgbd.data.processing.depth_utils.depthmap_rendering import DepthmapRenderer
from skrgbd.data.io import imgio
from skrgbd.calibration.camera_models import load_from_colmap_txt
from skrgbd.data.io.poses import load_poses
from skrgbd.calibration.depth_distortion_models import load_from_pt
from skrgbd.utils.logging import logger, tqdm
from skrgbd.data.processing.processing_pipeline.depth.configs import reproj_configs
from skrgbd.data.dataset.scene_paths import ScenePaths
from skrgbd.calibration.depth_distortion_models.uv_d_cubic_bspline import UvDUndistortionModel


def main():
    description = 'Reprojects sensor depthmaps.'
    parser = ArgumentParser(description=description)
    parser.add_argument('--addons-dir', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--raw-dir', type=str, required=True)
    parser.add_argument('--src-cam', type=str, required=True)
    parser.add_argument('--light', type=str, required=True)
    parser.add_argument('--dst-sensor', type=str, required=True)
    parser.add_argument('--scene-name', type=str, required=True)
    args = parser.parse_args()

    if args.light == 'none':
        args.light = None

    f'Reproj depth {args.scene_name}' >> logger.debug
    scene_paths = ScenePaths(args.scene_name, addons_dir=args.addons_dir, data_dir=args.data_dir, raw_dir=args.raw_dir)
    reproj_depthmaps(scene_paths, args.src_cam, args.dst_sensor, args.light)


def reproj_depthmaps(scene_paths, src_cam, dst_sensor, light=None, var='undist', dtype=torch.double):
    r"""Reprojects sensor depthmaps.

    Parameters
    ----------
    scene_paths : ScenePaths
    src_cam : str
    dst_sensor : str
    light : str
    var : str
    dtype : torch.dtype
    """
    f'Load cam data for {src_cam}' >> logger.debug
    config = reproj_configs[src_cam]
    src_cam_model = scene_paths.cam_model(src_cam, 'ir', 'generic')
    src_cam_model = CentralGeneric(src_cam_model, dtype=dtype)

    src_w2c = scene_paths.cam_poses(src_cam, 'ir')
    src_w2c = load_poses(src_w2c)
    src_c2w = src_w2c.inverse().to(dtype); del src_w2c

    'Load undist model' >> logger.debug
    undist_model = scene_paths.undist_model(src_cam, 'depth')
    undist_model = load_from_pt(undist_model).to(dtype)

    f'Load cam data for {dst_sensor}.{var}' >> logger.debug
    dst_cam, dst_mode = sensor_to_cam_mode[dst_sensor]
    if var == 'undist':
        dst_cam_model = scene_paths.cam_model(dst_cam, dst_mode)
        dst_cam_model = load_from_colmap_txt(dst_cam_model).to(dtype)

    dst_w2c = scene_paths.cam_poses(dst_cam, dst_mode)
    dst_w2c = load_poses(dst_w2c)
    dst_c2w = dst_w2c.inverse().to(dtype); del dst_w2c

    'Render' >> logger.debug
    for view_i in tqdm(cam_pos_ids):
        reproj_img(scene_paths, light, view_i, src_cam, src_cam_model, src_c2w[view_i], dst_sensor,
                   var, dst_cam_model, dst_c2w[view_i], undist_model, dtype=torch.double, **config)
    'Done' >> logger.debug


def reproj_img(scene_paths, light, view_i, src_cam, src_cam_model, src_c2w, dst_sensor,
               dst_var, dst_cam_model, dst_c2w, undist_model, max_rel_edge_len, dtype=torch.double):
    r"""Reprojects a sensor depthmap.

    Parameters
    ----------
    scene_paths : ScenePaths
    light : str
    view_i : int
    src_cam : str
    src_cam_model : CameraModel
    src_c2w : torch.Tensor
        of shape [4, 4].
    dst_sensor : str
    dst_var : str
    dst_cam_model : CameraModel
    dst_c2w : torch.Tensor
        of shape [4, 4].
    undist_model : UvDUndistortionModel
    max_rel_edge_len : float
    dtype : torch.dtype
    """
    # 'Load sensor depth' >> logger.debug
    raw_depthmap = scene_paths.img(src_cam, 'depth', view_i, light, 'raw')
    raw_depthmap = imgio.read[src_cam].raw_depth(raw_depthmap)
    raw_depthmap = torch.from_numpy(raw_depthmap).to(dtype).div_(1000)

    # 'Undistort in depth dimension' >> logger.debug
    undist_depthmap = undist_model(raw_depthmap); del undist_model, raw_depthmap

    # 'Undistort in image dimension' >> logger.debug
    renderer = DepthmapRenderer(undist_depthmap, undist_depthmap.isfinite(), src_cam_model, src_c2w, max_rel_edge_len)
    del undist_depthmap, src_cam_model, src_c2w
    renderer.set_rays_from_camera(dst_cam_model); del dst_cam_model

    c2w_t, c2w_rot = dst_c2w[:3, 3], dst_c2w[:3, :3]; del dst_c2w
    reproj_depth = renderer.render_to_camera(c2w_t, c2w_rot, ['z_depth'], cull_back_faces=True)['z_depth']
    del renderer, c2w_t, c2w_rot
    reproj_depth = reproj_depth.where(reproj_depth.isfinite(), reproj_depth.new_tensor(float('nan')))

    src_var = 'undist'
    depthmap_png = scene_paths.proj_depth(src_cam, src_var, dst_sensor, dst_var, view_i, light)
    # f'Save {depthmap_png}' >> logger.debug
    Path(depthmap_png).parent.mkdir(exist_ok=True, parents=True)
    reproj_depth = reproj_depth.float().numpy()
    imgio.write[src_cam].undist_depth(depthmap_png, reproj_depth); del reproj_depth


if __name__ == '__main__':
    main()
