from argparse import ArgumentParser
from pathlib import Path

import torch

from skrgbd.data.dataset.params import cam_pos_ids, light_setups
from skrgbd.calibration.camera_models.camera_model import CameraModel
from skrgbd.calibration.camera_models.central_generic import CentralGeneric
from skrgbd.data.processing.depth_utils.depthmap_rendering import DepthmapRenderer
from skrgbd.data.io import imgio
from skrgbd.calibration.camera_models import load_from_colmap_txt
from skrgbd.calibration.depth_distortion_models import load_from_pt
from skrgbd.utils.logging import logger, tqdm
from skrgbd.data.processing.processing_pipeline.depth.configs import reproj_configs
from skrgbd.data.dataset.scene_paths import ScenePaths
from skrgbd.calibration.depth_distortion_models.uv_d_cubic_bspline import UvDUndistortionModel


def main():
    description = 'Undistorts raw depthmaps.'
    parser = ArgumentParser(description=description)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--raw-dir', type=str, required=True)
    parser.add_argument('--cam', type=str, required=True)
    parser.add_argument('--scene-name', type=str, required=True)
    args = parser.parse_args()

    f'Undist depth {args.scene_name}' >> logger.debug
    scene_paths = ScenePaths(args.scene_name, data_dir=args.data_dir, raw_dir=args.raw_dir)
    undist_depthmaps(scene_paths, args.cam)


def undist_depthmaps(scene_paths, cam, dtype=torch.double):
    r"""Undistorts raw depthmaps.

    Parameters
    ----------
    scene_paths : ScenePaths
    cam : str
    dtype : torch.dtype
    """
    f'Load cam data for {cam}' >> logger.debug
    config = reproj_configs[cam]
    cam_model = scene_paths.cam_model(cam, 'ir', 'generic')
    cam_model = CentralGeneric(cam_model, dtype=dtype)

    pinhole_cam_model = scene_paths.cam_model(cam, 'ir')
    pinhole_cam_model = load_from_colmap_txt(pinhole_cam_model).to(dtype)

    'Load undist model' >> logger.debug
    undist_model = scene_paths.undist_model(cam, 'depth')
    undist_model = load_from_pt(undist_model).to(dtype)

    'Undistort' >> logger.debug
    if cam == 'real_sense':
        pairs = [(light, view_i) for light in light_setups for view_i in cam_pos_ids]
    else:
        pairs = [(None, view_i) for view_i in cam_pos_ids]
    for (light, view_i) in tqdm(pairs):
        undist_img(scene_paths, cam, light, view_i, cam_model, pinhole_cam_model, undist_model, dtype=dtype, **config)
    'Done' >> logger.debug


def undist_img(scene_paths, cam, light, view_i, src_cam_model, dst_cam_model,
               undist_model, max_rel_edge_len, dtype=torch.double):
    r"""Undistorts a raw depthmap.

    Parameters
    ----------
    scene_paths : ScenePaths
    cam : str
    light : str
    view_i : int
    src_cam_model : CameraModel
    dst_cam_model : CameraModel
    undist_model : UvDUndistortionModel
    max_rel_edge_len : float
        Triangles with edges longer than the distance to the triangle from the depthmap source camera multiplied by
        `max_rel_edge_len` are considered false boundaries.
    dtype : torch.dtype
    """
    # 'Load sensor depth' >> logger.debug
    raw_depthmap = scene_paths.img(cam, 'depth', view_i, light, 'raw')
    raw_depthmap = imgio.read[cam].raw_depth(raw_depthmap)
    raw_depthmap = torch.from_numpy(raw_depthmap).to(dtype).div_(1000)

    # 'Undistort in depth dimension' >> logger.debug
    undist_depthmap = undist_model(raw_depthmap); del undist_model, raw_depthmap

    # 'Undistort in image dimension' >> logger.debug
    renderer = DepthmapRenderer(undist_depthmap, undist_depthmap.isfinite(), src_cam_model, max_rel_edge_len=max_rel_edge_len)
    del undist_depthmap, src_cam_model
    renderer.set_rays_from_camera(dst_cam_model); del dst_cam_model

    c2w_t, c2w_rot = torch.zeros(3, dtype=dtype), torch.eye(3, dtype=dtype)
    reproj_depth = renderer.render_to_camera(c2w_t, c2w_rot, ['z_depth'], cull_back_faces=True)['z_depth']
    del renderer, c2w_t, c2w_rot
    reproj_depth = reproj_depth.where(reproj_depth.isfinite(), reproj_depth.new_tensor(float('nan')))

    depthmap_png = scene_paths.img(cam, 'depth', view_i, light, 'undist')
    # f'Save {depthmap_png}' >> logger.debug
    Path(depthmap_png).parent.mkdir(exist_ok=True, parents=True)
    reproj_depth = reproj_depth.float().numpy()
    imgio.write[cam].undist_depth(depthmap_png, reproj_depth); del reproj_depth


if __name__ == '__main__':
    main()
