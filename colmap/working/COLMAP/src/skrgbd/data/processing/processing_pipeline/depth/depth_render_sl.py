from argparse import ArgumentParser
from pathlib import Path

import open3d as o3d
import torch

from skrgbd.data.dataset.params import cam_pos_ids, sensor_to_cam_mode
from skrgbd.data.io import imgio
from skrgbd.calibration.camera_models import load_from_colmap_txt
from skrgbd.data.io.poses import load_poses
from skrgbd.utils.logging import logger, tqdm
from skrgbd.data.processing.depth_utils.occluded_mesh_rendering import MeshRenderer
from skrgbd.data.dataset.scene_paths import ScenePaths


def main():
    description = 'Renders SL scan as depth maps at image sensor coordinates.'
    parser = ArgumentParser(description=description)
    parser.add_argument('--addons-dir', type=str, required=True)
    parser.add_argument('--aux-dir', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--sensor', type=str, required=True)
    parser.add_argument('--supersampling', type=int, required=False)
    parser.add_argument('--scene-name', type=str, required=True)
    args = parser.parse_args()
    if args.supersampling == 1:
        args.supersampling = None

    f'Depth render SL {args.scene_name}' >> logger.debug
    scene_paths = ScenePaths(args.scene_name, addons_dir=args.addons_dir, aux_dir=args.aux_dir, data_dir=args.data_dir)
    render_sl_depth(scene_paths, args.sensor, args.supersampling)


def render_sl_depth(scene_paths, sensor, ss=None, var='undist', depth_var='z_depth', occ_thres=1e-3, poses_var='ref'):
    r"""Renders SL scan as depth maps at image sensor coordinates.

    Parameters
    ----------
    scene_paths : ScenePaths
    sensor : str
    ss : int
    var : {'undist'}
    depth_var : {'ray_depth', 'z_depth'}
    occ_thres : float
    poses_var : {'calib', 'ref'}
    """
    device, dtype = 'cpu', torch.float32

    cam, mode = sensor_to_cam_mode[sensor]
    f'Load cam data for {cam}.{mode}.{var}, {poses_var} poses' >> logger.debug
    if var == 'undist':
        cam_model = scene_paths.cam_model(cam, mode)
        cam_model = load_from_colmap_txt(cam_model).to(device, dtype)
    if ss is not None:
        f'Supersample camera model: {ss}' >> logger.debug
        cam_model = cam_model.resize_(cam_model.size_wh * ss)

    w2c = scene_paths.cam_poses(cam, mode, poses_var)
    w2c = load_poses(w2c)
    c2w = w2c.inverse().to(device, dtype); del w2c

    'Load SL data' >> logger.debug
    rec = scene_paths.sl_full()
    rec = o3d.io.read_triangle_mesh(rec)
    occ = scene_paths.sl_occ()
    occ = o3d.io.read_triangle_mesh(occ)

    'Initialize renderer' >> logger.debug
    renderer = MeshRenderer(rec, occ, occ_thres); del rec, occ
    renderer.set_rays_from_camera(cam_model); del cam_model

    'Render' >> logger.debug
    src, src_var = 'stl', 'clean_rec'
    if ss is not None:
        src_var = f'{src_var}.aa'
    for view_i in tqdm(cam_pos_ids):
        c2w_t, c2w_rot = c2w[view_i, :3, 3], c2w[view_i, :3, :3]
        render = renderer.render_to_camera(c2w_t, c2w_rot, [depth_var], cull_back_faces=True); del c2w_t, c2w_rot
        depthmap = render[depth_var]; del render
        depthmap = depthmap.where(depthmap.isfinite(), depthmap.new_tensor(float('nan')))
        if ss is not None:
            depthmap = torch.nn.functional.avg_pool2d(depthmap.unsqueeze(0).unsqueeze(1), ss, ss).squeeze(1).squeeze(0)

        depthmap_png = scene_paths.proj_depth(src, src_var, sensor, var, view_i)
        # f'Save depthmap to {depthmap_png}' >> logger.debug
        Path(depthmap_png).parent.mkdir(exist_ok=True, parents=True)
        depthmap = depthmap.float().numpy()
        imgio.write.stl.depth(depthmap_png, depthmap); del depthmap
    'Done' >> logger.debug


if __name__ == '__main__':
    main()
