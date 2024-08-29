from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from skrgbd.data.dataset.params import cam_pos_ids, light_setups
from skrgbd.calibration.camera_models.central_generic import CentralGeneric
from skrgbd.data.processing.processing_pipeline.depth.configs import sl_configs
from skrgbd.data.io import imgio
from skrgbd.data.io.poses import load_poses
from skrgbd.utils.logging import logger, tqdm
from skrgbd.data.processing.depth_utils.occluded_mesh_rendering import MeshRenderer
from skrgbd.data.dataset.scene_paths import ScenePaths


def main():
    description = 'Calculates depth undistortion data.'
    parser = ArgumentParser(description=description)
    parser.add_argument('--aux-dir', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--raw-dir', type=str, required=True)
    parser.add_argument('--cam', type=str, required=True)
    parser.add_argument('--scene-name', type=str, required=True)
    args = parser.parse_args()

    f'Calc undist data SL {args.scene_name}' >> logger.debug
    scene_paths = ScenePaths(args.scene_name, aux_dir=args.aux_dir, data_dir=args.data_dir, raw_dir=args.raw_dir)
    calc_undist_data_sl(scene_paths, args.cam)


def calc_undist_data_sl(scene_paths, cam, device='cpu', dtype=torch.float32):
    r"""Calculates depth undistortion data.

    Parameters
    ----------
    scene_paths : ScenePaths
    cam : str
    device : torch.device
    dtype : torch.dtype
    """
    f'Load cam data for {cam}' >> logger.debug
    config = sl_configs[cam, 'ir']
    cam_model = scene_paths.cam_model(cam, 'ir', 'generic')
    cam_model = CentralGeneric(cam_model, dtype=dtype).to(device)

    w2c = scene_paths.cam_poses(cam, 'ir', 'ref')
    w2c = load_poses(w2c)
    c2w = w2c.inverse().to(device, dtype); del w2c

    'Load SL data' >> logger.debug
    rec = scene_paths.sl_full()
    rec = o3d.io.read_triangle_mesh(rec)
    occ = scene_paths.sl_occ()
    occ = o3d.io.read_triangle_mesh(occ)

    'Initialize renderer' >> logger.debug
    renderer = Renderer(rec, occ, config.occ_thres); del rec, occ
    renderer.set_rays_from_camera(cam_model, config.ss); del cam_model

    'Render' >> logger.debug
    undist_data = dict()
    lights = light_setups if (cam == 'real_sense') else [None]
    for view_i in tqdm(cam_pos_ids):
        # 'Render SL depth' >> logger.debug
        c2w_t, c2w_rot = c2w[view_i, :3, 3], c2w[view_i, :3, :3]
        depthmap, cos, depthmap_aa, sl_is_valid = renderer.render(c2w_t, c2w_rot); del c2w_t, c2w_rot

        for light in lights:
            # 'Load sensor depth' >> logger.debug
            raw_depthmap = scene_paths.img(cam, 'depth', view_i, light, 'raw')
            raw_depthmap = imgio.read[cam].raw_depth(raw_depthmap)
            raw_depthmap = torch.from_numpy(raw_depthmap).to(dtype).div_(1000)

            # 'Load sensor IR' >> logger.debug
            raw_ir = scene_paths.img(cam, 'ir', view_i, light, 'raw')
            raw_ir = imgio.read[cam].ir(raw_ir)
            raw_ir = torch.from_numpy(raw_ir).to(dtype)

            raw_is_valid = raw_depthmap > .1
            is_valid = raw_is_valid.logical_and_(sl_is_valid); del raw_is_valid
            i, j = is_valid.nonzero(as_tuple=True); del is_valid

            data = np.empty([len(i)], dtype=[('i', np.int64), ('j', np.int64), ('d', np.float32), ('ir', np.float32),
                                             ('d_sl', np.float32), ('d_sl_aa', np.float32), ('cos', np.float32)])
            data['i'] = i.numpy()
            data['j'] = j.numpy()
            data['d'] = raw_depthmap[i, j].numpy(); del raw_depthmap
            data['ir'] = raw_ir[i, j].numpy(); del raw_ir
            data['d_sl'] = depthmap[i, j]
            data['d_sl_aa'] = depthmap_aa[i, j]
            data['cos'] = cos[i, j]; del i, j
            undist_data[view_i, light] = data; del data
        del depthmap, cos, depthmap_aa, sl_is_valid

    undist_data_pt = scene_paths.depth_undist_data(cam)
    f'Save undist data to {undist_data_pt}' >> logger.debug
    Path(undist_data_pt).parent.mkdir(exist_ok=True, parents=True)
    torch.save(undist_data, undist_data_pt)
    'Done' >> logger.debug


class Renderer(MeshRenderer):
    ss_rays = None

    def set_rays_from_camera(self, cam_model, ss):
        r"""Initializes an additional set of camera rays for the super-sampled camera model."""
        super().set_rays_from_camera(cam_model)

        cam_model = cam_model.resize_(cam_model.size_wh * ss)
        rays = cam_model.get_pixel_rays()
        rays = rays.view(3, self.h, ss, self.w, ss).permute(1, 3, 2, 4, 0).to(self.device, self.dtype,
                                                                              memory_format=torch.contiguous_format)
        self.ss_rays = rays.reshape(self.h, self.w, -1, 3)

    def render(self, c2w_t, c2w_rot):
        # 'Render' >> logger.debug
        c2w_rot = c2w_rot.to(self.rays)
        render = self.render_to_camera(c2w_t, c2w_rot,
                                       ['z_depth', 'world_normals', 'world_ray_dirs'], cull_back_faces=True)
        cos = torch.einsum('ijk,ijk->ij', render['world_ray_dirs'], render['world_normals'])
        depthmap = render['z_depth']; del render
        is_valid = depthmap.isfinite()
        depthmap = depthmap.where(is_valid, depthmap.new_tensor(float('nan')))

        # 'Render with ss' >> logger.debug
        rays = self.ss_rays[is_valid]  # n, ssss, 3
        rays_n = rays.shape[0]
        rays = rays.reshape(-1, 3)

        casted_rays = self.make_rays_from_cam(c2w_t, c2w_rot, rays); del c2w_t, c2w_rot
        render = self.render_rays(casted_rays, cull_back_faces=True); del casted_rays
        ray_depth = render['ray_hit_depth']; del render
        depth_aa = ray_depth.mul_(rays[:, 2]); del ray_depth, rays
        depth_aa = depth_aa.view(rays_n, -1).mean(1)
        depth_aa = depth_aa.where(depth_aa.isfinite(), depth_aa.new_tensor(float('nan')))

        depthmap_aa = torch.full_like(depthmap, float('nan'))
        depthmap_aa.masked_scatter_(is_valid, depth_aa); del depth_aa
        return depthmap, cos, depthmap_aa, is_valid


if __name__ == '__main__':
    main()
