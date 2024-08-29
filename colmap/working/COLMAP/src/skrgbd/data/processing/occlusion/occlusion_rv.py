import torch

from skrgbd.data.processing.occlusion.occlusion import OcclusionHelper as OcclusionHelperBase
raise DeprecationWarning  # from skrgbd.calibration.camera_models.rv_camera_model import RVCameraModel
from skrgbd.data.processing.mesh_utils.triangulation import triangulate_grid


class OcclusionHelper(OcclusionHelperBase):
    def init_partial_scans(self, pathfinder, scene_name, scan_ids=None):
        scan_ids = scan_ids or list(range(27))
        scan_paths = dict()
        for scan_i in scan_ids:
            for cam_i in [0, 1]:
                scan_paths[scan_i * 2 + cam_i] = pathfinder[scene_name].stl.partial.aligned[scan_i]
        self.partial_scans = scan_paths

        pix_rays = []
        cam_to_board = []
        for cam_i in ['a', 'b']:
            cam_model = f'{pathfinder[scene_name].stl.partial.raw}/scan_res_0000/Raw/impar{cam_i}01.txt'
            cam_model = RVCameraModel(cam_model)
            raise DeprecationWarning  # cam_to_board.append(cam_model.camera_to_board)
            pix_rays.append(cam_model.get_pixel_rays())
            del cam_model

        cam_to_board = torch.stack(cam_to_board)
        refined_board_to_world = torch.load(pathfinder[scene_name].stl.partial.aligned.refined_board_to_world)
        cam_to_world = refined_board_to_world.double().unsqueeze(1) @ cam_to_board
        self.cam_to_world = cam_to_world.view(-1, 4, 4).to(self.device, self.dtype)
        del refined_board_to_world, cam_to_board, cam_to_world

        assert pix_rays[0].shape == pix_rays[1].shape
        h, w = pix_rays[0].shape[1:]
        pix_tris = triangulate_grid(h, w)
        self.pix_tris = {scan_i: pix_tris for scan_i in scan_paths.keys()}

        self.pix_rays = dict()
        for cam_i in range(2):
            rays = pix_rays[0]
            rays = rays.permute(1, 2, 0).to('cpu', torch.float, memory_format=torch.contiguous_format).view(-1, 3)
            for scan_i in scan_ids:
                self.pix_rays[scan_i * 2 + cam_i] = rays
