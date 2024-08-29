import numpy as np
from torch.multiprocessing import Pool
import scipy.optimize
import torch
from tqdm.auto import tqdm

from skrgbd.calibration.eth_tool.dataset import Dataset
from skrgbd.calibration.eth_tool.ex_poses import Poses


class Calculator:
    def __init__(self, dataset_file, camera_tr_rig_file, camera_models, img_ids=None):
        dataset = Dataset.fromfile(dataset_file)
        cam_ids = list(range(dataset.cameras_n))
        rig_to = Poses(camera_tr_rig_file)
        camera_models = camera_models

        all_features = [[] for _ in cam_ids]
        if img_ids is None:
            img_ids = dataset.image_data.keys()
        for img_i in img_ids:
            image_data = dataset.image_data[img_i]

            cam_features = [set(image_data[cam_i]['features_id']) for cam_i in cam_ids]
            common_features = cam_features[0].intersection(*cam_features[1:])
            common_features = np.array(list(common_features))

            for cam_i in cam_ids:
                data = dataset.image_data[img_i][cam_i]
                _ = data['features_id']
                selection = (_[:, None] == common_features[None]).any(-1)
                feature_uv = data['features_xy'][selection]
                all_features[cam_i].append(feature_uv)
        all_features = np.stack([np.concatenate(_, 0) for _ in all_features])

        directions = []
        camera_poses = []
        uv_per_xyz_resid_at_unit_depth = []
        d = .01

        for cam_i, camera_model in zip(cam_ids, camera_models):
            uv = all_features[cam_i].T
            uv = torch.from_numpy(uv)
            directions_in_camera = camera_model.unproject(uv)
            _ = uv + uv.new_tensor([d, 0]).unsqueeze(1)
            _ = camera_model.unproject(_)
            _ = d / (_ - directions_in_camera).norm(dim=0)
            uv_per_xyz_resid_at_unit_depth.append(_)

            camera_to_rig = rig_to[cam_i].inverse()
            directions_in_rig = camera_to_rig[:3, :3] @ directions_in_camera
            directions.append(directions_in_rig)
            camera_poses.append(camera_to_rig[:3, 3])

        directions = torch.stack(directions)
        uv_per_xyz_resid_at_unit_depth = torch.stack(uv_per_xyz_resid_at_unit_depth)
        self.camera_poses = torch.stack(camera_poses)

        _ = uv_per_xyz_resid_at_unit_depth.isfinite().all(0)
        self.directions = directions[:, :, _]
        self.uv_per_xyz_resid_at_unit_depth = uv_per_xyz_resid_at_unit_depth[:, _]
        del _
        self.pts_n = self.directions.shape[-1]

    def calculate_errors(self, batch_size=1, verbose=0, workers_n=None, camera_weights=None, progress=True):
        if batch_size is None:
            batch_size = self.pts_n
        self.batch_size = batch_size
        self.verbose = verbose
        if camera_weights is not None:
            self.camera_weights = torch.tensor(camera_weights)
        else:
            self.camera_weights = None

        with Pool(workers_n) as pool:
            batch_starts = list(range(0, self.pts_n, self.batch_size))
            it = pool.imap(self.solve, batch_starts)
            if progress:
                it = tqdm(it, total=len(batch_starts))
            solution = np.concatenate(list(it), 0)
        return solution

    def solve(self, batch_start):
        batch_end = min(batch_start + self.batch_size, self.pts_n)
        solution = scipy.optimize.least_squares(
            self.cost_fn,
            x0=np.ones(3 * (batch_end - batch_start)),
            method='lm',
            verbose=self.verbose,
            kwargs=dict(batch_start=batch_start, batch_end=batch_end, camera_weights=self.camera_weights),
        )
        return solution.x.reshape(3, -1).T

    def cost_fn(self, x, batch_start=0, batch_end=None, camera_weights=None):
        if batch_end is None:
            batch_end = self.pts_n
        points = torch.from_numpy(x).view(3, -1)
        points = (points - self.camera_poses.unsqueeze(-1))  # cams_n, 3, batch_size
        cos = (points * self.directions[..., batch_start:batch_end]).sum(1, keepdim=True)
        residuals = points - self.directions[..., batch_start:batch_end] * cos
        norms = residuals.norm(dim=1)
        if camera_weights is not None:
            norms = norms * camera_weights.unsqueeze(1)
        return norms.numpy().ravel()

    def dir_residuals(self, x):
        points = torch.from_numpy(x).view(3, -1)
        points = (points - self.camera_poses.unsqueeze(-1))  # cams_n, 3, batch_size
        dir_to_points = torch.nn.functional.normalize(points, dim=1)
        residuals = dir_to_points - self.directions
        return residuals
