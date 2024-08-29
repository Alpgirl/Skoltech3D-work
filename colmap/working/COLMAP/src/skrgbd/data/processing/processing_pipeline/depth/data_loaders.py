import numpy as np
import torch

from skrgbd.calibration.camera_models.camera_model import CameraModel
from skrgbd.data.dataset.scene_paths import ScenePaths
from skrgbd.utils.logging import tqdm


class WallDataset:
    r"""Represents depth calibration data captured with the calibration wall.

    Parameters
    ----------
    data_npy : str
    cam_model : CameraModel
    """
    def __init__(self, data_npy, cam_model):
        data = np.load(data_npy)

        i = np.ascontiguousarray(data['i'])
        j = np.ascontiguousarray(data['j'])
        pix_uvs = cam_model.get_pix_uvs(); del cam_model
        self.uv = pix_uvs.permute(1, 2, 0)[i, j]; del pix_uvs, i, j

        self.raw_d = torch.from_numpy(np.ascontiguousarray(data['d']))
        self.sl_d = torch.from_numpy(np.ascontiguousarray(data['d_sl'])); del data

    def __len__(self):
        return len(self.uv)

    def __getitem__(self, i):
        return self.uv[i], self.raw_d[i], self.sl_d[i]


class SLDataset:
    r"""Represents depth calibration data computed from the SL scans.

    Parameters
    ----------
    scene_paths : iterable of ScenePaths
    cam_name : str
    cam_model : CameraModel
    """
    def __init__(self, scene_paths, cam_name, cam_model):
        modes = ['i', 'j', 'd', 'd_sl', 'cos']
        dataset = {mode: [] for mode in (modes + ['view_i'])}
        for paths in tqdm(scene_paths):
            undist_data = paths.depth_undist_data(cam_name)
            undist_data = torch.load(undist_data)
            for (view_i, light), data in undist_data.items():
                for mode in modes:
                    dataset[mode].append(data[mode])
                dataset['view_i'].append(np.full([len(data)], view_i, dtype=np.int64))
                del data
            del undist_data
        for mode in list(dataset.keys()):
            dataset[mode] = torch.from_numpy(np.concatenate(dataset[mode]))

        self.raw_d = dataset['d']; del dataset['d']
        self.sl_d = dataset['d_sl']; del dataset['d_sl']
        self.cos = dataset['cos']; del dataset['cos']
        self.view_i = dataset['view_i']; del dataset['view_i']

        i, j = dataset['i'], dataset['j']; del dataset
        pix_uvs = cam_model.get_pix_uvs(); del cam_model
        self.uv = pix_uvs.permute(1, 2, 0)[i, j]; del pix_uvs, i, j

    def __len__(self):
        return len(self.uv)

    def __getitem__(self, i):
        return self.uv[i], self.raw_d[i], self.sl_d[i]


class Loader:
    r"""Equivalent of torch.uitls.data.DataLoader but loads whole batches of data at once instead of element-by-element."""
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(len(self.dataset))
        else:
            perm = torch.arange(len(self.dataset))
        for batch_start in range(0, len(self.dataset), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(self.dataset))
            ids = perm[batch_start:batch_end]
            yield self.dataset[ids]
