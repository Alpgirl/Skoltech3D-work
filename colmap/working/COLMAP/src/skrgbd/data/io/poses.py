import numpy as np
import scipy.spatial
import torch

from colmap.read_write_model import Image, read_images_text, write_images_text

from skrgbd.data.dataset.params import cam_pos_ids


class Poses:
    index = None
    img_ids = None
    world_to_cam = None

    @classmethod
    def from_colmap(cls, images_txt, dtype=torch.double):
        colmap_images = read_images_text(images_txt)
        index = dict()
        img_ids = torch.empty(len(colmap_images), dtype=torch.long)
        world_to_cam = torch.zeros(len(colmap_images), 4, 4, dtype=dtype)
        world_to_cam[:, 3, 3] = 1
        rotquats_wxyz = torch.empty(len(colmap_images), 4, dtype=dtype)

        for i, (img_i, data) in enumerate(colmap_images.items()):
            index[img_i] = i
            img_ids[i] = img_i
            world_to_cam.numpy()[i, :3, 3] = data.tvec
            rotquats_wxyz.numpy()[i] = data.qvec

        xyzw = np.roll(rotquats_wxyz, -1, 1); del rotquats_wxyz
        rotmat = scipy.spatial.transform.Rotation.from_quat(xyzw).as_matrix(); del xyzw
        world_to_cam.numpy()[:, :3, :3] = rotmat; del rotmat

        poses = cls()
        poses.index = index
        poses.img_ids = img_ids
        poses.world_to_cam = world_to_cam
        return poses

    def to(self, device=None, dtype=None):
        self.world_to_cam = self.world_to_cam.to(device=device, dtype=dtype)
        self.img_ids = self.img_ids.to(device=device)
        return self

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, img_i):
        return self.world_to_cam[self.index[img_i]]


def save_poses(images_txt, poses, cam_i=0):
    r"""Converts camera poses from world-to-cam 4x4 matrix format to COLMAP TXT format.

    Parameters
    ----------
    images_txt : str
        Path to images.txt to save the poses to.
    poses : iterable of dict
        img_i: int
            ID of the pose. COLMAP uses one-based indices.
        w2c: torch.Tensor
            of shape [4, 4].
        filename: str
            Name of the respective image file.
    cam_i : int
        CAMERA_ID that will be used in the images.txt.
    """
    colmap_images = dict()
    for pose in poses:
        img_i = pose['img_i']
        rotmat = pose['w2c'][:3, :3].numpy()
        rotmat = scipy.spatial.transform.Rotation.from_matrix(rotmat)
        xyzw = rotmat.as_quat()
        wxyz = np.roll(xyzw, 1)
        tvec = pose['w2c'][:3, 3].numpy()
        filename = pose['filename']
        colmap_images[img_i] = Image(id=img_i, qvec=wxyz.astype(np.float32),tvec=tvec.astype(np.float32),
                                     camera_id=cam_i, name=filename, xys=[], point3D_ids=[])
    return write_images_text(colmap_images, images_txt)


def load_poses(images_txt, dtype=torch.double, pos_ids=cam_pos_ids):
    r"""Loads camera poses from COLMAP TXT format.

    Parameters
    ----------
    images_txt : str
    dtype : torch.dtype

    Returns
    -------
    world_to_cam : torch.Tensor
        of shape [views_n, 4, 4].
    """
    poses = Poses.from_colmap(images_txt, dtype)
    world_to_cam = torch.empty(len(pos_ids), 4, 4, dtype=dtype)
    for pos_i in pos_ids:
        world_to_cam[pos_i].copy_(poses[pos_i + 1])  # COLMAP's image_id is one-based
    del poses
    return world_to_cam
