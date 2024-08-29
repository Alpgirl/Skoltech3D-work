import yaml

from scipy.spatial.transform import Rotation
import torch

from skrgbd.calibration.eth_tool.dataset import Dataset


class Poses:
    def __init__(self, poses_yaml):
        with open(poses_yaml, 'r') as file:
            self.poses = yaml.load(file, yaml.SafeLoader)['poses']

    def __getitem__(self, i):
        return pose_to_extrinsics(self.poses[i])

    def __len__(self):
        return len(self.poses)

    def index(self, i):
        return self.poses[i]['index']


def pose_to_extrinsics(pose):
    extrinsics = torch.zeros([4, 4])
    extrinsics[3, 3] = 1
    extrinsics[0, 3] = pose['tx']
    extrinsics[1, 3] = pose['ty']
    extrinsics[2, 3] = pose['tz']
    r = Rotation.from_quat([pose['qx'], pose['qy'], pose['qz'], pose['qw']]).as_matrix()
    extrinsics.numpy()[:3, :3] = r
    return extrinsics


def get_poses(trajectory, poses_yaml, dataset_bin):
    r"""
    Parameters
    ----------
    trajectory : Trajectory
    poses_yaml : str
    dataset_bin : str

    Returns
    -------
    poses : torch.Tensor
        of shape [trajectory_points_n, 4, 4]
    pose_found : torch.BoolTensor
        of shape [trajectory_points_n]
    """
    poses_data = Poses(poses_yaml)
    image_names = list(Dataset.fromfile(dataset_bin).image_data.keys())

    poses_n = len(trajectory)
    poses = torch.empty(poses_n, 4, 4)
    pose_found = torch.zeros(poses_n, dtype=torch.bool)
    for i in range(len(poses_data)):
        calib_pos_i = poses_data.index(i)
        point_id = image_names[calib_pos_i]
        point_id = trajectory.name + point_id.split(trajectory.name)[1]
        if point_id.endswith(('.png', '.jpg')):
            point_id = point_id[:-4]
        if point_id not in trajectory:
            continue
        pos_i = trajectory[point_id]
        assert not pose_found[pos_i]
        poses[pos_i] = poses_data[i]
        pose_found[pos_i] = True
    return poses, pose_found