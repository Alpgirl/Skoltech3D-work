import numpy as np
import yaml


class Poses:
    r"""Represents camera poses.

    Parameters
    ----------
    w2c_t : np.ndarray
        of shape [cams_n, 3], world-to-camera translation vectors.
    w2c_rotq : np.ndarray
        of shape [cams_n, 4], world-to-camera rotation quaternions, in xyzw notation.

    Attributes
    ----------
    is_def : np.ndarray
        of shape [cams_n], False if the pose is undefined.
    """
    def __init__(self, w2c_t, w2c_rotq):
        self.w2c_t = w2c_t
        self.w2c_rotq = w2c_rotq
        self.is_def = np.logical_and(np.isfinite(w2c_t).all(1), np.isfinite(w2c_rotq).all(1))

    @classmethod
    def from_yaml(cls, poses_yaml, dtype=np.float64):
        r"""Reads Poses from poses.yaml.

        Parameters
        ----------
        poses_yaml : str
        dtype : np.dtype

        Returns
        -------
        poses : Poses
        """
        with open(poses_yaml, 'r') as file:
            data = yaml.load(file, yaml.SafeLoader)

        cams_n = data['pose_count']
        w2c_t = np.full([cams_n, 3], float('nan'), dtype=dtype)
        w2c_rotq = np.full([cams_n, 4], float('nan'), dtype=dtype)
        for pose in data['poses']:
            i = pose['index']
            w2c_t[i] = pose['tx'], pose['ty'], pose['tz']
            w2c_rotq[i] = pose['qx'], pose['qy'], pose['qz'], pose['qw']
        del data
        poses = cls(w2c_t, w2c_rotq)
        return poses

    def __len__(self):
        return len(self.w2c_t)

    def save_yaml(self, poses_yaml):
        r"""Saves Poses to poses.yaml

        Parameters
        ----------
        poses_yaml : str
        """
        data = dict()

        data['pose_count'] = len(self)
        data['poses'] = list()
        for i in range(len(self)):
            if not self.is_def[i]:
                continue
            pose = dict(index=i)
            pose['tx'], pose['ty'], pose['tz'] = self.w2c_t[i].tolist()
            pose['qx'], pose['qy'], pose['qz'], pose['qw'] = self.w2c_rotq[i].tolist()
            data['poses'].append(pose); del pose

        with open(poses_yaml, 'w') as file:
            yaml.dump(data, file)
