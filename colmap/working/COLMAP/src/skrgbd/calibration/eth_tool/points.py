import numpy as np
import yaml


class Points:
    r"""Represents calib board features.

    Parameters
    ----------
    pts : np.ndarray
        of shape [pts_n, 3], 3D coordinates of the calib board features, in the order of the features.
    """
    def __init__(self, pts):
        self.pts = pts

    @classmethod
    def from_yaml(cls, pts_yaml, dtype=np.float64):
        r"""Reads Points from points.yaml.

        Parameters
        ----------
        pts_yaml : str
        dtype : np.dtype

        Returns
        -------
        pts : Points
        """
        with open(pts_yaml, 'r') as file:
            data = yaml.load(file, yaml.SafeLoader)

        feat_i_to_point_j_data = data['feature_id_to_point_index']
        feat_i_to_point_j = np.full([len(feat_i_to_point_j_data)], -1, dtype=np.int64)
        for pair in feat_i_to_point_j_data:
            feat_i_to_point_j[pair['feature_id']] = pair['point_index']
        if (feat_i_to_point_j == -1).any():
            raise RuntimeError(f'Non-bijective feature_id_to_point_index mapping')
        del feat_i_to_point_j_data

        pts = np.array(data['points'], dtype=dtype).reshape(-1, 3); del data
        pts = pts[feat_i_to_point_j]; del feat_i_to_point_j

        pts = cls(pts)
        return pts

    def save_yaml(self, pts_yaml):
        r"""Saves Points to points.yaml

        Parameters
        ----------
        pts_yaml : str
        """
        data = dict()

        data['points'] = self.pts.ravel().tolist()
        data['feature_id_to_point_index'] = [dict(feature_id=i, point_index=i) for i in range(len(self.pts))]
        with open(pts_yaml, 'w') as file:
            yaml.dump(data, file)
