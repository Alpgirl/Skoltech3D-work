import torch

from colmap.read_write_model import read_cameras_text

from .camera_model import CameraModel
from .pinhole import PinholeCameraModel
from .rv_camera_model import RVCameraModel


def save_to_pt(model_dict_pt, model):
    r"""Saves the camera model to a pytorch .pt file.

    Parameters
    ----------
    model_dict_pt : str
    model : CameraModel
    """
    model_dict = dict()
    model_dict['class'] = model.__class__.__name__
    state_dict = model.state_dict()
    state_dict_cpu = state_dict.__class__()
    for k, v in state_dict.items():
        state_dict_cpu[k] = v.cpu()
    model_dict['state_dict'] = state_dict_cpu
    return torch.save(model_dict, model_dict_pt)


def load_from_pt(model_dict_pt):
    r"""Loads a camera model from the pytorch .pt file.

    Parameters
    ----------
    model_dict_pt : str

    Returns
    -------
    cam_model : CameraModel
    """
    model_dict = torch.load(model_dict_pt, map_location='cpu')
    state_dict = model_dict['state_dict']
    model = model_by_name[model_dict['class']].from_state_dict(state_dict)
    model.load_state_dict(state_dict)
    return model


def load_from_colmap_txt(cameras_txt, cam_i=None):
    r"""Loads a camera model from COLMAP cameras.txt.

    Parameters
    ----------
    cameras_txt : str
    cam_i : int
        Camera id in cameras.txt. If None, assume only one camera in cameras.txt.

    Returns
    -------
    camera_model : CameraModel
    """
    colmap_cams = read_cameras_text(cameras_txt)
    if cam_i is None:
        if len(colmap_cams.keys()) != 1:
            raise RuntimeError(f'There are {len(colmap_cams.keys())} cameras in {cameras_txt}')
        cam_i = list(colmap_cams.keys())[0]
    colmap_cam = colmap_cams[cam_i]
    cam_model = colmap_model_by_name[colmap_cam.model].from_colmap(colmap_cam)
    return cam_model


colmap_model_by_name = {
    'PINHOLE': PinholeCameraModel
}

model_by_name = {
    'RVCameraModel': RVCameraModel
}
