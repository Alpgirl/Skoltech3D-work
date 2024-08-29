import torch

from .uv_d_cubic_bspline import PixBatchUvDUndistortionModel, UvDUndistortionModel


def save_to_pt(model_dict_pt, model):
    r"""Saves the depth undistortion model to a pytorch .pt file.

    Parameters
    ----------
    model_dict_pt : str
    model : torch.nn.Module
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
    r"""Loads a depth undistortion model from the pytorch .pt file.

    Parameters
    ----------
    model_dict_pt : str

    Returns
    -------
    model : torch.nn.Module
    """
    model_dict = torch.load(model_dict_pt, map_location='cpu')
    state_dict = model_dict['state_dict']
    model = model_by_name[model_dict['class']].from_state_dict(state_dict)
    model.load_state_dict(state_dict)
    return model


model_by_name = {
    'PixBatchUvDUndistortionModel': PixBatchUvDUndistortionModel,
    'UvDUndistortionModel': UvDUndistortionModel,
}