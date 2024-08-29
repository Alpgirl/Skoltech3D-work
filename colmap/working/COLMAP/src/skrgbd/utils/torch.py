from collections import OrderedDict


def get_sub_dict(state_dict, sub_name):
    r"""Extracts state dict of a submodule.

    Parameters
    ----------
    state_dict : OrderedDict
    sub_name : str

    Returns
    -------
    sub_dict : OrderedDict
    """
    sub_dict = state_dict.__class__()
    pref = f'{sub_name}.'
    for k, v in state_dict.items():
        if k.startswith(pref):
            sub_dict[k[len(pref):]] = v
    return sub_dict
