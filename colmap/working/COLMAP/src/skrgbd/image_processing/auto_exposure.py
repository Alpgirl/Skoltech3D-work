import numpy as np
import scipy.optimize
import torch

from skrgbd.utils.logging import logger
from skrgbd.image_processing.utils import get_luma


def pick_camera_settings(preset, get_img, fg_mask, set_exp, exp_low, exp_high, fast_exp,
                         set_gain, gain_low, gain_high, best_gain, exp_high_tol=.9):
    if preset == 'best':
        # Find the best exposure at the lowest gain and then find the best gain at this exposure
        set_gain(best_gain)
        exposure = pick_parameter('AutoExposure', set_exp, get_img, fg_mask, exp_low, exp_high)
        if exposure >= exp_high * exp_high_tol:
            set_exp(exposure)
            gain = pick_parameter('AutoGain', set_gain, get_img, fg_mask, gain_low, gain_high)
        else:
            gain = best_gain
    elif preset == 'fast':
        # Find the best gain at the real-time exposure and then find the best exposure at this gain
        set_exp(fast_exp)
        gain = pick_parameter('AutoGain', set_gain, get_img, fg_mask, gain_low, gain_high)
        set_gain(gain)
        exposure = pick_parameter('AutoExposure', set_exp, get_img, fg_mask, exp_low, exp_high)
    exposure = int(round(exposure))
    gain = int(round(gain))
    return exposure, gain


def pick_parameter(name, set_par, get_img, fg_mask, low, high, rstep=2, max_quality_steps_down=2, change_tol=1.01):
    low = pick_low_value(get_img, fg_mask, set_par, low)
    low = max(low, 1)
    if low >= high:
        return high

    max_power = np.ceil(np.log(high / low) / np.log(rstep))
    values = low * np.power(rstep, np.arange(0, max_power + 1))
    values = np.clip(values, a_min=low, a_max=high)

    def f(x):
        set_par(x)
        img = get_img()
        im_quality = get_img_quality(img, fg_mask)
        logger.debug(f'{name}: {int(x)}\t {im_quality:.3f}')
        return im_quality

    max_quality = 0
    best_value_i = 0
    steps_down = 0
    for i, value in enumerate(values):
        quality = f(value)
        if quality > max_quality:
            max_quality = quality
            best_value_i = i
        elif quality < max_quality / change_tol:
            steps_down += 1
            if steps_down == max_quality_steps_down:
                break
    low_i = max(best_value_i - 1, 0)
    high_i = min(best_value_i + 1, len(values) - 1)
    low = values[low_i]
    high = values[high_i]
    tol = values[best_value_i] / 100
    return pick_parameter_fine(name, set_par, get_img, fg_mask, low, high, tol)


def pick_parameter_fine(name, set_par, get_img, fg_mask, low, high, tol):
    def f(x):
        x = x
        set_par(x)
        img = get_img()
        im_quality = get_img_quality(img, fg_mask)
        logger.debug(f'{name}: {int(x)}\t {im_quality:.3f}')
        return -im_quality

    solution = scipy.optimize.minimize_scalar(f, bounds=(low, high), method='bounded', options={'xatol': tol})
    return solution.x


def get_img_quality(img, is_fg):
    r"""TODO

    Parameters
    ----------
    img : torch.Tensor
    of shape [h, w, 3] or [h, w].
    is_fg : torch.Tensor
    of shape [h, w].

    Returns
    -------
    img_quality : float
    """
    multichannel = img.ndim == 3
    luma = torch.from_numpy(get_luma(img.numpy())) if multichannel else img
    entropy = get_img_entropy(luma[is_fg]); del luma
    return entropy


def get_img_entropy(img, norm_factor=.125):
    r"""TODO

    Parameters
    ----------
    img : torch.Tensor
    norm_factor : float

    Returns
    -------
    entropy : float
    """
    probs = img.ravel().bincount() / img.numel()
    entropy = torch.distributions.Categorical(probs).entropy(); del probs
    entropy = entropy / np.log(2) * norm_factor
    return entropy.item()


def pick_low_value(get_img, fg_mask, set_par, par_low, stop_max_luma=50, rel_step=3):
    par_low = max(par_low, 1)
    par = par_low
    while True:
        set_par(par)
        img = get_img()
        multichannel = img.ndim == 3
        luma = get_luma(img.numpy()) if multichannel else img
        max_luma = luma[fg_mask].max()
        if max_luma >= stop_max_luma:
            return par
        par = par * rel_step
