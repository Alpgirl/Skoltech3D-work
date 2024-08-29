import torch


def segment_fg(src, bg, device='cuda:1'):
    r"""
    Parameters
    ----------
    src : torch.Tensor
        of shape [3, h, w] in [0, 1] range.
    bg : torch.Tensor
        of shape [3, h, w] in [0, 1] range.

    Returns
    -------
    is_fg : torch.BoolTensor
        of shape [h, w].
    """
    with torch.no_grad():
        src = src.unsqueeze(0).to(device)
        bg = bg.unsqueeze(0).to(device)

        model = torch.jit.load(
            '/home/universal/Downloads/dev.sk_robot_rgbd_data/src/skrgbd/scanning/data/bg_matting.torchscript_resnet50_fp32.pth')
        model.backbone_scale = 1
        model.refine_mode = 'thresholding'
        model = model.to(device)

        is_fg = model(src, bg)[0]
        is_fg = is_fg.detach().greater(.5).cpu().squeeze(1).squeeze(0)
    return is_fg
