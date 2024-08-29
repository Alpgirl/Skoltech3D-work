import torch

from skrgbd.calibration.camera_models.camera_model import CameraModel
from skrgbd.utils import ignore_warnings


class Redistorter:
    dst_pxs_in_src = None
    mode = 'bilinear'

    @classmethod
    def from_cam_models(cls, src_model, dst_pxs_in_src_uv):
        r"""Re-distorts image from one camera model to another.

        The algorithm is:
        1. Unproject each pixel of the target camera into the camera space.
        2. Project them into the image space of the source camera
        3. Interpolate source values at the projected positions.

        Parameters
        ----------
        src_model : CameraModel
        dst_pxs_in_src_uv : torch.Tensor
            of shape [dst_h, dst_w, 2]
        """
        src_wh = src_model.size_wh.to(dst_pxs_in_src_uv); del src_model

        invalid_pix_location = -2
        dst_pxs_in_src = dst_pxs_in_src_uv.mul(2 / src_wh).sub_(1); del dst_pxs_in_src_uv
        dst_pxs_in_src.nan_to_num_(invalid_pix_location, invalid_pix_location, invalid_pix_location)

        redistorter = cls()
        redistorter.dst_pxs_in_src = dst_pxs_in_src
        return redistorter

    @classmethod
    def from_file(cls, model2_pxs_in_model1_pt, device='cpu'):
        r"""
        Parameters
        ----------
        model2_pxs_in_model1_pt : str
            Path to the file saved with `save`.
        device : torch.device
        """
        redistorter = cls()
        redistorter.dst_pxs_in_src = torch.load(model2_pxs_in_model1_pt, map_location=device)
        return redistorter

    def save(self, path):
        r"""
        Parameters
        ----------
        path : str
        """
        model2_pxs_in_model1 = self.dst_pxs_in_src.cpu()
        torch.save(model2_pxs_in_model1, path)

    @ignore_warnings(['Default grid_sample and affine_grid behavior has changed to align_corners=False'])
    def redistort(self, images):
        r"""
        Parameters
        ----------
        images : torch.Tensor
            of shape [images_n, channels_n, src_height, src_width]

        Returns
        -------
        redist_images : torch.Tensor
            of shape [images_n, channels_n, dst_height, dst_width]
        """
        images_n = len(images)
        grid = self.dst_pxs_in_src.unsqueeze(0).expand(images_n, -1, -1, -1)
        undist_images = torch.nn.functional.grid_sample(images, grid, self.mode, 'zeros', align_corners=None)
        return undist_images

    @property
    def device(self):
        return self.dst_pxs_in_src.device

    @property
    def dtype(self):
        return self.dst_pxs_in_src.dtype

    def to(self, *args, **kwargs):
        self.dst_pxs_in_src = self.dst_pxs_in_src.to(*args, **kwargs)
        return self


def redist_grid_from_cam_models(src_model, dst_model, batch_size=2048 * 2048, show_progress=True):
    r"""Calculates the parameters of re-distortion from one camera to another,
    i.e coordinates of the destination pixels in the image space of the source camera.

    Parameters
    ----------
    src_model : CameraModel
    dst_model : CameraModel
    batch_size : int
    show_progress : bool

    Returns
    -------
    dst_pxs_in_src_uv : torch.Tensor
        of shape [2, dst_h, dst_w], UV coordinates of pixels of the destination camera in the source camera space.
    deviations : torch.Tensor
        of shape [dst_h, dst_w], deviations between unit ray-vectors casted through the centers of pixels
        from the destination camera, and casted through their found projections from the source camera.
    """
    dst_pix_rays = dst_model.get_pixel_rays()
    dst_h, dst_w = dst_pix_rays.shape[1:]
    dst_pix_rays = dst_pix_rays.view(3, -1)
    ray_is_calibrated = dst_pix_rays.isfinite().all(0)
    xyz = dst_pix_rays[:, ray_is_calibrated]; del dst_pix_rays
    ray_is_calibrated = ray_is_calibrated.cpu()

    def project(xyz, fine=True):
        if fine:
            uv = src_model.project_fine(xyz, show_progress=show_progress)
        else:
            uv = src_model.project(xyz)
        est_dir = src_model.unproject(uv)
        true_dir = torch.nn.functional.normalize(xyz, dim=0)
        dev = est_dir.sub_(true_dir); del est_dir, true_dir
        dev = dev.norm(dim=0)
        return uv, dev

    uv = []
    dev = []
    rays_n = xyz.shape[1]

    for batch_start in range(0, rays_n, batch_size):
        batch_end = min(xyz.shape[1], batch_start + batch_size)
        uv_batch, dev_batch = project(xyz[:, batch_start: batch_end], True)
        uv.append(uv_batch.cpu()); del uv_batch
        dev.append(dev_batch.cpu()); del dev_batch

    uv = torch.cat(uv, 1)
    dev = torch.cat(dev, 0)

    # If deviations ended up with nans, we assume that this is due to divergence in optimization during fine projection,
    # so we simply keep the unoptimized values.
    dev_is_finite = dev.isfinite()
    if not dev_is_finite.all():
        mask = dev_is_finite.logical_not_()
        uv_batch, dev_batch = project(xyz[:, mask.to(xyz.device)], False)
        uv[:, mask] = uv_batch.cpu(); del uv_batch
        dev[mask] = dev_batch.cpu(); del dev_batch, mask
    del dev_is_finite, xyz,

    uv_calibrated = uv
    dev_calibrated = dev

    uv = uv.new_full([2, dst_h, dst_w], float('nan'))
    uv.view(2, -1)[:, ray_is_calibrated] = uv_calibrated; del uv_calibrated

    dev = dev.new_full([dst_h, dst_w], float('nan'))
    dev.view(-1)[ray_is_calibrated] = dev_calibrated

    return uv, dev
