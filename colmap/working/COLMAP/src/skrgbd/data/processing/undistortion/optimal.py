import torch

from skrgbd.calibration.camera_models.camera_model import CameraModel
from skrgbd.calibration.camera_models.pinhole import PinholeCameraModel


def get_optimal_pinhole(cam_models, figsize=None):
    r"""Finds an optimal pinhole camera model for a set of arbitrary camera models.

    For a single camera model, the algorithm is:
    1. Find the inscribed rectangle of the calibrated region.
    2. Set the height / width of the pinhole camera model to the size of the longest column / row of the original pixels
       inside the inscribed rectangle.
    3. Calculate the parameters of the pinhole model so that the corners of the inscribed rectangle are mapped
       to the centers of the corner pixels.

    For multiple camera models, the inscribed rectangle is calculated for the intersection of the calibrated regions.

    Parameters
    ----------
    cam_models : CameraModel or iterable of CameraModel
    figsize : tuple of float
        (w, h) of the figure. If not None, visualize the original and undistorted pixel grids in normalized coordinates.

    Returns
    -------
    pin_cam_model : PinholeCameraModel
    """
    if isinstance(cam_models, CameraModel):
        cam_models = [cam_models]

    # 1. Find the boundary of the calibrated region in normalized uv-coordinates
    uvn_min_all = []
    uvn_max_all = []
    uvns = []

    for cam_model in cam_models:
        pix_rays = cam_model.get_pixel_rays(); del cam_model
        uvn = pix_rays[:2] / pix_rays[2]; del pix_rays

        is_calibrated = uvn[0].isfinite()
        h, w = is_calibrated.shape

        # We assume that the calibrated region is rectangular,
        # so if any pixel ray in a column / row is valid, then all pixel rays in that column / row are valid.
        row_is_calibrated = is_calibrated.any(1).byte()
        i_min = row_is_calibrated.argmax()
        i_max = h - row_is_calibrated.flip(0).argmax() - 1; del row_is_calibrated

        col_is_calibrated = is_calibrated.any(0).byte(); del is_calibrated
        j_min = col_is_calibrated.argmax()
        j_max = w - col_is_calibrated.flip(0).argmax() - 1; del col_is_calibrated

        h = i_max - i_min; del i_max
        w = j_max - j_min; del j_max
        uvn = uvn[:, i_min: i_min + h, j_min: j_min + w].contiguous(); del i_min, j_min
        assert uvn.isfinite().all()

        # 2. Calculate the inscribed rectangle of the calibrated region
        first_col, last_col = uvn[..., 0], uvn[..., -1]
        first_row, last_row = uvn[:, 0], uvn[:, -1]
        uvn_min = torch.stack([first_col[0].max(), first_row[1].max()]); del first_col, first_row
        uvn_max = torch.stack([last_col[0].min(), last_row[1].min()]); del last_col, last_row

        uvn_min_all.append(uvn_min)
        uvn_max_all.append(uvn_max)
        uvns.append(uvn)
    del cam_models

    uvn_min_all = torch.stack(uvn_min_all)
    uvn_max_all = torch.stack(uvn_max_all)
    uvn_min = uvn_min_all.max(0)[0]; del uvn_min_all
    uvn_max = uvn_max_all.min(0)[0]; del uvn_max_all

    # 3. Choose the resolution of the pinhole camera equal to the dimensions of the inscribed rectangle
    # in the image space of the original camera model. Each original column / row of N pixels is mapped to
    # at least N pixels in the pinhole camera image space, so we only upsample and never downsample,
    # and the information from the original image is preserved as much as possible.
    # The calculation below is approximate.
    h, w = 0, 0
    for uvn in uvns:
        src_ray_is_in_rec = (uvn >= uvn_min.unsqueeze(1).unsqueeze(2)).all(0)
        src_ray_is_in_rec = src_ray_is_in_rec.logical_and_((uvn <= uvn_max.unsqueeze(1).unsqueeze(2)).all(0))
        h = max(h, src_ray_is_in_rec.sum(0).max())
        w = max(w, src_ray_is_in_rec.sum(1).max()); del src_ray_is_in_rec
    wh = torch.tensor([w, h])

    # 4. Calculate focal lengths and principal point
    # so that the corners of the inscribed rectangle are mapped to the centers of the corner pixels.
    # Note, that the edges of the inscribed rectangle correspond to the centers of some original pixels.
    fxy = (wh - 1) / (uvn_max - uvn_min)
    cxy = .5 - uvn_min * fxy
    pin_cam_model = PinholeCameraModel(focal=fxy, principal=cxy, size_wh=wh)

    # 5. Visualize the results
    if figsize:
        import matplotlib.pyplot as plt
        import matplotlib.colors

        plt.figure(figsize=figsize)

        colors = list(matplotlib.colors.TABLEAU_COLORS.values())
        for uvn, color in zip(uvns, colors):
            h, w = uvn.shape[1:]
            grid_lines_i = torch.linspace(0, h - 1, 30).round().long()
            grid_lines_j = torch.linspace(0, w - 1, 30).round().long()
            for i in grid_lines_i:
                plt.plot(uvn[0, i], uvn[1, i], c=color)
            for j in grid_lines_j:
                plt.plot(uvn[0, :, j], uvn[1, :, j], c=color)

        v, u = torch.meshgrid([torch.linspace(uvn_min[1], uvn_max[1], h), torch.linspace(uvn_min[0], uvn_max[0], w)])
        uvn_pin = torch.stack([u, v], 0); del u, v
        grid_lines_i = torch.linspace(0, h - 1, 30).round().long()
        grid_lines_j = torch.linspace(0, w - 1, 30).round().long()
        for i in grid_lines_i:
            plt.plot(uvn_pin[0, i], uvn_pin[1, i], c='black')
        for j in grid_lines_j:
            plt.plot(uvn_pin[0, :, j], uvn_pin[1, :, j], c='black')

    return pin_cam_model
