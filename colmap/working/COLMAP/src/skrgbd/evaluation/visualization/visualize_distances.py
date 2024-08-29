import numpy as np
import open3d as o3d
import torch

from skrgbd.evaluation.statistics import centered_stat
from skrgbd.evaluation.visualization.colorize_distances import colorize_dists
from skrgbd.data.image_utils import get_trim
from skrgbd.utils.logging import logger
from skrgbd.evaluation.visualization.splat_renderer import make_square_splat, SplatRenderer
from skrgbd.data.processing.depth_utils.mesh_rendering_gl import MeshRenderer
from skrgbd.calibration.camera_models.pinhole import PinholeCameraModel


def visualize_distances(
        ref, dist_from_ref, rec, rec_normals, visible_rec_pt_ids, dist_to_ref, dist_to_occ,
        cam_model, z_near=.3, z_far=3,
        occ_eps=1e-4, max_dist=1e-2, dist_range=(0, 1e-3), color_range=(.02, 1),
        no_dist_color=(.75, .75, .75), cmap='hot_r', bg_color=(1, 1, 1),
        light_dir=(0, 0, 1), two_sided=False, min_shading=.5,
        splat_size=5e-4, antialias=True, crop=True, margin=32,
        draw_ref=False, draw_rec=True, shading_range=(.05, .95),
        patch_size=64, device='cuda'
):
    r"""Draws distributions of distances from reference surface to reconstruction and visa-versa as images.

    Parameters
    ----------
    ref : o3d.geometry.TriangleMesh
        SL reconstruction, transformed to the camera space.
    dist_from_ref : torch.FloatTensor
        of shape [ref_verts_n]. Distance from the reference surface to the reconstruction points.
    rec : torch.FloatTensor
        of shape [pts_n, 3]. Points of the reconstruction produced by a method, transformed to the camera space.
    rec_normals : torch.FloatTensor
        of shape [pts_n, 3]. Normals of the points of the reconstruction, transformed to the camera space.
    visible_rec_pt_ids : torch.LongTensor
        of shape [vis_pts_n]. Ids of reconstruction points in visible space.
    dist_to_ref : torch.FloatTensor
        of shape [vis_pts_n]. Distance to the reference surface from the reconstruction points in visible space.
    dist_to_occ : torch.FloatTensor
        of shape [vis_pts_n].
        Distance to the occluded space boundary from the reconstruction points in visible space.
    cam_model : PinholeCameraModel
        Pinhole camera model used to draw the visualizations.
    z_near : float
        Points closer to camera than this are not rendered.
    z_far : float
        Points farther from camera than this are not rendered.
    occ_eps : float
        If the distance from a reconstructed point to the reference surface is greater than the distance
        to the boundary of the occluded space plus occ_eps, then the distance from the point to the true surface
        is considered unknown.
    max_dist : float
        Used for visualization of accuracy on surface.
        Reconstructed points farther than this from the reference surface do not affect the visualization.
    dist_range : tuple of float
        (min_dist, max_dist).
        Distance range to colorize: min_dist is mapped to min_color, max_dist is mapped to max_color.
    color_range : tuple of float
        (min_color, max_color).
    no_dist_color : tuple of float
        (r, g, b), in range [0, 1]. Points with invalid distance are drawn in this color.
    cmap : str
        Colormap to draw distances in.
    bg_color : tuple of float
        (r, g, b), in range [0, 1]. Background color.
    light_dir : tuple of float
        (x, y, z). Normalized direction of light.
    two_sided : bool
        If True, shade reconstruction points as if they were always turned to the light.
    min_shading : float
        Minimal value of shading.
    splat_size : float
        Size of a splat in world units.
    antialias : bool
        If True, antialias the visualizations.
    crop : bool
        If True, crop the visualizations to the bounding box of the SL visualization.
    margin : int
        Size of the margins of cropped visualizations, in pixels.
    draw_ref : bool
        If True, additionally render reference surface without colors.
    draw_rec : bool
        If True, additionally render reconstructed points without colors.
    shading_range : tuple of float
        (s_min, s_max). Non-colored visualizations are shaded in this range of grayscale values.
    patch_size : int
        Splat-based accuracy is rendered in patches of this size, to fit in GPU memory.
    device : torch.device
        GPU device used for rendering.

    Returns
    -------
    results : dict
        reference : torch.FloatTensor
            of shape [height, width], in range [0, 1].
            Visualization of the reference surface.
        reconstruction : torch.FloatTensor
            of shape [height, width], in range [0, 1].
            Visualization of the reconstructed points.
        completeness : torch.FloatTensor
            of shape [height, width, 3], in range [0, 1].
            Visualization of distances from reference surface to reconstruction, on the reference surface.
        accuracy : torch.FloatTensor
            of shape [height, width, 3], in range [0, 1].
            Visualization of distances from reconstruction to reference surface, on the reconstructed points.
        surf_accuracy : torch.FloatTensor
            of shape [height, width, 3], in range [0, 1].
            Visualization of distances from reconstruction to reference surface,
            projected to and averaged on the reference surface.
        crop_left_top : torch.LongTensor
            (left, top).
    """
    dtype = torch.float

    bg_color = torch.tensor(bg_color, device=device, dtype=dtype)
    bg_color4 = torch.cat([bg_color, bg_color.mean(0, keepdim=True)], 0)
    world_to_cam = torch.eye(4, device=device, dtype=dtype)
    light_dir = torch.tensor(light_dir, dtype=dtype)

    'Visualize completeness' >> logger.debug
    'Colorize dists' >> logger.debug
    dists = dist_from_ref.where(dist_from_ref.isfinite(), dist_from_ref.new_tensor(dist_range[1]))
    colors = colorize_dists(dists, dist_range, color_range, no_dist_color, cmap); del dists

    'Init renderer for completeness' >> logger.debug
    ref = o3d.geometry.TriangleMesh(ref)
    ref.compute_vertex_normals()
    renderer = MeshRenderer(ref, device)
    renderer.init_mesh_data()
    renderer.set_cam_model(cam_model, near=z_near, far=z_far)
    renderer.set_resolution(cam_model.size_wh[1], cam_model.size_wh[0])

    'Render completeness' >> logger.debug
    ref_normals = torch.from_numpy(np.asarray(ref.vertex_normals)).to(dtype)
    ref_shading = get_shading(ref_normals, light_dir); del ref_normals
    colors = shade_colors_(colors, ref_shading, min_shading)
    if not draw_ref:
        completeness = renderer.render_colors_to_camera(colors.to(device, dtype), world_to_cam, bg_color, antialias).cpu()
        ref_render = None
    else:
        shading = ref_shading.mul(shading_range[1] - shading_range[0]).add(shading_range[0])
        cols_and_shading = torch.cat([colors, shading.unsqueeze(1)], 1); del shading
        render = renderer.render_colors_to_camera(cols_and_shading.to(device, dtype), world_to_cam, bg_color4, antialias)
        del cols_and_shading
        completeness = render[..., :3].cpu()
        ref_render = render[..., 3].cpu(); del render

    if crop:
        'Crop' >> logger.debug
        comp_is_nonempty = completeness.ne(bg_color.to(completeness)).any(-1)
        i_min, i_max, j_min, j_max = get_trim(comp_is_nonempty); del comp_is_nonempty
        i_min = max(0, i_min - margin)
        j_min = max(0, j_min - margin)
        w, h = cam_model.size_wh
        i_max = min(h, i_max + margin)
        j_max = min(w, j_max + margin)

        completeness = completeness[i_min: i_max, j_min: j_max]
        crop_left_top = cam_model.size_wh.new_tensor([j_min, i_min])
        new_wh = cam_model.size_wh.new_tensor([j_max - j_min, i_max - i_min])
        cam_model = cam_model.crop_(crop_left_top, new_wh)
        renderer.set_cam_model(cam_model, near=z_near, far=z_far)
        renderer.set_resolution(cam_model.size_wh[1], cam_model.size_wh[0])

        if ref_render is not None:
            ref_render = ref_render[i_min: i_max, j_min: j_max]
    else:
        crop_left_top = cam_model.size_wh.new_tensor([0, 0])

    'Visualize accuracy on surface' >> logger.debug
    dists = dist_to_ref.where(dist_to_ref <= dist_to_occ.add(occ_eps), dist_to_ref.new_tensor(float('nan')))
    del dist_from_ref, dist_to_occ
    dists = torch.full([len(rec)], float('nan'), dtype=dists.dtype).scatter_(0, visible_rec_pt_ids, dists)
    del visible_rec_pt_ids

    ref_verts = np.asarray(ref.vertices); del ref
    rec_pt_is_visible = dists.isfinite()
    vert_ids, avg_dists = centered_stat(
        rec[rec_pt_is_visible].T, dists[rec_pt_is_visible].unsqueeze(0), ref_verts, 'mean', max_dist)
    del rec_pt_is_visible, ref_verts

    avg_colors = colorize_dists(avg_dists.squeeze(0), dist_range, color_range, no_dist_color, cmap); del avg_dists
    colors[:] = colors.new_tensor(no_dist_color)
    colors[vert_ids] = avg_colors; del vert_ids, avg_colors

    'Render surf_accuracy' >> logger.debug
    colors = shade_colors_(colors, ref_shading, min_shading); del ref_shading
    surf_accuracy = renderer.render_colors_to_camera(colors.to(device, dtype), world_to_cam, bg_color, antialias).cpu()
    del world_to_cam, renderer, colors

    'Visualize accuracy on points' >> logger.debug
    colors = colorize_dists(dists, dist_range, color_range, no_dist_color, cmap); del dists

    'Eliminate rec points outside of canvas' >> logger.debug
    is_in_bounds = cam_model.uv_is_in_bounds(cam_model.project(rec.T))
    rec = rec[is_in_bounds]
    rec_normals = rec_normals[is_in_bounds]
    colors = colors[is_in_bounds]; del is_in_bounds

    'Init renderer' >> logger.debug
    renderer = SplatRenderer(rec, device); del rec
    renderer.set_cam_model(cam_model, near=z_near, far=z_far)
    template_verts, template_tris = make_square_splat(splat_size)
    renderer.set_template(template_verts, template_tris); del template_tris, template_verts

    'Render accuracy' >> logger.debug
    rec_shading = get_shading(rec_normals, light_dir, two_sided); del rec_normals, light_dir
    colors = shade_colors_(colors, rec_shading, min_shading)
    if not draw_rec:
        accuracy = renderer.render_colors_to_camera_patches(colors, bg_color, antialias, patch_size); del renderer
        rec_render = None
    else:
        shading = rec_shading.mul(shading_range[1] - shading_range[0]).add(shading_range[0])
        cols_and_shading = torch.cat([colors, shading.unsqueeze(1)], 1); del shading
        render = renderer.render_colors_to_camera_patches(cols_and_shading, bg_color4, antialias, patch_size)
        del renderer, cols_and_shading
        accuracy = render[..., :3]
        rec_render = render[..., 3].cpu(); del render
    del rec_shading

    'Clip to [0, 1]' >> logger.debug
    completeness, accuracy, surf_accuracy = (img.clamp_(0, 1) for img in [completeness, accuracy, surf_accuracy])
    if ref_render is not None:
        ref_render = ref_render.clamp_(0, 1)
    if rec_render is not None:
        rec_render = rec_render.clamp_(0, 1)

    results = dict(
        completeness=completeness,
        accuracy=accuracy,
        surf_accuracy=surf_accuracy,
        crop_left_top=crop_left_top
    )
    if ref_render is not None:
        results['reference'] = ref_render
    if rec_render is not None:
        results['reconstruction'] = rec_render
    return results


def get_shading(normals, light_dir, two_sided=False):
    r"""Calculates diffuse shading of points with normals.

    Parameters
    ----------
    normals : torch.Tensor
        of shape [pts_n, 3].
    light_dir : torch.Tensor
        (x, y, z). Normalized direction of light.
    two_sided : bool
        If True, shade points as if they were always turned to the light.

    Returns
    -------
    shading : torch.Tensor
        of shape [pts_n].
    """
    shading = (normals @ light_dir.neg().unsqueeze(1)).squeeze(1); del normals, light_dir
    if two_sided:
        shading = shading.abs_()
    shading = shading.clamp_(0, 1)
    return shading


def shade_colors_(colors, shading, min_shading=.5):
    r"""Shades colors.

    Parameters
    ----------
    colors : torch.Tensor
        of shape [pts_n, 3].
    shading : torch.Tensor
        of shape [pts_n].
    min_shading : float
        Minimal value of shading.

    Returns
    -------
    shaded_colors : torch.Tensor
        of shape [pts_n, 3].
    """
    shading = shading.mul(1 - min_shading).add_(min_shading).clamp_(min_shading, 1)
    shaded_colors = colors.mul_(shading.unsqueeze(1))
    return shaded_colors


def setup_pov_cam(cam_model, pts, pov_wh=(2560, 2048), resolution_wh=None):
    r"""Sets up camera model for visualization.

    Parameters
    ----------
    cam_model : PinholeCameraModel
    pts : torch.Tensor
        of shape [3, pts_n]. Points of the reference surface to focus on.
    pov_wh : tuple of int
        (w, h).
    resolution_wh : tuple of int
        (w, h).

    Returns
    -------
    pov_cam_model : PinholeCameraModel
    crop_left_top : torch.LongTensor
        (left, top).
    """
    pov_uv = cam_model.project(pts).mean(1); del pts

    old_wh = cam_model.size_wh
    pov_wh = old_wh.new_tensor(pov_wh)
    crop_left_top = (pov_uv - pov_wh / 2).round().int()
    cam_model = cam_model.crop_(crop_left_top, pov_wh)

    if resolution_wh is not None:
        resolution_wh = pov_wh.new_tensor(resolution_wh)
        cam_model = cam_model.resize_(resolution_wh)
    return cam_model, crop_left_top
