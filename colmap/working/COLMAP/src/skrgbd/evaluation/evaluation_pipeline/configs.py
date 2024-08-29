from skrgbd.utils import SimpleNamespace

r"""
Config parameters
-----------------
occ_threshold : float
    Points of the reference surface below the boundary of the occluded region deeper than this are occluded.
max_visible_depth : float
    For reconstructed points below the reference surface deeper than this the distance to reference surface is unknown.
max_dist : float
    For point-to-point calculation of distance from reference to reconstruction, distances larger than this are set to inf.
    For calculation of mean distances, distances are max-clipped by this value.
max_edge_len : float
    For reconstruction in form of triangle mesh, the maximal edge length for resampling of the reconstructed surface.
max_dist_to_sample : float
    For reconstruction in form of triangle mesh, only triangles closer than this are resampled.
vis_view_i : int
    in range(100). Point of view used to for visualization of distances.
vis_pov_wh : tuple of int
    (w, h). POV window used for visualization of distances.
vis_resolution_wh : tuple of int
    (w, h). Resolution used for visualization of distances. If None, vis_pov_wh is used.
z_near : float
    Points closer to camera than this are not rendered during visualization.
z_far : float
    Points farther from camera than this are not rendered during visualization.
occ_eps : float
    If the distance from a reconstructed point to the reference surface is greater than the distance
    to the boundary of the occluded space plus occ_eps, then the distance from the point to the true surface
    is considered unknown.
dist_range : tuple of float
    (min_dist, max_dist).
    Distance range to colorize: min_dist is mapped to min_color, max_dist is mapped to max_color.
color_range : tuple of float
    (min_color, max_color).
cmap : str
    Color map to visualize distances in.
splat_size : float
    Size of a splat in world units.
crop : bool
    If True, crop the visualizations to the bounding box of the SL visualization.
margin : int
    Size of the margins of cropped visualizations, in pixels.
draw_ref : bool
    If True, additionally render reference surface without colors.
two_sided : bool
    If True, shade reconstruction points as if they were always turned to the light.
thres_range : tuple of float
    (thres_min, thres_max, thres_n). Min, max value and the number of steps for calculation of thresholded metrics.
stat_cell_size : float
    Size of the cell for intermediate averaging.
vox_dists_thres : float
    FIXME add desc
"""

default_config = SimpleNamespace(
    occ_threshold=1e-3,
    max_visible_depth=3e-3,
    max_dist=1e-2,
    max_edge_len=1e-4,
    max_dist_to_sample=1e-1,
    vis_view_i=53,
    vis_pov_wh=[2560, 2048],
    vis_resolution_wh=None,
    z_near=.3,
    z_far=3.,
    occ_eps=1e-4,
    dist_range=[0, 1e-3],
    color_range=[.02, 1],
    cmap='hot_r',
    splat_size=5e-4,
    crop=True,
    margin=32,
    draw_ref=False,
    two_sided=False,
    thres_range=[1e-4, 3e-3, 101],
    stat_cell_size=1e-3,
    vox_dists_thres=5e-4,
    config_ver='v1',
)

config_ext_3cm = default_config.copy_with(
    max_dist=3e-2,
    dist_range=[0, 3e-2],
    thres_range=[3e-3, 3e-2, 101],
    config_ver='v2',
)

config_ext_2cm = default_config.copy_with(
    max_visible_depth=2e-2,
    max_dist=2e-2,
    dist_range=[0, 2e-2],
    cmap='cold_r',
    config_ver='v3',
)

configs = {
    'colmap': {('tis_right', 'ambient@best'): {
        'v2.0.0': default_config.copy_with(draw_ref=True),
        'v1.0.0': default_config,
        'v0.0.0': default_config,
        'v0.0.0_unrefined': default_config,
    }},
    'acmp': {('tis_right', 'ambient@best'): {
        'v1.0.0': default_config,
    }},
    'vismvsnet': {('tis_right', 'ambient@best'): {
        'v1.0.0_trained_on_blendedmvg': default_config,
    }},
    'unimvsnet': {('tis_right', 'ambient@best'): {
        'v1.0.0_authors_checkpoint': default_config.copy_with(two_sided=True),
    }},
    'neus': {('tis_right', 'ambient@best'): {
        'v1.1.0': default_config,
        'v1.0.0': default_config,
    }},
    'geo_neus': {('tis_right', 'ambient@best'): {
        'v0.0.0_colmap': default_config,
    }},
    'fneus': {('tis_right', 'ambient@best'): {
        'v0.0.0': default_config,
        'v0.1.0': default_config,
        'v0.2.0': default_config,
        'v0.3.0_1vol1surf': default_config,
        'v0.3.0_1vol2vol': default_config,
    }},
    'tsdf_fusion': {('kinect_v2', None): {
        'v0.0.0': config_ext_2cm,
    }},
    'surfel_meshing': {('kinect_v2', None): {
        'v1.0.0': config_ext_2cm,
    }},
    'routed_fusion': {('kinect_v2', None): {
        'v0.0.0': config_ext_2cm.copy_with(two_sided=True),
    }},
    'azinovic22neural': {('kinect_v2@tis_right', 'ambient@best'): {
        'v0.0.0': config_ext_2cm
    }},
    'spsr_colmap': {('tis_right', 'ambient@best'): {
        'v1.0.0': default_config,
    }},
    'spsr_acmp': {('tis_right', 'ambient@best'): {
        'v1.0.0': default_config,
    }},
    'spsr_vismvsnet': {('tis_right', 'ambient@best'): {
        'v1.0.0_trained_on_blendedmvg': default_config,
    }},
    'spsr_unimvsnet': {('tis_right', 'ambient@best'): {
        'v1.0.0_authors_checkpoint': default_config.copy_with(two_sided=True),
    }},
    'indi_sg': {('tis_right', 'ambient@best'): {
        'v0.0.0': default_config,
    }},
    'physg': {('tis_right', 'ambient@best'): {
        'v0.0.0': default_config,
    }},
}
