from skrgbd.utils import SimpleNamespace as SNS


colmap_default_config = SNS(
    extract_features=SNS(
        max_num_features_n=8192,
        estimate_affine_shape=False,
        use_domain_size_pooling=False,
    ),
    match_features=SNS(
        max_matches_n=32768,
        max_error_px=4,
        min_confidence=.999,
        estimate_multiple_models=False,
        do_guided_matching=False,
        block_size=50,
    ),
    undistort_images=SNS(
        patch_match_src_n=20,
    ),
    run_patch_match=SNS(
        window_radius_px=5,
        window_step_px=1,
        sigma_spatial_px=-1,
        sigma_color=.2,
        mc_samples_n=15,
        ncc_sigma=.6,
        max_image_size=2000,
        min_tri_angle_deg=1,
        incident_angle_sigma=.9,
        patchmatch_iters_n=5,
        use_geom_consistency=True,
        geom_consistency_w=.3,
        max_geom_inconsistency_px=3,
        do_filtering=True,
        filt_min_consistent_src_n=2,
        filt_min_ncc=.1,
        filt_min_tri_angle_deg=3,
        filt_max_geom_inconsistency_px=1,
        cache_size_gb=32,
    ),
    fuse_points=SNS(
        input_type='geometric',
        min_pix_n=5,
        max_pix_n=10_000,
        max_traversal_depth=100,
        max_reproj_err_px=2,
        max_depth_err=.01,
        max_normal_err_deg=10,
        max_images_to_fuse_n=50,
    )
)

default_config = colmap_default_config.copy_with(
    extract_features=SNS(
        estimate_affine_shape=True,    # Recommended for best performance. Stops feature extractor from using GPU.
        use_domain_size_pooling=True,  # Recommended for best performance. Stops feature extractor from using GPU.
    ),
    match_features=SNS(
        do_guided_matching=True,       # Recommended for best performance.
    ),
    run_patch_match=SNS(
        max_image_size=-1,             # use the full image resolution
    ),
    fuse_points=SNS(
        max_images_to_fuse_n=100,      # fuse all images for each pixel, if they are consistent
        max_traversal_depth=10_000,    # fuse all source pixels for each reference pixel, if they are consistent
        max_reproj_err_px=.5,          # for higher accuracy, and also, higher completeness
        max_normal_err_deg=3,          # for higher accuracy
        max_depth_err=.001,            # for higher accuracy
    ),
)

configs = dict()
configs['v3.0.0'] = {
    ('tis_right', 'ambient@best'): default_config,  # Same for v2.0.0
    ('phone_left', 'ambient@best'): default_config.copy_with(run_patch_match=SNS(max_image_size=2560)),
    ('real_sense', 'ambient@best'): default_config,
    ('kinect_v2', 'ambient'): default_config,
}
