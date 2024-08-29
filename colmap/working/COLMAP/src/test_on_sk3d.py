from argparse import ArgumentParser
from pathlib import Path
import shutil
import subprocess

import os
import sys
# sys.path.append('skrgbd/utils/logging')
# sys.path.append('/sk3d_colmap')
# sys.path.append('/colmap')

from skrgbd.utils.logging import logger
from skrgbd.data.dataset.scene_paths import ScenePaths

from sk3d_colmap.configs import configs
from sk3d_colmap.eval_paths import EvalPaths

from colmap.read_write_model import CAMERA_MODEL_NAMES, read_cameras_text, read_images_text
from colmap.database import COLMAPDatabase



def main():
    description = 'Runs one stage of COLMAP testing.'
    parser = ArgumentParser(description=description)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--results-dir', type=str, required=True)
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--cam', type=str, required=True)
    parser.add_argument('--light', type=str, required=True)
    parser.add_argument('--scene-name', type=str, required=True)
    parser.add_argument('--colmap-bin', type=str, required=True)
    parser.add_argument('--stage', type=str, required=True)
    parser.add_argument('--threads-n', type=int, default=-1)
    args = parser.parse_args()
    print(os.getcwd())

    f'COLMAP {args.scene_name}' >> logger.debug
    scene_paths = ScenePaths(args.scene_name, data_dir=args.data_dir)
    eval_paths = EvalPaths(args.scene_name, rec_root=args.results_dir)
    test_colmap(args.stage, scene_paths, eval_paths, args.colmap_bin, args.version, args.cam, args.light, args.threads_n)


def test_colmap(stage, scene_paths, eval_paths, colmap_bin, version, cam, light, threads_n=-1):
    r"""Runs one stage of COLMAP testing.

    Parameters
    ----------
    stage : {'init', 'extract_features', 'match_features', 'triangulate', 'init_dense', 'run_patch_match', 'fuse_points'}
        Stage of testing.
    scene_paths : ScenePaths
    eval_paths : EvalPaths
    colmap_bin : str
        Path to the COLMAP executable.
    version : str
    cam : {'tis_right'}
    light : str
    threads_n : int
        Number of threads to use.
    """
    'Load config' >> logger.debug
    config = configs[version][cam, light]

    img_root = str(Path(scene_paths.img(cam, 'rgb', 0, light)).parent)
    precomp_sparse_dir = Path(eval_paths.rec_precomp_sparse_dir(version, cam, light))
    sparse_dir = eval_paths.rec_sparse_dir(version, cam, light)
    database_db = f'{sparse_dir}/database.db'
    print(database_db)
    dense_dir = eval_paths.rec_dense_dir(version, cam, light)

    if stage == 'init':
        subsystem = Path(eval_paths.rec_subsystem_dir(version, cam, light))
        if subsystem.exists():
            f'Clean subsystem {subsystem}' >> logger.debug
            shutil.rmtree(subsystem)

        precomp_sparse_dir.mkdir(parents=True)

        cameras_txt = scene_paths.cam_model(cam, 'rgb')
        precomp_cameras_txt = precomp_sparse_dir / 'cameras.txt'
        precomp_cameras_txt.symlink_to(cameras_txt)

        images_txt = scene_paths.cam_poses(cam, 'rgb')
        precomp_images_txt = precomp_sparse_dir / 'images.txt'
        precomp_images_txt.symlink_to(images_txt)

        precomp_points3d_txt = f'{precomp_sparse_dir}/points3D.txt'
        open(precomp_points3d_txt, 'w').close()

        Path(sparse_dir).mkdir(parents=True)
        init_db(database_db, cameras_txt, images_txt)

    elif stage == 'extract_features':
        extract_features(colmap_bin, database_db, img_root, threads_n=threads_n, **config.extract_features)

    elif stage == 'match_features':
        match_features(colmap_bin, database_db, threads_n=threads_n, **config.match_features)

    elif stage == 'triangulate':
        triangulate_points(colmap_bin, database_db, img_root, precomp_sparse_dir, sparse_dir, threads_n=threads_n)

    elif stage == 'init_dense':
        Path(dense_dir).mkdir(parents=True)
        undistort_images(colmap_bin, img_root, sparse_dir, dense_dir, **config.undistort_images)

    elif stage == 'run_patch_match':
        run_patch_match(colmap_bin, dense_dir, **config.run_patch_match)

    elif stage == 'fuse_points':
        fuse_points(colmap_bin, dense_dir, threads_n=threads_n, **config.fuse_points)

    else:
        raise ValueError(f'Unknown stage {stage}')


def init_db(database_db, cameras_txt, images_txt):
    r"""Initializes COLMAP database.db with known camera model and poses.

    Parameters
    ----------
    database_db : str
        Path to the new database.db.
    cameras_txt : str
        Path to cameras.txt from dataset calibration.
    images_txt : str
        Path to images.txt from dataset calibration.
    """
    f'Create db at {database_db}' >> logger.debug
    db = COLMAPDatabase.connect(database_db)
    with db:
        db.create_tables()

        f'Read cameras from {cameras_txt}' >> logger.debug
        colmap_cams = read_cameras_text(cameras_txt)
        assert len(colmap_cams.keys()) == 1
        camera_id = 0
        cam = colmap_cams[camera_id]
        model_id = CAMERA_MODEL_NAMES[cam.model].model_id

        f'Add camera {cam}' >> logger.debug
        # prior_focal_length=1 tells COLMAP that the camera parameters are precise
        db.add_camera(model_id, cam.width, cam.height, cam.params, prior_focal_length=1, camera_id=camera_id)

        f'Read images from {images_txt}' >> logger.debug
        images = read_images_text(images_txt)

        for image_id, image in images.items():
            # f'Add image {image_id:04}: {image}' >> logger.debug
            db.add_image(image.name, camera_id, image.qvec, image.tvec, image.id)

    db.close()


def extract_features(
        colmap_bin, database_db, images_root, max_num_features_n=8192, use_gpu=True, threads_n=-1,
        estimate_affine_shape=False, use_domain_size_pooling=False,
):
    r"""Runs COLMAP feature extraction.

    Description of the parameters: https://github.com/colmap/colmap/blob/3.8/src/feature/sift.h#L44

    Parameters
    ----------
    colmap_bin : str
        Path to the COLMAP executable.
    database_db : str
        Path to the COLMAP database.db.
    images_root : str
        Path to the directory with images.
    max_num_features_n : int
        Maximum number of features to detect, keeping larger-scale features.
    use_gpu : bool
        Whether to use the GPU for feature extraction.
    threads_n : int
        Number of threads to use.
    estimate_affine_shape : bool
        Estimate affine shape of SIFT features in the form of oriented ellipses
        as opposed to original SIFT which estimates oriented disks.
        Stops feature extractor from using GPU.
    use_domain_size_pooling : bool
        Domain-size pooling computes an average SIFT descriptor across multiple scales around the detected scale.
        Stops feature extractor from using GPU.
    """
    command = (
        f'{colmap_bin} feature_extractor'
        f' --database_path {database_db}'
        f' --image_path {images_root}'
        f' --ImageReader.existing_camera_id 0'  # in Sk3D calibration the camera is always id 0
        f' --SiftExtraction.max_image_size 102400'  # prevent image downsampling
        f' --SiftExtraction.max_num_features {max_num_features_n}'
        f' --SiftExtraction.use_gpu {int(use_gpu)}'
        f' --SiftExtraction.num_threads {threads_n}'
        f' --SiftExtraction.estimate_affine_shape {int(estimate_affine_shape)}'
        f' --SiftExtraction.domain_size_pooling {int(use_domain_size_pooling)}'
    )
    f'Run colmap feature extractor: {command}' >> logger.debug
    subprocess.run(command.split(), check=True)
    'Finished feature extraction' >> logger.debug


def match_features(
        colmap_bin, database_db, threads_n=-1, use_gpu=True, max_matches_n=32768, max_error_px=4, min_confidence=.999,
        estimate_multiple_models=False, do_guided_matching=False, block_size=50,
):
    r"""Runs COLMAP exhaustive feature matching.

    Description of the parameters:
        https://github.com/colmap/colmap/blob/3.8/src/mvs/patch_match.h#L59
        https://github.com/colmap/colmap/blob/3.8/src/feature/matching.h#L49


    Parameters
    ----------
    colmap_bin : str
        Path to the COLMAP executable.
    database_db : str
        Path to the COLMAP database.db.
    threads_n : int
        Number of threads to use.
    use_gpu : bool
        Whether to use the GPU for feature matching.
    max_matches_n : int
        Maximum number of matches. Decrease in case of OOM.
    max_error_px : float
        Maximum epipolar error in pixels for geometric verification.
    min_confidence : float
        Confidence threshold for geometric verification.
    estimate_multiple_models : bool
        Whether to attempt to estimate multiple geometric models per image pair.
    do_guided_matching : bool
        Whether to perform guided matching, if geometric verification succeeds.
    block_size : int
        Number of images to simultaneously load into memory. Use lower values for faster "progress".
    """
    command = (
        f'{colmap_bin} exhaustive_matcher'
        f' --database_path {database_db}'
        f' --SiftMatching.num_threads {threads_n}'
        f' --SiftMatching.use_gpu {int(use_gpu)}'
        f' --SiftMatching.max_num_matches {max_matches_n}'
        f' --TwoViewGeometry.max_error {max_error_px}' 
        f' --TwoViewGeometry.confidence {min_confidence}'
        f' --TwoViewGeometry.multiple_models {int(estimate_multiple_models)}'
        f' --SiftMatching.guided_matching {int(do_guided_matching)}'
        f' --ExhaustiveMatching.block_size {block_size}'
    )
    f'Run colmap feature matching: {command}' >> logger.debug
    subprocess.run(command.split(), check=True)
    'Finished feature matching' >> logger.debug


def triangulate_points(colmap_bin, database_db, images_root, precomp_sparse_root, sparse_root, threads_n=-1):
    r"""Runs COLMAP point triangulation.

    Description of the parameters:
        https://github.com/colmap/colmap/blob/3.8/src/controllers/incremental_mapper.h#L41
        https://github.com/colmap/colmap/blob/3.8/src/sfm/incremental_mapper.h#L66
        https://github.com/colmap/colmap/blob/3.8/src/sfm/incremental_triangulator.h#L45

    Parameters
    ----------
    colmap_bin : str
        Path to the COLMAP executable.
    database_db : str
        Path to the COLMAP database.db.
    images_root : str
        Path to the directory with images.
    precomp_sparse_root : str
        Path to the directory with cameras.txt, images.txt, points3D.txt with precomputed camera calibration.
    sparse_root : str
        Path to the output sparse reconstruction results.
    threads_n : int
        Number of threads to use.
    """
    print("Precomp Path", precomp_sparse_root)
    command = (
        f'{colmap_bin} point_triangulator'
        f' --database_path {database_db}'
        f' --image_path {images_root}'
        f' --input_path {precomp_sparse_root}'
        f' --output_path {sparse_root}'
        f' --Mapper.num_threads {threads_n}'
        f' --Mapper.ba_refine_focal_length {int(False)}'     # Fix camera parameters
        f' --Mapper.ba_refine_principal_point {int(False)}'  #
        f' --Mapper.ba_refine_extra_params {int(False)}'     #
        f' --Mapper.fix_existing_images {int(True)}'         #
    )
    f'Run colmap triangulation: {command}' >> logger.debug
    subprocess.run(command.split(), check=True)
    'Finished triangulation' >> logger.debug


def undistort_images(colmap_bin, images_root, sparse_root, dense_root, patch_match_src_n=20):
    r"""Runs COLMAP image undistortion.

    Description of the parameters: https://github.com/colmap/colmap/blob/3.8/src/base/undistortion.h#L43

    Parameters
    ----------
    colmap_bin : str
        Path to the COLMAP executable.
    images_root : str
        Path to the directory with images.
    sparse_root : str
        Path to the sparse reconstruction results.
    dense_root : str
        Path to the output dense reconstruction directory.
    patch_match_src_n : int
        Number of source images for dense stereo, that will be written to the config file
        (see description of run_patch_match).
    """
    command = (
        f'{colmap_bin} image_undistorter'
        f' --image_path {images_root}'
        f' --input_path {sparse_root}'
        f' --output_path {dense_root}'
        f' --output_type COLMAP'
        f' --copy_policy soft-link'  # link images if they are already undistorted
        f' --num_patch_match_src_images {patch_match_src_n}'
        f' --max_image_size -1'  # prevent downsampling
    )
    f'Run image undistortion: {command}' >> logger.debug
    subprocess.run(command.split(), check=True)
    'Finished undistortion' >> logger.debug


def run_patch_match(
        colmap_bin, dense_root, max_image_size=-1, window_radius_px=5, window_step_px=1, sigma_spatial_px=-1,
        sigma_color=.2, mc_samples_n=15, ncc_sigma=.6, min_tri_angle_deg=1, incident_angle_sigma=.9,
        patchmatch_iters_n=5, use_geom_consistency=True, geom_consistency_w=.3, max_geom_inconsistency_px=3,
        do_filtering=True, filt_min_consistent_src_n=2, filt_min_ncc=.1, filt_min_tri_angle_deg=3,
        filt_max_geom_inconsistency_px=1, cache_size_gb=32,
):
    r"""Runs COLMAP patch match stereo.

    Original description of the parameters: https://github.com/colmap/colmap/blob/3.8/src/mvs/patch_match.h#L59

    Parameters
    ----------
    colmap_bin : str
        Path to the COLMAP executable.
    dense_root : str
        Path to the output dense reconstruction directory.
    max_image_size : int
    window_radius_px : int
        Half window size to compute NCC photo-consistency cost.
        Documentation recommends increasing this parameter for better performance in featureless regions,
        in exchange for longer run time.
    window_step_px : int
        Number of pixels to skip when computing NCC. For a value of 1, every pixel is used to compute the NCC.
        For larger values, only every n-th row and column is used and the computation speed thereby increases roughly by
        a factor of window_step^2.
    sigma_spatial_px : float
    sigma_color : float
        Parameters for bilaterally weighted NCC.
        If sigma_spatial_px is negative the value of window_radius_px is used.
        In the paper, \sigma_x and \sigma_g, in Section 4.4.
    mc_samples_n : int
        Number of random Monte Carlo samples (i.e source images) to estimate the value of matching cost.
        In the paper, M, in Section 3.
    ncc_sigma : float
        Spread of the NCC likelihood function.
        In the paper, \sigma_\rho, in Section 3.
    min_tri_angle_deg : float
        Minimum triangulation angle in degrees.
        In the paper, \overline\alpha, in 4.2 -> Triangulation Prior.
    incident_angle_sigma : float
        Spread of the incident angle likelihood function.
        In the paper, \sigma_\kappa, in 4.2 -> Incident Prior,
        but there the formula is exp(-0.5 angle^2 / sigma^2),
        and in the code it is exp(-0.5 (1 - cos angle)^2 / sigma^2).
    patchmatch_iters_n : int
        Number of coordinate descent iterations.
    use_geom_consistency : bool
        Whether to use regularized geometric consistency term.
        In the paper, described in Section 4.5.
    geom_consistency_w : float
        The relative weight of the geometric consistency term w.r.t. to the photo-consistency term.
        In the paper, \eta, in Section 4.5.
    max_geom_inconsistency_px : float
        Maximum geometric consistency cost in terms of the forward-backward reprojection error in pixels.
        In the paper, \psi_{max}, in Section 4.5.
    do_filtering : bool
        Whether to filter depth / normal pixels w.r.t consistency.
        In the paper, described in Section 4.7, but implementation is different,
        see https://github.com/colmap/colmap/blob/3.8/src/mvs/patch_match_cuda.cu#L1091
    filt_min_consistent_src_n : int
        Minimum number of consistent reference-source pixel pairs required for the reference pixel to not be filtered out.
        To be consistent, the pair must satisfy filt_min_ncc (see below), and if use_geom_consistency is True, also
        filt_min_tri_angle_deg and filt_max_geom_inconsistency_px.
        In the paper, s in Section 4.7.
    filt_min_ncc : float
        Minimum NCC value for a pixel pair to be consistent.
    filt_min_tri_angle_deg : float
        Minimum triangulation angle for a pixel pair to be consistent.
    filt_max_geom_inconsistency_px : float
        Maximum forward-backward reprojection error for a pixel pair to be consistent.
        In the paper, \psi_{max}, in Section 4.7.
    cache_size_gb : float
        Cache size in gigabytes for patch match, which keeps the bitmaps, depth maps, and normal maps
        of this number of images in memory. A higher value leads to less disk access and faster computation,
        while a lower value leads to reduced memory usage. Note that a single image can consume a lot of memory,
        if the consistency graph is dense.

    Description of some other patch_match_stereo parameters
    -------------------------------------------------------
    --config_path : str
        Path to view selection config, similar to pair.txt in MVSNet. See dense/stereo/patch-match.cfg for example.
        By default, COLMAP selects 20 source views for each reference view automatically based on sparse reconstruction.

        The format of the file is (based on https://github.com/colmap/colmap/blob/3.8/src/mvs/patch_match.cc#L273)
        ```
        ref_file_name_1
        src_files_description_1
        ref_file_name_2
        src_files_description_2
        ...
        ```

        Where src_files_description can be one of
        ```
        __all__
        __auto__, src_views_n
        src_img_1, src_img_2, ...
        ```
        In the first case, all images are used as source images.
            https://github.com/colmap/colmap/blob/3.8/src/mvs/patch_match.cc#L320
        In the second case, top src_views_n images are used, based on the number of sparse points shared with the
        reference and taking into account triangulation angle.
            https://github.com/colmap/colmap/blob/3.8/src/mvs/patch_match.cc#L330
        In the third case, the images from the config are used.

    --PatchMatchStereo.depth_min : float
    --PatchMatchStereo.depth_max : float
        Depth hypotheses range.
        If the values are negative values (default), COLMAP uses min and max depth from sparse reconstruction.
    """

    command = (
        f'{colmap_bin} patch_match_stereo'
        f' --workspace_path {dense_root}'
        f' --workspace_format COLMAP'
        f' --PatchMatchStereo.max_image_size {max_image_size}'
        f' --PatchMatchStereo.window_radius {window_radius_px}'
        f' --PatchMatchStereo.window_step {window_step_px}'
        f' --PatchMatchStereo.sigma_spatial {sigma_spatial_px}'
        f' --PatchMatchStereo.sigma_color {sigma_color}'
        f' --PatchMatchStereo.num_samples {mc_samples_n}'
        f' --PatchMatchStereo.ncc_sigma {ncc_sigma}'
        f' --PatchMatchStereo.min_triangulation_angle {min_tri_angle_deg}'
        f' --PatchMatchStereo.incident_angle_sigma {incident_angle_sigma}'
        f' --PatchMatchStereo.num_iterations {patchmatch_iters_n}'
        f' --PatchMatchStereo.geom_consistency {int(use_geom_consistency)}'
        f' --PatchMatchStereo.geom_consistency_regularizer {geom_consistency_w}'
        f' --PatchMatchStereo.geom_consistency_max_cost {max_geom_inconsistency_px}'
        f' --PatchMatchStereo.filter {int(do_filtering)}'
        f' --PatchMatchStereo.filter_min_ncc {filt_min_ncc}'
        f' --PatchMatchStereo.filter_min_triangulation_angle {filt_min_tri_angle_deg}'
        f' --PatchMatchStereo.filter_min_num_consistent {filt_min_consistent_src_n}'
        f' --PatchMatchStereo.filter_geom_consistency_max_cost {filt_max_geom_inconsistency_px}'
        f' --PatchMatchStereo.cache_size {cache_size_gb}'
    )
    f'Run patch match: {command}' >> logger.debug
    subprocess.run(command.split(), check=True)
    'Finished patch match' >> logger.debug


def fuse_points(
        colmap_bin, dense_root, input_type='geometric', min_pix_n=5, max_pix_n=10_000, max_traversal_depth=100,
        max_reproj_err_px=2, max_depth_err=.01, max_normal_err_deg=10, max_images_to_fuse_n=50, threads_n=-1,
):
    r"""Runs COLMAP stereo fusion.

    Original description of the parameters: https://github.com/colmap/colmap/blob/3.8/src/mvs/fusion.h#L56

    Parameters
    ----------
    colmap_bin : str
        Path to the COLMAP executable.
    dense_root : str
        Path to the output dense reconstruction directory.
    input_type : {'geometric', 'photometric'}
        Whether to use depth- and normal-maps estimated without geometric verification or with it.
    min_pix_n : int
        Minimum number of fused pixels to produce a point.
    max_pix_n : int
        Maximum number of pixels to fuse into a single point.
    max_traversal_depth : int
        Maximum depth in consistency graph traversal.
    max_reproj_err_px : float
        Maximum relative difference between measured and projected pixel.
    max_depth_err : float
        Maximum relative difference between measured and projected depth.
    max_normal_err_deg : float
        Maximum angular difference in degrees of normals of pixels to be fused.
    max_images_to_fuse_n : int
        Number of overlapping images to transitively check for fusing points.
    threads_n : int
        Number of threads to use.
    """
    command = (
        f'{colmap_bin} stereo_fusion'
        f' --workspace_path {dense_root}'
        f' --workspace_format COLMAP'
        f' --input_type {input_type}'
        f' --output_type PLY'
        f' --output_path {dense_root}/fused.ply'
        f' --StereoFusion.num_threads {threads_n}'
        f' --StereoFusion.max_image_size -1'  # prevent image donwsampling
        f' --StereoFusion.min_num_pixels {min_pix_n}'
        f' --StereoFusion.max_num_pixels {max_pix_n}'
        f' --StereoFusion.max_traversal_depth {max_traversal_depth}'
        f' --StereoFusion.max_reproj_error {max_reproj_err_px}'
        f' --StereoFusion.max_depth_error {max_depth_err}'
        f' --StereoFusion.max_normal_error {max_normal_err_deg}'
        f' --StereoFusion.check_num_images {max_images_to_fuse_n}'
    )
    f'Run colmap stereo fusion: {command}' >> logger.debug
    subprocess.run(command.split(), check=True)
    'Finished fusion' >> logger.debug


if __name__ == '__main__':
    main()
