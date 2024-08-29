from pathlib import Path
import shutil
import subprocess

import yaml

from skrgbd.calibration.eth_tool.dataset import Dataset
from skrgbd.utils.logging import logger

calibration_exe = '/home/universal/.local/opt/camera_calibration/build/applications/camera_calibration/camera_calibration'


class Calibrator:
    r"""
    Parameters
    ----------
    pattern_yamls : str
        Comma separated paths to the YAML configurations of the calibration patterns.
    img_dir : str
        Directory with the images of the calibration boards.
    results_dir : str
        Directory to save the calibration results to.
    half_window_size : int
        Half size of the search window for the corner features of the pattern.
        Good starting value is the half-extent of the feature square in pixels.
        "There is a tradeoff here: on the one hand, it should be as small as possible to be able to detect features
        close to the image borders and include little distortion.
        On the other hand, it needs to be large enough to be able to detect features properly.
        Especially if corners are blurred, it is necessary to increase this extent."
    calib_model : {'central_generic'}
    pyramid_levels_n : int
        Number of multi-resolution pyramid levels to use for the resolution of the intrinsics grid
        in bundle adjustment.
        "More levels may improve convergence. Less levels make it applicable to less smooth cameras.
        Setting this to 1 uses only the full resolution. Different settings from 1 are only sensible
        for (generic) camera models with a grid."
    cell_size : int
        Approximate cell side length in pixels.
        Good starting value is 1/50 of the image height.
        Will be slightly adjusted such that the calibrated image area size is an integer multiple of the cell size.
    cuda : bool
        If True, use CUDA for feature extraction and bundle adjustment.
    """

    def __init__(
            self, pattern_yamls, img_dir, results_dir, half_window_size,
            calib_model='central_generic', pyramid_levels_n=4, cell_size=40, cuda=True):
        self.pattern_yamls = pattern_yamls
        self.img_dir = img_dir
        self.results_dir = Path(results_dir)
        self.half_window_size = half_window_size
        self.calib_model = calib_model
        self.pyramid_levels_n = pyramid_levels_n if 'generic' in calib_model else 1
        self.cell_size = cell_size
        self.cuda = cuda

        self.dataset_bin = self.results_dir / 'dataset.bin'
        self.calib_dir = self.results_dir / f'calibration@{calib_model},{cell_size}px'

    def extract_features(self, visualize=False):
        command = (
            f'{calibration_exe}'
            f' --pattern_files {self.pattern_yamls}'
            f' --image_directories {self.img_dir}'
            f' --dataset_output_path {self.dataset_bin}'
            f' --refinement_window_half_extent {self.half_window_size}'
            f'{" --no_cuda_feature_detection" if not self.cuda else ""}'
            f'{" --show_visualizations" if visualize else ""}'
        )
        logger.info(f'Calibration: Extract features with {command}')
        subprocess.run(command.split())

    def calibrate(self, visualize=False):
        calib_model = self.calib_model
        command = (
            f'{calibration_exe}'
            f' --dataset_files {self.dataset_bin}'
            f' --output_directory {self.calib_dir}'
            f' --model {calib_model}'
            f' --num_pyramid_levels {self.pyramid_levels_n}'
            f'{f" --cell_length_in_pixels {self.cell_size}" if "generic" in calib_model else ""}'
            f' --init_try_harder'
            f'{" --schur_mode dense_cuda" if self.cuda else ""}'
            f'{" --show_visualizations" if visualize else ""}'
        )
        logger.info(f'Calibration: Calibrate with {command}')
        subprocess.run(command.split())


class Localizer:
    def __init__(self, calib_dirs, dataset_bins, results_dir, cuda=True):
        self.calib_dirs = calib_dirs
        self.dataset_bins = dataset_bins
        self.results_dir = Path(results_dir)
        self.cuda = cuda
        self.dataset_bin = self.results_dir / 'dataset.bin'

    def prepare_to_localize(self):
        # Copy camera intrinsics and other calibration files to the output directory
        shutil.rmtree(self.results_dir, ignore_errors=True)
        self.results_dir.mkdir()

        for file in 'points.yaml', 'rig_tr_global.yaml':
            shutil.copy(f'{self.calib_dirs[0]}/{file}', self.results_dir / file)
        for i, calib_dir in enumerate(self.calib_dirs):
            shutil.copy(f'{calib_dir}/intrinsics0.yaml', self.results_dir / f'intrinsics{i}.yaml')

        cameras_n = len(self.calib_dirs)
        rig = dict()
        rig['pose_count'] = cameras_n
        rig['poses'] = [dict(index=i, tx=0, ty=0, tz=0, qx=0, qy=0, qz=0, qw=1) for i in range(cameras_n)]
        with open(self.results_dir / 'camera_tr_rig.yaml', 'w') as file:
            yaml.dump(rig, file)

        # Merge feature datasets
        if len(self.dataset_bins) > 1:
            merged_dataset = Dataset.merge_cameras(self.dataset_bins)
            merged_dataset.save(self.dataset_bin)
        else:
            self.dataset_bin = self.dataset_bins[0]

    def localize(self, visualize=False):
        command = (
            f'{calibration_exe}'
            f' --localize_only'
            # f' --outlier_removal_factor 0'
            f' --dataset_files {self.dataset_bin}'
            f' --state_directory {self.results_dir}'
            f' --output_directory {self.results_dir}'
            f'{" --schur_mode dense_cuda" if self.cuda else ""}'
            f'{" --show_visualizations" if visualize else ""}'
        )
        logger.info(f'Calibration: Localize with {command}')
        subprocess.run(command.split())
