from pathlib import Path

from skrgbd.calibration.calibration.calibrator import Localizer
from skrgbd.calibration.calibration.presets.kinect_v2 import KinectV2IRCalibrator, KinectV2RGBCalibrator
from skrgbd.calibration.calibration.presets.realsense import RealSenseIRCalibrator, RealSenseRGBCalibrator
from skrgbd.calibration.calibration.presets.phone import PhoneRGBCalibrator0, PhoneIRCalibrator20, PhoneRGBCalibrator20
from skrgbd.calibration.calibration.presets.stl import STLCalibrator
from skrgbd.calibration.calibration.presets.tis import TISCalibrator

from skrgbd.calibration.eth_tool.dataset import Dataset


if __name__ == '__main__':
    visualize = True
    calib_root = Path('/home/universal/Downloads/dev.sk_robot_rgbd_data/experiments/calibration')

    # Intrinsics and Spheres
    # ----------------------
    camera_plane_images = calib_root / 'images/camera_plane'
    camera_plane_results = calib_root / 'results/camera_plane'
    stl_plane_images = calib_root / 'images/stl_plane'
    stl_plane_results = calib_root / 'results/stl_plane'
    phone_plane_images = calib_root / 'images/phone_plane'
    phone_plane_results = calib_root / 'results/phone_plane'
    camera_sphere_images = calib_root / 'images/camera_sphere'
    camera_sphere_results = calib_root / 'results/camera_sphere'
    stl_sphere_images = calib_root / 'images/stl_sphere'
    stl_sphere_results = calib_root / 'results/stl_sphere'

    # Tables
    # ------
    camera_table_images = calib_root / 'images/camera_table'
    camera_table_results = calib_root / 'results/camera_table'
    stl_table_images = calib_root / 'images/stl_table'
    stl_table_results = calib_root / 'results/stl_table'

    # Calibrate cameras on Camera plane: DONE
    # for (calib_class, name) in [
    #     (RealSenseRGBCalibrator, 'real_sense_rgb'),  # DONE
    #     (RealSenseIRCalibrator, 'real_sense_ir'),  # DONE
    #     (RealSenseIRCalibrator, 'real_sense_ir_right'),  # DONE
    #     (KinectV2RGBCalibrator, 'kinect_v2_rgb'),  # DONE
    #     (KinectV2IRCalibrator, 'kinect_v2_ir'),  # DONE
    #     (TISCalibrator, 'tis_left'),  # DONE
    #     (TISCalibrator, 'tis_right'),  # DONE
    #     (PhoneRGBCalibrator0, 'phone_left_rgb'),  # DONE
    #     (PhoneRGBCalibrator0, 'phone_right_rgb'),  # DONE
    # ]:
    #     calibrator = calib_class(img_dir=f'{camera_plane_images}/{name}', results_dir=f'{camera_plane_results}/{name}')
    #     calibrator.extract_features()
    #     calibrator.calibrate(visualize)

    # Extract features on STL plane, Calibrate STL camera: DONE
    # for (calib_class, name) in [
    #     # (STLCalibrator, 'stl_right'),  # DONE
    #     # (RealSenseRGBCalibrator, 'real_sense_rgb'),  # DONE
    #     # (KinectV2RGBCalibrator, 'kinect_v2_rgb'),  # DONE
    #     # (TISCalibrator, 'tis_left'),  # DONE
    #     # (TISCalibrator, 'tis_right'),  # DONE
    # ]:
    #     calibrator = calib_class(img_dir=f'{stl_plane_images}/{name}', results_dir=f'{stl_plane_results}/{name}')
    #     calibrator.extract_features()
    #     if name == 'stl_right':
    #         calibrator.calibrate(visualize)

    # Calibrate phone IRs: DONE
    # for (calib_class, name) in [
    #     (PhoneRGBCalibrator20, 'phone_left_rgb'),  # DONE
    #     (PhoneIRCalibrator20, 'phone_left_ir'),  # DONE
    #     (PhoneRGBCalibrator20, 'phone_right_rgb'),  # DONE
    #     (PhoneIRCalibrator20, 'phone_right_ir'),  # DONE
    # ]:
    #     calibrator = calib_class(img_dir=f'{phone_plane_images}/{name}', results_dir=f'{phone_plane_results}/{name}')
    #     calibrator.extract_features()
    #     if name.endswith('_ir'):
    #         calibrator.calibrate(visualize)

    # Extract features on Camera Sphere: DONE
    # for (calib_class, name) in [
    #     (RealSenseRGBCalibrator, 'real_sense_rgb'),  # DONE
    #     (KinectV2RGBCalibrator, 'kinect_v2_rgb'),  # DONE
    #     (TISCalibrator, 'tis_left'),  # DONE
    #     (TISCalibrator, 'tis_right'),  # DONE
    #     (PhoneRGBCalibrator0, 'phone_left_rgb'),  # DONE
    #     (PhoneRGBCalibrator0, 'phone_right_rgb'),  # DONE
    # ]:
    #     calibrator = calib_class(img_dir=f'{camera_sphere_images}/{name}', results_dir=f'{camera_sphere_results}/{name}')
    #     calibrator.extract_features()

    # Extract features on STL Sphere: DONE
    # for (calib_class, name) in [
    #     (STLCalibrator, 'stl_right'),  # DONE
    #     (RealSenseRGBCalibrator, 'real_sense_rgb'),  # DONE
    #     (KinectV2RGBCalibrator, 'kinect_v2_rgb'),  # DONE
    #     (TISCalibrator, 'tis_left'),  # DONE
    #     (TISCalibrator, 'tis_right'),  # DONE
    # ]:
    #     calibrator = calib_class(img_dir=f'{stl_sphere_images}/{name}', results_dir=f'{stl_sphere_results}/{name}')
    #     calibrator.extract_features()

    # Extract features on Camera Table: DONE
    # for (calib_class, name) in [
    #     (RealSenseRGBCalibrator, 'real_sense_rgb'),  # DONE
    #     (KinectV2RGBCalibrator, 'kinect_v2_rgb'),  # DONE
    #     (TISCalibrator, 'tis_left'),  # DONE
    #     (TISCalibrator, 'tis_right'),  # DONE
    #     (PhoneRGBCalibrator0, 'phone_left_rgb'),  # DONE
    #     (PhoneRGBCalibrator0, 'phone_right_rgb'),  # DONE
    # ]:
    #     calibrator = calib_class(img_dir=f'{camera_table_images}/{name}', results_dir=f'{camera_table_results}/{name}')
    #     calibrator.extract_features()

    # Extract features on STL Table: DONE
    # for (calib_class, name) in [
    #     (STLCalibrator, 'stl_right'),  # DONE
    #     (RealSenseRGBCalibrator, 'real_sense_rgb'),  # DONE
    #     (KinectV2RGBCalibrator, 'kinect_v2_rgb'),  # DONE
    #     (TISCalibrator, 'tis_left'),  # DONE
    #     (TISCalibrator, 'tis_right'),  # DONE
    # ]:
    #     calibrator = calib_class(img_dir=f'{stl_table_images}/{name}', results_dir=f'{stl_table_results}/{name}')
    #     calibrator.extract_features()

    # Localization
    # ------------
    # calib_dirs = dict()
    # for (calib_class, name) in [
    #     (RealSenseRGBCalibrator, 'real_sense_rgb'),
    #     (RealSenseIRCalibrator, 'real_sense_ir'),
    #     (RealSenseIRCalibrator, 'real_sense_ir_right'),
    #     (KinectV2RGBCalibrator, 'kinect_v2_rgb'),
    #     (KinectV2IRCalibrator, 'kinect_v2_ir'),
    #     (TISCalibrator, 'tis_left'),
    #     (TISCalibrator, 'tis_right'),
    #     (PhoneRGBCalibrator0, 'phone_left_rgb'),
    #     (PhoneRGBCalibrator0, 'phone_right_rgb'),
    # ]:
    #     calibrator = calib_class(img_dir=f'{camera_plane_images}/{name}', results_dir=f'{camera_plane_results}/{name}')
    #     calib_dirs[name] = calibrator.calib_dir
    # for (calib_class, name) in [
    #     (STLCalibrator, 'stl_right'),
    # ]:
    #     calibrator = calib_class(img_dir=f'{stl_plane_images}/{name}', results_dir=f'{stl_plane_results}/{name}')
    #     calib_dirs[name] = calibrator.calib_dir
    # for (calib_class, name) in [
    #     (PhoneIRCalibrator20, 'phone_left_ir'),
    #     (PhoneIRCalibrator20, 'phone_right_ir'),
    # ]:
    #     calibrator = calib_class(img_dir=f'{phone_plane_images}/{name}', results_dir=f'{phone_plane_results}/{name}')
    #     calib_dirs[name] = calibrator.calib_dir

    # Localize RealSense IR and IR right to RGB: DONE
    # cameras = ['real_sense_rgb', 'real_sense_ir', 'real_sense_ir_right']
    # localizer = Localizer(
    #     [calib_dirs[camera] for camera in cameras],
    #     [f'{camera_plane_results}/{camera}/dataset.bin' for camera in cameras],
    #     results_dir=f'{camera_plane_results}/real_sense_all')

    # Localize Kinect IR to RGB: DONE
    # cameras = ['kinect_v2_rgb', 'kinect_v2_ir']
    # localizer = Localizer(
    #     [calib_dirs[camera] for camera in cameras],
    #     [f'{camera_plane_results}/{camera}/dataset.bin' for camera in cameras],
    #     results_dir=f'{camera_plane_results}/kinect_v2_all')

    # Localize RGB cameras to rig: DONE
    # cameras = ['stl_right', 'real_sense_rgb', 'kinect_v2_rgb', 'tis_left', 'tis_right']
    # localizer = Localizer(
    #     [calib_dirs[camera] for camera in cameras],
    #     [f'{stl_plane_results}/{camera}/dataset.bin' for camera in cameras],
    #     results_dir=f'{stl_plane_results}/all')

    # Localize left phone to reference camera: DONE
    # cameras = ['kinect_v2_rgb', 'phone_left_rgb']
    # localizer = Localizer(
    #     [calib_dirs[camera] for camera in cameras],
    #     [f'{camera_plane_results}/{camera}/dataset.bin' for camera in cameras],
    #     results_dir=f'{camera_plane_results}/phone_left_to_{cameras[0]}')

    # Localize right phone to reference camera: DONE
    # cameras = ['kinect_v2_rgb', 'phone_right_rgb']
    # localizer = Localizer(
    #     [calib_dirs[camera] for camera in cameras],
    #     [f'{camera_plane_results}/{camera}/dataset.bin' for camera in cameras],
    #     results_dir=f'{camera_plane_results}/phone_right_to_{cameras[0]}')

    # Localize left phone IR to RGB: DONE
    # cameras = ['phone_left_rgb', 'phone_left_ir']
    # localizer = Localizer(
    #     [calib_dirs[camera] for camera in cameras],
    #     [f'{phone_plane_results}/{camera}/dataset.bin' for camera in cameras],
    #     results_dir=f'{phone_plane_results}/phone_left_all')

    # Localize right phone IR to RGB: DONC
    # cameras = ['phone_right_rgb', 'phone_right_ir']
    # localizer = Localizer(
    #     [calib_dirs[camera] for camera in cameras],
    #     [f'{phone_plane_results}/{camera}/dataset.bin' for camera in cameras],
    #     results_dir=f'{phone_plane_results}/phone_right_all')

    # Merge boards on STL sphere: DONE
    # for camera in [
    #     'stl_right',  # DONE
    #     'real_sense_rgb',  # DONE
    #     'kinect_v2_rgb',  # DONE
    #     'tis_left',  # DONE
    #     'tis_right',  # DONE
    # ]:
    #     dataset = Dataset.fromfile(f'{stl_sphere_results}/{camera}/dataset.bin')
    #     dataset.merge_boards(
    #         ['front', 'tilted_left', 'tilted_right', 'turned_left', 'turned_right', 'turned_up', 'turned_down'])
    #     dataset.save(f'{stl_sphere_results}/{camera}/merged_dataset.bin')

    # Merge boards on Camera sphere: DONE
    # for camera in [
    #     'real_sense_rgb',  # DONE
    #     'kinect_v2_rgb',  # DONE
    #     'tis_left',  # DONE
    #     'tis_right',  # DONE
    #     'phone_left_rgb',  # DONE
    #     'phone_right_rgb',  # DONE
    # ]:
    #     dataset = Dataset.fromfile(f'{camera_sphere_results}/{camera}/dataset.bin')
    #     dataset.merge_boards(
    #         ['front', 'tilted_left', 'tilted_right', 'turned_left', 'turned_right', 'turned_up', 'turned_down'])
    #     dataset.save(f'{camera_sphere_results}/{camera}/merged_dataset.bin')

    # Merge boards on Camera table: DONE
    # for camera in [
    #     'real_sense_rgb',  # DONE
    #     'kinect_v2_rgb',  # DONE
    #     'tis_left',  # DONE
    #     'tis_right',  # DONE
    #     'phone_left_rgb',  # DONE
    #     'phone_right_rgb',  # DONE
    # ]:
    #     dataset = Dataset.fromfile(f'{camera_table_results}/{camera}/dataset.bin')
    #     dataset.merge_boards(
    #         ['front_10', 'front_0', 'front_1', 'front_2', 'front_3', 'front_4',
    #          'front_5', 'front_6', 'front_7', 'front_8', 'front_9'])
    #     dataset.save(f'{camera_table_results}/{camera}/merged_dataset.bin')

    # Merge boards on STL table: DONE
    # for camera in [
    #     'stl_right',  # DONE
    #     'real_sense_rgb',  # DONE
    #     'kinect_v2_rgb',  # DONE
    #     'tis_left',  # DONE
    #     'tis_right',  # DONE
    # ]:
    #     dataset = Dataset.fromfile(f'{stl_table_results}/{camera}/dataset.bin')
    #     dataset.merge_boards(
    #         ['front_10', 'front_0', 'front_1', 'front_2', 'front_3', 'front_4',
    #          'front_5', 'front_6', 'front_7', 'front_8', 'front_9'])
    #     dataset.save(f'{stl_table_results}/{camera}/merged_dataset.bin')

    # Combine positions on STL and Camera spheres: DONE
    # for camera in [
    #     'real_sense_rgb',  # DONE
    #     'kinect_v2_rgb',  # DONE
    #     'tis_left',  # DONE
    #     'tis_right',  # DONE
    # ]:
    #     dataset = Dataset.merge_positions([f'{camera_sphere_results}/{camera}/merged_dataset.bin',
    #                                        f'{stl_sphere_results}/{camera}/merged_dataset.bin'])
    #     dataset.save(f'{camera_sphere_results}/{camera}/merged_dataset_plus_stl_sphere.bin')

    # Combine positions on STL and Camera tables: DONE
    # for camera in [
    #     'real_sense_rgb',  # DONE
    #     'kinect_v2_rgb',  # DONE
    #     'tis_left',  # DONE
    #     'tis_right',  # DONE
    # ]:
    #     dataset = Dataset.merge_positions([f'{camera_table_results}/{camera}/merged_dataset.bin',
    #                                        f'{stl_table_results}/{camera}/merged_dataset.bin'])
    #     dataset.save(f'{camera_table_results}/{camera}/merged_dataset_plus_stl_table.bin')

    # Localize STL camera on STL sphere: DONE
    # localizer = Localizer([calib_dirs['stl_right']], [f'{stl_sphere_results}/stl_right/merged_dataset.bin'],
    #                       results_dir=f'{stl_sphere_results}/stl_right/localization')

    # Localize STL camera on STL table: DONE
    # localizer = Localizer([calib_dirs['stl_right']], [f'{stl_table_results}/stl_right/merged_dataset.bin'],
    #                       results_dir=f'{stl_table_results}/stl_right/localization')

    # localizer.prepare_to_localize()
    # localizer.localize(visualize)

    # Localize cameras on Camera and STL spheres: DONE
    # for camera in [
    #     'real_sense_rgb',  # DONE
    #     'kinect_v2_rgb',  # DONE
    #     'tis_left',  # DONE
    #     'tis_right',  # DONE
    # ]:
    #     localizer = Localizer([calib_dirs[camera]], [f'{camera_sphere_results}/{camera}/merged_dataset_plus_stl_sphere.bin'],
    #                           results_dir=f'{camera_sphere_results}/{camera}/localization')
    #     localizer.prepare_to_localize()
    #     localizer.localize(visualize)

    # Localize cameras on Camera and STL tables: DONE
    # for camera in [
    #     'real_sense_rgb',  # DONE
    #     'kinect_v2_rgb',  # DONE
    #     'tis_left',  # DONE
    #     'tis_right',  # DONE
    # ]:
    #     localizer = Localizer([calib_dirs[camera]], [f'{camera_table_results}/{camera}/merged_dataset_plus_stl_table.bin'],
    #                           results_dir=f'{camera_table_results}/{camera}/localization')
    #     localizer.prepare_to_localize()
    #     localizer.localize(visualize)

    # Localize phones on Camera sphere: DONE
    # for camera in [
    #     'phone_left_rgb',  # DONE
    #     'phone_right_rgb',  # DONE
    # ]:
    #     localizer = Localizer([calib_dirs[camera]], [f'{camera_sphere_results}/{camera}/merged_dataset.bin'],
    #                           results_dir=f'{camera_sphere_results}/{camera}/localization')
    #     localizer.prepare_to_localize()
    #     localizer.localize(visualize)

    # Localize phones on Camera table: DONE
    # for camera in [
    #     'phone_left_rgb',  # DONE
    #     'phone_right_rgb',  # DONE
    # ]:
    #     localizer = Localizer([calib_dirs[camera]], [f'{camera_table_results}/{camera}/merged_dataset.bin'],
    #                           results_dir=f'{camera_table_results}/{camera}/localization')
    #     localizer.prepare_to_localize()
    #     localizer.localize(visualize)
