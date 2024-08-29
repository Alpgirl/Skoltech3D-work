from argparse import ArgumentParser
from pathlib import Path


from skrgbd.calibration.calibrations.small_scale_sphere import Calibration
from skrgbd.data.dataset.pathfinder import Pathfinder, sensor_to_cam_mode

raise DeprecationWarning
# from skrgbd.data.io.poses import save_poses  # save_poses is changed


def main():
    description = r"""This script saves calibrated extrinsics for all sensors in COLMAP format."""
    parser = ArgumentParser(description=description)
    parser.add_argument('--calib-dir', type=str, required=True)
    parser.add_argument('--processed-scans-dir', type=str, required=True)
    args = parser.parse_args()

    calibration = Calibration(args.calib_dir)
    pathfinder = Pathfinder(data_root=args.processed_scans_dir)

    for sensor in {'real_sense_rgb', 'real_sense_ir', 'real_sense_ir_right', 'kinect_v2_rgb', 'kinect_v2_ir',
                   'tis_left', 'tis_right', 'phone_left_rgb', 'phone_left_ir', 'phone_right_rgb', 'phone_right_ir'}:
        save_calib_poses(sensor, pathfinder, calibration)


def save_calib_poses(sensor, pathfinder, calibration, cam_i=0):
    cam, mode = sensor_to_cam_mode[sensor]
    world_to_rig = calibration.rig_to_cam['stl_right'].inverse()
    world_to_cam = calibration.cam_extrinsics[sensor].inverse() @ calibration.rig_to_cam[sensor] @ world_to_rig

    images_txt = pathfinder[cam][mode].calibrated_extrinsics
    Path(images_txt).parent.mkdir(exist_ok=True, parents=True)
    save_poses(images_txt, world_to_cam, sensor, pathfinder, cam_i)


if __name__ == '__main__':
    main()
