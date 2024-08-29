from itertools import chain, repeat
import shutil
from pathlib import Path

import torch

from skrgbd.calibration.camera_models.central_generic import CentralGeneric
from skrgbd.calibration.eth_tool.ex_poses import get_poses, Poses
from skrgbd.devices.robot.robot_on_sphere import RobotOnSphere, RobotOnSTLSphere
from skrgbd.calibration.trajectories.camera_sphere import CameraCalibrationSphere
from skrgbd.calibration.trajectories.stl_sphere import STLCalibrationSphere


class Calibration:
    r"""

    Notes
    -----
    Structured light scan is calculated in the RangeVision calibration coordinate system (after applying scenematrix and vertexmatrix),
    which is fixed w.r.t the cameras of the structured light scanner.

    Attributes
    ----------
    rv_calib_to_stl_right : torch.Tensor
        of shape [4, 4].
        Transforms from RangeVision calibration coordinates to stl_right camera coordinates.
    stl_sphere_extrinsics : list of torch.Tensor
        of shape [stl_sphere_points_n, 4, 4].
        Transforms from stl_right camera coordinates at a point on STL sphere
        to stl_right camera coordinates at the zero point.
    cam_sphere_extrinsics : dict
        camera_name: list of torch.Tensor
            of shape [camera_sphere_points_n, 4, 4].
            Transforms from the camera coordinates at a point on Camera sphere
            to the camera coordinates at the zero point.
    rig_to_cam : dict
        camera_name: torch.Tensor
            of shape [4, 4].
            Transforms from the rig coordinates at the zero point to the camera coordinates at the zero point.
    cam_model : dict
        camera_name: CentralGeneric
            Intrinsic camera model.
    """

    calib_dir = '/home/universal/Downloads/dev.sk_robot_rgbd_data/experiments/calibration/results/small_scale_sphere'
    cameras = (
        'stl_right',
        'real_sense_rgb',
        'real_sense_ir',
        'real_sense_ir_right',
        'kinect_v2_rgb',
        'kinect_v2_ir',
        'tis_left',
        'tis_right',
        'phone_left_rgb',
        'phone_left_ir',
        'phone_right_rgb',
        'phone_right_ir',
    )

    def __init__(self, calib_dir=None):
        if calib_dir is None:
            calib_dir = self.calib_dir
        else:
            self.calib_dir = calib_dir

        self.rv_calib_to_stl_right = torch.load(f'{calib_dir}/rv_calib_to_stl_right.pt')
        self.stl_extrinsics = self.stl_sphere_extrinsics = torch.load(f'{calib_dir}/stl_right@stl_sphere_to_zero.pt')

        self.cam_model = dict()
        self.rig_to_cam = dict()
        self.cam_extrinsics = self.cam_sphere_extrinsics = dict()
        for camera in self.cameras:
            self.cam_model[camera] = CentralGeneric(f'{calib_dir}/{camera}_intrinsics.yaml',
                                                    f'{calib_dir}/{camera}_inverse_grid.pt')
            self.rig_to_cam[camera] = torch.load(f'{calib_dir}/rig_to_{camera}.pt')
            if camera != 'stl_right':
                self.cam_sphere_extrinsics[camera] = torch.load(f'{calib_dir}/{camera}@camera_sphere_to_zero.pt')


def combine_calibrations():
    r"""

    Notes
    -----
    Here we calculate forward camera models, rig_to_cam, stl_sphere_extrinsics, and cam_sphere_extrinsics.
    Then, inverse camera models are calculated in 3_fit_inverse_models.ipynb,
    and rv_calib_to_stl_right is calculated in 4_sync_rv_and_our_calibrations.ipynb.
    """
    zero_point_i = 0
    phone_ref_cam = 'kinect_v2_rgb'
    phone_ref_zero_point_i = 0
    raw_calib_root = '/home/universal/Downloads/dev.sk_robot_rgbd_data/experiments/calibration/results'

    raw_calib_root = Path(raw_calib_root)
    calib_dir = Path(Calibration.calib_dir)
    calib_dir.mkdir()

    # Copy intrinsics
    for (camera, trajectory) in chain(
            zip(['real_sense_rgb', 'real_sense_ir', 'real_sense_ir_right', 'kinect_v2_rgb', 'kinect_v2_ir', 'tis_left',
                 'tis_right', 'phone_left_rgb', 'phone_right_rgb'],
                repeat('camera_plane')),
            [('stl_right', 'stl_plane')],
            [('phone_left_ir', 'phone_plane')],
            [('phone_right_ir', 'phone_plane')],
    ):
        camera_calib_dir = list(Path(f'{raw_calib_root}/{trajectory}/{camera}').glob('calibration@*'))[0]
        shutil.copy(f'{camera_calib_dir}/intrinsics0.yaml', f'{calib_dir}/{camera}_intrinsics.yaml')

    # Calculate rig_to_cam
    _rig_to_cam = Poses(f'{raw_calib_root}/stl_plane/all/camera_tr_rig.yaml')
    rig_to_cam = dict()
    for i, camera in enumerate(['stl_right', 'real_sense_rgb', 'kinect_v2_rgb', 'tis_left', 'tis_right']):
        rig_to_cam[camera] = _rig_to_cam[i]

    _rig_to_cam = Poses(f'{raw_calib_root}/camera_plane/real_sense_all/camera_tr_rig.yaml')
    rig_to_cam['real_sense_ir'] = _rig_to_cam[1] @ _rig_to_cam[0].inverse() @ rig_to_cam['real_sense_rgb']
    rig_to_cam['real_sense_ir_right'] = _rig_to_cam[2] @ _rig_to_cam[0].inverse() @ rig_to_cam['real_sense_rgb']

    _rig_to_cam = Poses(f'{raw_calib_root}/camera_plane/kinect_v2_all/camera_tr_rig.yaml')
    rig_to_cam['kinect_v2_ir'] = _rig_to_cam[1] @ _rig_to_cam[0].inverse() @ rig_to_cam['kinect_v2_rgb']

    _rig_to_cam = Poses(f'{raw_calib_root}/camera_plane/phone_left_to_{phone_ref_cam}/camera_tr_rig.yaml')
    rig_to_cam['phone_left_rgb'] = _rig_to_cam[1] @ _rig_to_cam[0].inverse() @ rig_to_cam[phone_ref_cam]

    _rig_to_cam = Poses(f'{raw_calib_root}/camera_plane/phone_right_to_{phone_ref_cam}/camera_tr_rig.yaml')
    rig_to_cam['phone_right_rgb'] = _rig_to_cam[1] @ _rig_to_cam[0].inverse() @ rig_to_cam[phone_ref_cam]

    _rig_to_cam = Poses(f'{raw_calib_root}/phone_plane/phone_left_all/camera_tr_rig.yaml')
    rig_to_cam['phone_left_ir'] = _rig_to_cam[1] @ _rig_to_cam[0].inverse() @ rig_to_cam['phone_left_rgb']

    _rig_to_cam = Poses(f'{raw_calib_root}/phone_plane/phone_right_all/camera_tr_rig.yaml')
    rig_to_cam['phone_right_ir'] = _rig_to_cam[1] @ _rig_to_cam[0].inverse() @ rig_to_cam['phone_right_rgb']

    for camera, matrix in rig_to_cam.items():
        torch.save(matrix, f'{calib_dir}/rig_to_{camera}.pt')

    # Calculate stl_sphere_extrinsics
    trajectory = STLCalibrationSphere(RobotOnSTLSphere(simulation=True))
    _world_to_rig, pose_found = get_poses(trajectory,
                                          f'{raw_calib_root}/stl_sphere/stl_right/localization/rig_tr_global.yaml',
                                          f'{raw_calib_root}/stl_sphere/stl_right/localization/dataset.bin')
    assert pose_found.all()
    _rig_to_cam = Poses(f'{raw_calib_root}/stl_sphere/stl_right/localization/camera_tr_rig.yaml')
    stl_sphere_extrinsics = _rig_to_cam[0] @ _world_to_rig[zero_point_i] @ _world_to_rig.inverse() @ _rig_to_cam[0].inverse()
    torch.save(stl_sphere_extrinsics, f'{calib_dir}/stl_right@stl_sphere_to_zero.pt')

    # Calculate cam_sphere_extrinsics
    cam_sphere_extrinsics = dict()
    for camera in ['real_sense_rgb', 'kinect_v2_rgb', 'tis_left', 'tis_right']:
        poses_yaml = f'{raw_calib_root}/camera_sphere/{camera}/localization/rig_tr_global.yaml'
        dataset_bin = f'{raw_calib_root}/camera_sphere/{camera}/localization/dataset.bin'

        trajectory = CameraCalibrationSphere(RobotOnSphere(simulation=True))
        _world_to_rig, pose_found = get_poses(trajectory, poses_yaml, dataset_bin)
        assert pose_found.all()
        trajectory = STLCalibrationSphere(RobotOnSTLSphere(simulation=True))
        _world_to_rig_on_stl, pose_found = get_poses(trajectory, poses_yaml, dataset_bin)
        assert pose_found[zero_point_i]
        _rig_to_cam = Poses(f'{raw_calib_root}/camera_sphere/{camera}/localization/camera_tr_rig.yaml')
        cam_sphere_extrinsics[camera] = (
                _rig_to_cam[0] @ _world_to_rig_on_stl[zero_point_i] @ _world_to_rig.inverse() @ _rig_to_cam[0].inverse())

    refcam = phone_ref_cam
    ref_zero = phone_ref_zero_point_i
    for camera in ['phone_left_rgb', 'phone_right_rgb']:
        poses_yaml = f'{raw_calib_root}/camera_sphere/{camera}/localization/rig_tr_global.yaml'
        dataset_bin = f'{raw_calib_root}/camera_sphere/{camera}/localization/dataset.bin'

        trajectory = CameraCalibrationSphere(RobotOnSphere(simulation=True))
        _world_to_rig, pose_found = get_poses(trajectory, poses_yaml, dataset_bin)
        assert pose_found.all()
        _rig_to_cam = Poses(f'{raw_calib_root}/camera_sphere/{camera}/localization/camera_tr_rig.yaml')

        camera_to_ref_zero = _rig_to_cam[0] @ _world_to_rig[ref_zero] @ _world_to_rig.inverse() @ _rig_to_cam[0].inverse()
        cam_to_ref = rig_to_cam[refcam] @ rig_to_cam[camera].inverse()
        ref_to_cam = rig_to_cam[camera] @ rig_to_cam[refcam].inverse()
        cam_sphere_extrinsics[camera] = (
                ref_to_cam @ cam_sphere_extrinsics[refcam][ref_zero] @ cam_to_ref @ camera_to_ref_zero)

    for (cam, refcam) in [('real_sense_ir', 'real_sense_rgb'), ('real_sense_ir_right', 'real_sense_rgb'),
                          ('kinect_v2_ir', 'kinect_v2_rgb'),
                          ('phone_left_ir', 'phone_left_rgb'), ('phone_right_ir', 'phone_right_rgb')]:
        cam_to_ref = rig_to_cam[refcam] @ rig_to_cam[cam].inverse()
        ref_to_cam = rig_to_cam[cam] @ rig_to_cam[refcam].inverse()
        cam_sphere_extrinsics[cam] = ref_to_cam @ cam_sphere_extrinsics[refcam] @ cam_to_ref

    for camera, matrices in cam_sphere_extrinsics.items():
        torch.save(matrices, f'{calib_dir}/{camera}@camera_sphere_to_zero.pt')
