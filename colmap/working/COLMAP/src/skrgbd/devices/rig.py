from contextlib import contextmanager

from ppadb.client import Client as AdbClient
import torch

from skrgbd.devices.stl_camera import RightSTLCamera
from skrgbd.devices.tis import LeftTisCamera, RightTisCamera
from skrgbd.devices.realsense import RealSense
from skrgbd.devices.kinect import Kinect
from skrgbd.devices.phone import LeftPhone, RightPhone

from skrgbd.calibration.camera_models.central_generic import CentralGeneric
from skrgbd.calibration.eth_tool.ex_poses import Poses
from skrgbd.utils.parallel import ThreadSet, PropagatingThread as Thread


class Rig:
    stl_right = None
    tis_left = None
    tis_right = None
    realsense = None
    kinect = None
    phone_left = None
    phone_right = None

    cameras = None
    phones = None
    not_phones = None

    stl_scan_to = None
    camera_model = None

    _stop_warmup = False
    _camera_warmup_thread = None

    def __init__(
            self, stl_right=False, tis_left=True, tis_right=True, realsense=('rgb', 'ir', 'depth'),
            kinect=('rgb', 'ir', 'depth'), phone_left=True, phone_right=True,
    ):
        self.stl_right = stl_right or None
        self.tis_left = tis_left or None
        self.tis_right = tis_right or None
        self.realsense = realsense or None
        self.kinect = kinect or None
        self.phone_left = phone_left or None
        self.phone_right = phone_right or None

    def init_cameras(self):
        self.cameras = []
        self.phones = []
        self.not_phones = []

        def init_stl():
            if self.stl_right is not None:
                self.stl_right = RightSTLCamera()
                self.cameras.append(self.stl_right)
                self.not_phones.append(self.stl_right)

        def init_tis_left():
            if self.tis_left is not None:
                self.tis_left = LeftTisCamera()
                self.cameras.append(self.tis_left)
                self.not_phones.append(self.tis_left)

        def init_tis_right():
            if self.tis_right is not None:
                self.tis_right = RightTisCamera()
                self.cameras.append(self.tis_right)
                self.not_phones.append(self.tis_right)

        def init_realsense():
            if self.realsense is not None:
                self.realsense = RealSense(
                    rgb_enabled=('rgb' in self.realsense),
                    depth_enabled=('depth' in self.realsense),
                    ir_enabled=('ir' in self.realsense)
                )
                self.cameras.append(self.realsense)
                self.not_phones.append(self.realsense)

        def init_kinect():
            if self.kinect is not None:
                self.kinect = Kinect(
                    rgb_enabled=('rgb' in self.kinect),
                    depth_enabled=('depth' in self.kinect),
                    ir_enabled=('ir' in self.kinect)
                )
                self.cameras.append(self.kinect)
                self.not_phones.append(self.kinect)

        if (self.phone_left is not None) or (self.phone_right is not None):
            client = AdbClient(host='127.0.0.1', port=5037)

        def init_phone_left():
            if self.phone_left is not None:
                self.phone_left = LeftPhone(client)
                self.cameras.append(self.phone_left)
                self.phones.append(self.phone_left)

        def init_phone_right():
            if self.phone_right is not None:
                self.phone_right = RightPhone(client)
                self.cameras.append(self.phone_right)
                self.phones.append(self.phone_right)

        threads = ThreadSet([Thread(target=target) for target in [
            init_stl, init_tis_left, init_tis_right, init_realsense, init_kinect, init_phone_left, init_phone_right]])
        threads.start()
        return threads

    @contextmanager
    def working_cameras(self, cameras=None):
        try:
            self.start_cameras(cameras).join()
            yield None
        finally:
            self.stop_cameras(cameras).join()

    def start_cameras(self, cameras=None):
        if cameras is None:
            cameras = self.cameras
        else:
            cameras = [camera for camera in self.cameras if camera.name in cameras]
        threads = ThreadSet([Thread(target=camera.start) for camera in cameras])
        threads.start()
        return threads

    def stop_cameras(self, cameras=None):
        if cameras is None:
            cameras = self.cameras
        else:
            cameras = [camera for camera in self.cameras if camera.name in cameras]
        threads = ThreadSet([Thread(target=camera.stop) for camera in cameras])
        threads.start()
        return threads

    def _load_camera_parameters(self):
        self.stl_scan_to = scan_to = dict()
        self.camera_model = dict()
        # scale ~= 1.000209267469264

        calib_dir = ('/home/universal/Downloads/dev.sk_robot_rgbd_data/experiments/calibration'
                     '/results/set_7/all')

        scan_to['stl_right'] = torch.load(f'{calib_dir}/from_rv_board_to_stl_camera_right.pt')
        rig_to = Poses(f'{calib_dir}/camera_tr_rig.yaml')
        stl_right_id = 4
        scan_to_rig = rig_to[stl_right_id].inverse() @ scan_to['stl_right']

        for i, camera in enumerate(['kinect_ir', 'kinect_rgb', 'realsense_ir', 'realsense_rgb',
                                    'stl_right', 'tis_left', 'tis_right']):
            self.camera_model[camera] = CentralGeneric(f'{calib_dir}/intrinsics{i}.yaml',
                                                       f'{calib_dir}/inverse_grid{i}.pt')
            if camera != 'stl_right':
                scan_to[camera] = rig_to[i] @ scan_to_rig
                # scan_to[camera][:3, 3] *= scale

        # intrinsics of phone cameras
        calib_dir = ('/home/universal/Downloads/dev.sk_robot_rgbd_data/experiments/calibration'
                     '/results/set_3/all')
        for i, camera in enumerate(['phone_left_ir', 'phone_left_rgb', 'phone_right_ir', 'phone_right_rgb']):
            self.camera_model[camera] = CentralGeneric(f'{calib_dir}/intrinsics{i}.yaml',
                                                       f'{calib_dir}/inverse_grid{i}.pt')

        # RGB cameras in phones
        calib_dir = ('/home/universal/Downloads/dev.sk_robot_rgbd_data/experiments/calibration'
                     '/results/set_1/all')
        rig_to = Poses(f'{calib_dir}/camera_tr_rig.yaml')
        tis_right_id = 7
        scan_to_rig = rig_to[tis_right_id].inverse() @ scan_to['tis_right']
        for i, camera in zip([2, 3], ['phone_left_rgb', 'phone_right_rgb']):
            scan_to[camera] = rig_to[i] @ scan_to_rig

        # IR cameras in phones
        calib_dir = ('/home/universal/Downloads/dev.sk_robot_rgbd_data/experiments/calibration'
                     '/results/set_3/all')
        rig_to = Poses(f'{calib_dir}/camera_tr_rig.yaml')
        phone_right_rgb_id = 3
        scan_to_rig = rig_to[phone_right_rgb_id].inverse() @ scan_to['phone_right_rgb']
        for i, camera in zip([0, 2], ['phone_left_ir', 'phone_right_ir']):
            scan_to[camera] = rig_to[i] @ scan_to_rig
