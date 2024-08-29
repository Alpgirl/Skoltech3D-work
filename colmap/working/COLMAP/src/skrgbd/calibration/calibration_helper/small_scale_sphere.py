from time import sleep

from IPython.display import clear_output, display

from skrgbd.calibration.trajectories.camera_plane import CameraCalibrationPlane
from skrgbd.calibration.trajectories.camera_sphere import CameraCalibrationSphere
from skrgbd.calibration.trajectories.phone_plane import PhoneCalibrationPlane
from skrgbd.calibration.trajectories.stl_plane import STLCalibrationPlane
from skrgbd.calibration.trajectories.stl_sphere import STLCalibrationSphere

from skrgbd.devices.rig import Rig
from skrgbd.devices.robot.robot_on_plane import RobotOnPlane
from skrgbd.devices.robot.robot_on_sphere import RobotOnSphere, RobotOnSTLSphere

from skrgbd.utils.logging import logger
from skrgbd.utils.parallel import ThreadSet, PropagatingThread as Thread
from skrgbd.utils.camera_utils import auto_white_balance


class SmallScaleSphereCalibrationHelper:
    name = 'S1Calibration'
    warmup_shoot_interval = 1.5

    def __init__(self, images_root):
        self.images_root = images_root

        self.plane_board_poses = [
            'front', 'tilted_left', 'tilted_right', 'far', 'close', 'turned_left', 'turned_left_left', 'turned_right',
            'turned_right_right', 'turned_up', 'turned_up_up', 'turned_down', 'turned_down_down']
        self.sphere_board_poses = [
            'front', 'tilted_left', 'tilted_right', 'turned_left', 'turned_right', 'turned_up', 'turned_down',]

        self.rig = Rig(stl_right=True)
        self.rig.init_cameras().join()
        self.rig.start_cameras().join()
        self.cameras = self.rig.cameras
        self.stl_right = self.rig.stl_right
        self.realsense = self.rig.realsense
        self.tis_left = self.rig.tis_left
        self.tis_right = self.rig.tis_right
        self.kinect = self.rig.kinect
        self.phone_left = self.rig.phone_left
        self.phone_right = self.rig.phone_right

    def stop(self):
        threads = ThreadSet([Thread(target=camera.stop_warmup) for camera in self.cameras])
        threads.start_and_join()
        self.rig.stop_cameras().join()

    def setup_cameras(self):
        self.realsense.laser_off()
        for camera in self.cameras:
            camera.setup('room_lights', 'calibration')
        auto_white_balance([self.realsense, self.tis_left, self.tis_right])

    def start_initial_warmup(self):
        for camera in self.cameras:
            camera.start_warmup(self.warmup_shoot_interval)

    def do_internal_calibration(self):
        self.realsense.stop_warmup()
        for i in range(len(self.cameras)):
            if self.cameras[i] is self.realsense:
                break
        self.cameras.pop(i)
        self.realsense.__del__()
        input(f'Проведите внутреннюю заводскую процедуру калибровки для RealSense, и после нажмите Enter в поле ниже.')

        self.stop()
        print('Теперь перезапустите процедуру калибровки (перезапустите IPython ядро),'
              ' запустите начальный прогрев камер на 5 минут, после чего продолжайте со следующего шага.')

    def calibrate_on_camera_plane(self, board_poses=None):
        if board_poses is None:
            board_poses = self.plane_board_poses
        calibrated_cameras = {
            self.realsense: {'image', 'ir', 'ir_right'},
            self.tis_left: {'image'},
            self.tis_right: {'image'},
            self.kinect: {'image', 'ir'},
            self.phone_left: {'image'},
            self.phone_right: {'image'}
        }
        trajectory = CameraCalibrationPlane(RobotOnPlane())
        self.calibrate_on_trajectory(
            f'{self.images_root}/camera_plane', calibrated_cameras, trajectory, board_poses, .1, .5)

    def calibrate_on_phone_ir_plane(self, board_poses=None):
        if board_poses is None:
            board_poses = self.plane_board_poses
        calibrated_cameras = {
            self.phone_left: {'image', 'ir'},
            self.phone_right: {'image', 'ir'}
        }
        trajectory = PhoneCalibrationPlane(RobotOnPlane())
        self.calibrate_on_trajectory(
            f'{self.images_root}/phone_plane', calibrated_cameras, trajectory, board_poses, .1, .5)

    def calibrate_on_camera_sphere(self, board_poses=None):
        if board_poses is None:
            board_poses = self.sphere_board_poses
        calibrated_cameras = {
            self.realsense: {'image'},
            self.tis_left: {'image'},
            self.tis_right: {'image'},
            self.kinect: {'image'},
            self.phone_left: {'image'},
            self.phone_right: {'image'}
        }
        trajectory = CameraCalibrationSphere(RobotOnSphere())
        self.calibrate_on_trajectory(
            f'{self.images_root}/camera_sphere', calibrated_cameras, trajectory, board_poses, .1, .5)

    def calibrate_on_stl_plane(self, board_poses=None):
        if board_poses is None:
            board_poses = self.plane_board_poses
        calibrated_cameras = {
            self.stl_right: {'image'},
            self.realsense: {'image'},
            self.tis_left: {'image'},
            self.tis_right: {'image'},
            self.kinect: {'image'},
        }
        trajectory = STLCalibrationPlane(RobotOnPlane())
        self.calibrate_on_trajectory(
            f'{self.images_root}/stl_plane', calibrated_cameras, trajectory, board_poses, .1, 1.5)

    def calibrate_on_stl_sphere(self, board_poses=None):
        if board_poses is None:
            board_poses = self.sphere_board_poses
        calibrated_cameras = {
            self.stl_right: {'image'},
            self.realsense: {'image'},
            self.tis_left: {'image'},
            self.tis_right: {'image'},
            self.kinect: {'image'},
        }
        trajectory = STLCalibrationSphere(RobotOnSTLSphere())
        self.calibrate_on_trajectory(
            f'{self.images_root}/stl_sphere', calibrated_cameras, trajectory, board_poses, .1, 1.5)

    def calibrate_on_trajectory(self, images_root, calibrated_cameras, trajectory, board_poses, velocity, shaking_timeout):
        threads = ThreadSet([])
        for board_pos in board_poses:
            trajectory.move_zero(velocity)
            calib_or_not = input(
                f'Заряды телефонов {self.phone_left.battery_level} and {self.phone_right.battery_level}.'
                'Начать шаг калибровки?')
            if calib_or_not.startswith('n') or calib_or_not.startswith('N'):
                break
            clear_output()
            threads.join()
            threads = ThreadSet([Thread(target=camera.stop_warmup) for camera in calibrated_cameras.keys()])
            threads.start_and_join()

            display(trajectory.stream_tag(*calibrated_cameras.keys()))
            input(f'Расположите доску в положении {board_pos}')
            trajectory.stop_tag_streaming()
            clear_output()
            self.take_images(images_root, board_pos, calibrated_cameras, trajectory.robot,
                             trajectory.points, velocity, shaking_timeout)
            clear_output()

            threads = ThreadSet([Thread(target=camera.start_warmup, args=[self.warmup_shoot_interval])
                                 for camera in calibrated_cameras.keys()])
            threads.start()
        trajectory.robot.rest(velocity)
        trajectory.robot.stop()
        del trajectory.robot
        threads.join()

    def take_images(self, images_root, board_pos, cameras, robot, points, velocity, shaking_timeout):
        logger.info(f'{self.name}: Image capture for {board_pos}')

        def shoot(point_id, point_pos):
            sleep(shaking_timeout)
            image_name = f'{board_pos}_{point_id}'
            threads = []
            for camera, modalities in cameras.items():
                threads.append(
                    Thread(target=camera.save_calib_data, args=[images_root, image_name, modalities, False]))
            threads = ThreadSet(threads)
            threads.start_and_join()
        robot.move_over_points(points, velocity, closure=shoot, show_progress=True)
        logger.info(f'{self.name}: Image capture for {board_pos} DONE')
