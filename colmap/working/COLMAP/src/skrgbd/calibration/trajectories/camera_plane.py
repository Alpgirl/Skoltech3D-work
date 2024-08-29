from ipywidgets import Box, Layout

from skrgbd.image_processing.utils import make_im_slice
from skrgbd.devices.robot.robot_on_plane import RobotOnPlane
from skrgbd.calibration.trajectories.trajectory import Trajectory


class CameraCalibrationPlane(Trajectory):
    area = (0, 1, 0, 1)
    points = None
    robot = None

    def __init__(self, robot=None):
        if robot is None:
            robot = RobotOnPlane(simulation=True)
        self.robot = robot
        self._stop_streaming = None
        self.points = self.generate_trajectory_points()

    def move_zero(self, velocity):
        x = (self.area[0] + self.area[1]) / 2
        y = (self.area[2] + self.area[3]) / 2
        self.robot.move_to((x, y), velocity)

    def stream_tag(self, realsense, tis_left, tis_right, kinect, phone_left, phone_right):
        realsense_rgb_w = realsense.start_streaming('image', make_im_slice((654, 1048), 134))
        realsense_ir_w = realsense.start_streaming('ir', make_im_slice((426, 661), 64))
        realsense_irr_w = realsense.start_streaming('ir_right', make_im_slice((426, 608), 64))
        tis_left_w = tis_left.start_streaming('image', make_im_slice((1291, 2141), 238))
        tis_right_w = tis_right.start_streaming('image', make_im_slice((1338, 1562), 238))
        kinect_rgb_w = kinect.start_streaming('image', make_im_slice((561, 1023), 103))
        kinect_ir_w = kinect.start_streaming('ir', make_im_slice((208, 305), 36))
        phone_left_w = phone_left.start_streaming('image', make_im_slice((1802, 5396), 580, 4))
        phone_right_w = phone_right.start_streaming('image', make_im_slice((1806, 3593), 580, 4))

        def _stop_streaming():
            for camera in [realsense, tis_left, tis_right, kinect, phone_left, phone_right]:
                camera.stop_streaming('image')
            realsense.stop_streaming('ir')
            kinect.stop_streaming('ir')
            realsense.stop_streaming('ir_right')
        self._stop_streaming = _stop_streaming

        images = [realsense_rgb_w, kinect_rgb_w, tis_left_w, phone_left_w,
                  realsense_ir_w, kinect_ir_w, tis_right_w, phone_right_w,
                  realsense_irr_w]
        for image in images:
            image.width = '220px'
            image.layout.object_fit = 'contain'
        widget = Box(images, layout=Layout(display='flex', flex_flow='row wrap'))
        return widget

    def stop_tag_streaming(self):
        self._stop_streaming()

    def generate_trajectory_points(self, step=None):
        r"""Generates a trajectory with evenly spaced points.
        The trajectory starts in (0, 0), then goes all the way to the right, then one step up,
        then all the way to the left, then one step up, and so on.

        Parameters
        ----------
        step : float
            Distance between points in meters.
            The default value generates 15x23 points
            optimally picked for the calibration board with 11x16 squares of size 0.02875 mm.

        Returns
        -------
        points : torch.Tensor
            of shape [points_n, 2].
        """
        if step is None:
            square_size = 0.02875
            features_n_min = 11
            step = square_size * (1 - 1 / (features_n_min - 1)) * 2.03
        points = self.robot.meshgrid(step, area=self.area, endpoint=True, metric_step_size=True)
        for i in range(1, len(points), 2):
            points[i] = points[i][::-1]
        points = points.reshape(-1, 2)
        return points
