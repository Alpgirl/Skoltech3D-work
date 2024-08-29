from ipywidgets import Box, Layout

from skrgbd.image_processing.utils import make_im_slice
from skrgbd.devices.robot.robot_on_plane import RobotOnPlane
from skrgbd.calibration.trajectories.trajectory import Trajectory


class PhoneCalibrationPlane(Trajectory):
    area = (.33, .98, .21, .72)
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

    def stream_tag(self, phone_left, phone_right):
        phone_left_ir_w = phone_left.start_streaming('ir', make_im_slice((94, 87), 37))
        phone_right_ir_w = phone_right.start_streaming('ir', make_im_slice((87, 157), 37))

        def _stop_streaming():
            for camera in [phone_left, phone_right]:
                camera.stop_streaming('ir')
        self._stop_streaming = _stop_streaming

        images = [phone_left_ir_w, phone_right_ir_w]
        for image in images:
            image.width = '220px'
            image.layout.object_fit = 'contain'
        widget = Box(images, layout=Layout(display='flex', flex_flow='row wrap'))
        return widget

    def stop_tag_streaming(self):
        self._stop_streaming()

    def generate_trajectory_points(self, step=None):
        r"""Generates a trajectory with evenly spaced points.
        The trajectory starts in (left, bottom), then goes all the way to the right, then one step up,
        then all the way to the left, then one step up, and so on.

        Parameters
        ----------
        step : float
            Distance between points in meters.
            The default value generates 10x19 points
            optimally picked for the calibration board with 6x9 squares of size 0.0492 mm.

        Returns
        -------
        points : torch.Tensor
            of shape [points_n, 2].
        """
        if step is None:
            square_size = 0.0492
            features_n_min = 6
            step = square_size * (1 - 1 / features_n_min) * 1.015
        points = self.robot.meshgrid(step, area=self.area, endpoint=True, metric_step_size=True)
        for i in range(1, len(points), 2):
            points[i] = points[i][::-1]
        points = points.reshape(-1, 2)
        return points
