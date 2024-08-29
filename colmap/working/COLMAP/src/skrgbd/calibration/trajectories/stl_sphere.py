from ipywidgets import Box, Layout

from skrgbd.image_processing.utils import make_im_slice
from skrgbd.devices.robot.robot_on_sphere import RobotOnSTLSphere
from skrgbd.calibration.trajectories.trajectory import Trajectory


class STLCalibrationSphere(Trajectory):
    points = None
    robot = None

    def __init__(self, robot=None):
        if robot is None:
            robot = RobotOnSTLSphere(simulation=True)
        self.robot = robot
        self._stop_streaming = None
        self.points = self.robot.generate_trajectory_points()

    def move_zero(self, velocity):
        self.robot.move_to((.5, .5), velocity)

    def stream_tag(self, stl_right, realsense, tis_left, tis_right, kinect):
        stl_right_w = stl_right.start_streaming('image', make_im_slice((768, 1050), 320))
        realsense_rgb_w = realsense.start_streaming('image', make_im_slice((254, 785), 111))
        tis_left_w = tis_left.start_streaming('image', make_im_slice((586, 1528), 190))
        tis_right_w = tis_right.start_streaming('image', make_im_slice((630, 1057), 190))
        kinect_rgb_w = kinect.start_streaming('image', make_im_slice((279, 819), 84))

        def _stop_streaming():
            for camera in [stl_right, realsense, tis_left, tis_right, kinect]:
                camera.stop_streaming('image')
        self._stop_streaming = _stop_streaming

        images = [stl_right_w, tis_left_w, tis_right_w, realsense_rgb_w,
                  kinect_rgb_w]
        for image in images:
            image.width = '220px'
            image.layout.object_fit = 'contain'
        widget = Box(images, layout=Layout(display='flex', flex_flow='row wrap'))
        return widget

    def stop_tag_streaming(self):
        self._stop_streaming()
