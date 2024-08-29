from skrgbd.calibration.calibration.calibrator import Calibrator


class Tag0Calibrator(Calibrator):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            pattern_yamls=(
                '/home/universal/Downloads/dev.sk_robot_rgbd_data/experiments/calibration'
                '/patterns/pattern_resolution_12x17_segments_16_apriltag_0_0.0.yaml'
            ),
            **kwargs
        )
