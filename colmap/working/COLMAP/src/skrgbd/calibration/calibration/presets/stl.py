from skrgbd.calibration.calibration.presets.tag0 import Tag0Calibrator


class STLCalibrator(Tag0Calibrator):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            cell_size=30,
            half_window_size=63,
            **kwargs
        )
