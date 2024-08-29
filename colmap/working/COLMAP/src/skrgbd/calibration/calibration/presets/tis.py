from skrgbd.calibration.calibration.presets.tag0 import Tag0Calibrator


class TISCalibrator(Tag0Calibrator):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            cell_size=40,
            half_window_size=36,
            **kwargs
        )
