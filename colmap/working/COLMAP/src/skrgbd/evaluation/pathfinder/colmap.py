from skrgbd.utils import SimpleNamespace


class ColmapReconstructions:
    def __init__(self, results_root):
        self._results_root = results_root

    def __call__(self, version, scene_name, cam, light_setup):
        return ColmapResults(self._results_root, scene_name, version, cam, light_setup)


class ColmapResults(SimpleNamespace):
    def __init__(self, results_root, scene_name, version, cam, light_setup):
        self.scene_root = f'{results_root}/experiments/colmap/{version}/{cam}/{light_setup}/{scene_name}'
        self.points = f'{self.scene_root}/points.ply'
        self.reconstruction = self.points  # LEGACY
