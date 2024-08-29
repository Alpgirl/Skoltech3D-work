from skrgbd.utils import SimpleNamespace


class UniMVSNetReconstructions:
    def __init__(self, results_root):
        self._results_root = results_root

    def __call__(self, version, scene_name, cam, light_setup):
        return UniMVSNetResults(self._results_root, scene_name, version, cam, light_setup)


class UniMVSNetResults(SimpleNamespace):
    def __init__(self, results_root, scene_name, version, cam, light_setup):
        self.points = f'{results_root}/experiments/unimvsnet/{version}/{cam}/{light_setup}/{scene_name}/points.ply'
        self.reconstruction = self.points  # LEGACY
