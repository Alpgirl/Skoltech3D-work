from skrgbd.utils import SimpleNamespace


class VisMVSNetReconstructions:
    def __init__(self, results_root):
        self._results_root = results_root

    def __call__(self, version, scene_name, cam, light_setup):
        return VisMVSNetResults(self._results_root, scene_name, version, cam, light_setup)


class VisMVSNetResults(SimpleNamespace):
    def __init__(self, results_root, scene_name, version, cam, light_setup):
        self.points = f'{results_root}/experiments/vismvsnet/{version}/{cam}/{light_setup}/{scene_name}/{scene_name}.ply'
        self.reconstruction = self.points  # LEGACY
