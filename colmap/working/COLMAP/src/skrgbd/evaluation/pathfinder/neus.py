from skrgbd.utils import SimpleNamespace


class NeuSReconstructions:
    def __init__(self, results_root):
        self._results_root = results_root

    def __call__(self, version, scene_name, cam, light_setup):
        return NeuSResults(self._results_root, scene_name, version, cam, light_setup)


class NeuSResults(SimpleNamespace):
    def __init__(self, results_root, scene_name, version, cam, light_setup):
        self.mesh = f'{results_root}/experiments/neus/{version}/{cam}/{light_setup}/{scene_name}/mesh.ply'
        self.reconstruction = self.mesh  # LEGACY
