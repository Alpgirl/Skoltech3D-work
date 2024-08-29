from skrgbd.utils import SimpleNamespace


class FNeuSReconstructions:
    def __init__(self, results_root):
        self._results_root = results_root

    def __call__(self, version, scene_name, cam, light_setup):
        return FNeuSResults(self._results_root, scene_name, version, cam, light_setup)


class FNeuSResults(SimpleNamespace):
    def __init__(self, results_root, scene_name, version, cam, light_setup):
        self.scene_root = f'{results_root}/experiments/fneus/{version}/{cam}/{light_setup}/{scene_name}'
        self.mesh = f'{self.scene_root}/mesh.ply'
        self.reconstruction = self.mesh
