from skrgbd.utils import SimpleNamespace


class PhySGReconstructions:
    def __init__(self, results_root):
        self._results_root = results_root

    def __call__(self, version, scene_name, cam, light_setup):
        return PhySGResults(self._results_root, scene_name, version, cam, light_setup)


class PhySGResults(SimpleNamespace):
    def __init__(self, results_root, scene_name, version, cam, light_setup):
        self.scene_root = f'{results_root}/experiments/physg/{version}/{cam}/{light_setup}/{scene_name}'
        self.mesh = f'{self.scene_root}/mesh.obj'
        self.reconstruction = self.mesh
