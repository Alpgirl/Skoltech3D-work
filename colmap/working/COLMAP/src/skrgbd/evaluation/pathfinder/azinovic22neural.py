from skrgbd.utils import SimpleNamespace


class Azinovic22NeuralReconstructions:
    def __init__(self, results_root):
        self._results_root = results_root

    def __call__(self, version, scene_name, cam, light_setup):
        return Azinovic22NeuralUniMVSNetResults(self._results_root, scene_name, version, cam, light_setup)


class Azinovic22NeuralUniMVSNetResults(SimpleNamespace):
    def __init__(self, results_root, scene_name, version, cam, light_setup):
        self.mesh = f'{results_root}/experiments/azinovic22neural/{version}/{cam}/{light_setup}/{scene_name}/mesh.ply'
        self.reconstruction = self.mesh  # LEGACY
