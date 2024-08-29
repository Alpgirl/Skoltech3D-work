from skrgbd.utils import SimpleNamespace


class RoutedFusionReconstructions:
    def __init__(self, results_root):
        self._results_root = results_root

    def __call__(self, version, scene_name, cam, light_setup=None):
        return RoutedFusionResults(self._results_root, scene_name, version, cam)


class RoutedFusionResults(SimpleNamespace):
    def __init__(self, results_root, scene_name, version, cam):
        self.mesh = f'{results_root}/experiments/routed_fusion/{version}/{cam}/{scene_name}/mesh.ply'
        self.reconstruction = self.mesh  # LEGACY
