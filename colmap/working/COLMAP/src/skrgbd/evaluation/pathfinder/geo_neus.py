from skrgbd.utils import SimpleNamespace


class GeoNeuSReconstructions:
    def __init__(self, results_root):
        self._results_root = results_root

    def __call__(self, version, scene_name, cam, light_setup):
        return GeoNeuSResults(self._results_root, scene_name, version, cam, light_setup)


class GeoNeuSResults(SimpleNamespace):
    def __init__(self, results_root, scene_name, version, cam, light_setup):
        self.scene_root = f'{results_root}/experiments/geo_neus/{version}/{cam}/{light_setup}/{scene_name}'
        self.mesh = f'{self.scene_root}/mesh.ply'
        self.reconstruction = self.mesh
