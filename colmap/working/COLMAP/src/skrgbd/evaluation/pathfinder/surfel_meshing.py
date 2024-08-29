from skrgbd.utils import SimpleNamespace


class SurfelMeshingReconstructions:
    def __init__(self, results_root):
        self._results_root = results_root

    def __call__(self, version, scene_name, cam, light_setup=None):
        return SurfelMeshingResults(self._results_root, scene_name, version, cam)


class SurfelMeshingResults(SimpleNamespace):
    def __init__(self, results_root, scene_name, version, cam):
        self.mesh = f'{results_root}/experiments/surfel_meshing/{version}/{cam}/{scene_name}/mesh.obj'
        self.reconstruction = self.mesh  # LEGACY
