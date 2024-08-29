from skrgbd.utils import SimpleNamespace


class TSDFFusionReconstructions:
    def __init__(self, results_root):
        self._results_root = results_root

    def __call__(self, version, scene_name, cam, light_setup=None):
        return TSDFFusionResults(self._results_root, scene_name, version, cam)


class TSDFFusionResults(SimpleNamespace):
    def __init__(self, results_root, scene_name, version, cam):
        self.mesh = f'{results_root}/experiments/tsdf_fusion/{version}/{cam}/{scene_name}/reconstruction.ply'
        self.meta = f'{results_root}/experiments/tsdf_fusion/{version}/{cam}/{scene_name}/meta.yaml'
        self.reconstruction = self.mesh  # LEGACY
