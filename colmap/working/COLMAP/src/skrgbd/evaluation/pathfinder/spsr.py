from skrgbd.utils import SimpleNamespace


class SPSRReconstructions:
    def __init__(self, results_root, mvs_method):
        self._results_root = results_root
        self._mvs_method = mvs_method

    def __call__(self, version, scene_name, cam, light_setup):
        return SPSRResults(self._mvs_method, self._results_root, scene_name, version, cam, light_setup)


class SPSRResults(SimpleNamespace):
    def __init__(self, mvs_method, results_root, scene_name, version, cam, light_setup):
        self.mesh = f'{results_root}/experiments/spsr/{mvs_method}/{version}/{cam}/{light_setup}/{scene_name}/reconstruction.ply'
        self.meta = f'{results_root}/experiments/spsr/{mvs_method}/{version}/{cam}/{light_setup}/{scene_name}/meta.yaml'
        self.reconstruction = self.mesh
