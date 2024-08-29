from skrgbd.utils import SimpleNamespace


class IDRInputsFinder:
    def __init__(self, addons_root):
        self._addons_root = addons_root

    def __call__(self, scene_name, cam, mode):
        return IDRInputs(self._addons_root, scene_name, cam, mode)


class IDRInputs(SimpleNamespace):
    def __init__(self, addons_root, scene_name, cam, mode):
        self.cameras_npz = f'{addons_root}/dataset/{scene_name}/{cam}/{mode}/idr_input/cameras.npz'
