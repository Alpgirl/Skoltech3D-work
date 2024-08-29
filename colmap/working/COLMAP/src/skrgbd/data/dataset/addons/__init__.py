from skrgbd.utils import SimpleNamespace

from skrgbd.data.dataset.addons.idr_inputs import IDRInputsFinder


class AddonsPathfinder(SimpleNamespace):
    def __init__(self, addons_root=None):
        self.idr_inputs = IDRInputsFinder(addons_root)

    def set_dirs(self, addons_root=None):
        addons_root = addons_root if addons_root else self.addons_root
        self.__init__(addons_root)


addons_pathfinder = AddonsPathfinder()
