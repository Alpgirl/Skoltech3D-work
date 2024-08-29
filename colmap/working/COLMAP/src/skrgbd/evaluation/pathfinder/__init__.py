from skrgbd.utils import SimpleNamespace

from .azinovic22neural import Azinovic22NeuralReconstructions
from .acmp import ACMPReconstructions
from .colmap import ColmapReconstructions
from .fneus import FNeuSReconstructions
from .geo_neus import GeoNeuSReconstructions
from .indi_sg import IndiSGReconstructions
from .neus import NeuSReconstructions
from .physg import PhySGReconstructions
from .routed_fusion import RoutedFusionReconstructions
from .spsr import SPSRReconstructions
from .surfel_meshing import SurfelMeshingReconstructions
from .tsdf_fusion import TSDFFusionReconstructions
from .vismvsnet import VisMVSNetReconstructions
from .unimvsnet import UniMVSNetReconstructions


class Pathfinder(SimpleNamespace):
    def __init__(self, results_root):
        self.results_root = results_root
        self.reconstructions = SimpleNamespace(
            colmap=ColmapReconstructions(results_root),
            acmp=ACMPReconstructions(results_root),
            vismvsnet=VisMVSNetReconstructions(results_root),
            unimvsnet=UniMVSNetReconstructions(results_root),
            neus=NeuSReconstructions(results_root),
            geo_neus=GeoNeuSReconstructions(results_root),
            fneus=FNeuSReconstructions(results_root),
            tsdf_fusion=TSDFFusionReconstructions(results_root),
            surfel_meshing=SurfelMeshingReconstructions(results_root),
            routed_fusion=RoutedFusionReconstructions(results_root),
            azinovic22neural=Azinovic22NeuralReconstructions(results_root),
            spsr_colmap=SPSRReconstructions(results_root, 'colmap'),
            spsr_acmp=SPSRReconstructions(results_root, 'acmp'),
            spsr_vismvsnet=SPSRReconstructions(results_root, 'vismvsnet'),
            spsr_unimvsnet=SPSRReconstructions(results_root, 'unimvsnet'),
            indi_sg=IndiSGReconstructions(results_root),
            physg=PhySGReconstructions(results_root),
        )
        self.evaluation = SimpleNamespace(
            all_stats=f'{results_root}/evaluation/all_stats.db',
            figures=Figures(results_root),
            colmap=MethodEvaluation('colmap', results_root),
            acmp=MethodEvaluation('acmp', results_root),
            vismvsnet=MethodEvaluation('vismvsnet', results_root),
            unimvsnet=MethodEvaluation('unimvsnet', results_root),
            neus=MethodEvaluation('neus', results_root),
            geo_neus=MethodEvaluation('geo_neus', results_root),
            fneus=MethodEvaluation('fneus', results_root),
            tsdf_fusion=MethodEvaluation('tsdf_fusion', results_root),
            surfel_meshing=MethodEvaluation('surfel_meshing', results_root),
            routed_fusion=MethodEvaluation('routed_fusion', results_root),
            azinovic22neural=MethodEvaluation('azinovic22neural', results_root),
            spsr_colmap=MethodEvaluation('spsr_colmap', results_root),
            spsr_acmp=MethodEvaluation('spsr_acmp', results_root),
            spsr_vismvsnet=MethodEvaluation('spsr_vismvsnet', results_root),
            spsr_unimvsnet=MethodEvaluation('spsr_unimvsnet', results_root),
            indi_sg=MethodEvaluation('indi_sg', results_root),
            physg=MethodEvaluation('physg', results_root),
        )

    def set_dirs(self, results_root=None):
        results_root = results_root if results_root else self.results_root
        self.__init__(results_root)


class MethodEvaluation:
    def __init__(self, method_name, results_root):
        self._method_name = method_name
        self._results_root = results_root

    def __call__(self, version, scene_name, cam, light_setup):
        return MethodEvalResults(self._results_root, self._method_name, version, scene_name, cam, light_setup)


class MethodEvalResults(SimpleNamespace):
    def __init__(self, results_root, method_name, version, scene_name, cam, light_setup=None):
        if light_setup is None:
            light_setup_str = ''
        else:
            light_setup_str = f'/{light_setup}'
        scene_dir = f'{results_root}/evaluation/{method_name}/{version}/{cam}{light_setup_str}/{scene_name}'

        self.distributions = SimpleNamespace(data=f'{scene_dir}/distributions/distributions.pt',
                                             meta=f'{scene_dir}/distributions/distributions_meta.yaml')
        self.vox_distributions = SimpleNamespace(data=f'{scene_dir}/vox_distributions/vox_distributions.pt',
                                                 meta=f'{scene_dir}/vox_distributions/vox_distributions_meta.yaml')
        self.stats = SimpleNamespace(data=f'{scene_dir}/stats/stats.db',
                                     meta=f'{scene_dir}/stats/stats_meta.yaml')
        self.visualizations = SimpleNamespace(reference=f'{scene_dir}/visualizations/reference.png',
                                              reconstruction=f'{scene_dir}/visualizations/reconstruction.png',
                                              accuracy=f'{scene_dir}/visualizations/accuracy.png',
                                              surf_accuracy=f'{scene_dir}/visualizations/surf_accuracy.png',
                                              completeness=f'{scene_dir}/visualizations/completeness.png',
                                              meta=f'{scene_dir}/visualizations/visualizations_meta.yaml')


class Figures:
    def __init__(self, results_root):
        self._results_root = results_root

    def __call__(self, config_ver, scene_name):
        return SceneFigures(self._results_root, config_ver, scene_name)


class SceneFigures(SimpleNamespace):
    def __init__(self, results_root, config_ver, scene_name):
        self.distances = f'{results_root}/evaluation/figures/{config_ver}/{scene_name}/distances.pdf'


eval_pathfinder = Pathfinder(None)
