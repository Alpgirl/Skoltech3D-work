class EvalPaths:
    def __init__(self, scene_name, *, eval_root=None, rec_root=None):
        self.scene_name = scene_name
        self.eval_root = eval_root
        self.rec_root = rec_root

    # Reconstruction
    # --------------
    def rec_dir(self, method, version, cam, light=None):
        light_part = '' if (light is None) else f'/{light}'
        return f'{self.rec_root}/{method}/{version}/{cam}{light_part}/{self.scene_name}'

    def rec_pts(self, method, version, cam, light=None):
        return f'{self.rec_dir(method, version, cam, light)}/points.ply'

    def rec_mesh(self, method, version, cam, light=None):
        return f'{self.rec_dir(method, version, cam, light)}/mesh.ply'

    def rec_meta(self, method, version, cam, light=None):
        return f'{self.rec_dir(method, version, cam, light)}/meta.yaml'

    # Evaluation
    # ----------
    def eval_dir(self, method, version, cam, light=None):
        light_part = '' if (light is None) else f'/{light}'
        return f'{self.eval_root}/{method}/{version}/{cam}{light_part}/{self.scene_name}'

    def distribs(self, method, version, cam, light=None):
        return f'{self.eval_dir(method, version, cam, light)}/distributions/distributions.pt'

    def distribs_meta(self, method, version, cam, light=None):
        return f'{self.eval_dir(method, version, cam, light)}/distributions/distributions_meta.yaml'

    def stats(self, method, version, cam, light=None):
        return f'{self.eval_dir(method, version, cam, light)}/stats/stats.db'

    def stats_meta(self, method, version, cam, light=None):
        return f'{self.eval_dir(method, version, cam, light)}/stats/stats_meta.yaml'

    def vis(self, var, method, version, cam, light=None):
        vis_dir = f'{self.eval_dir(method, version, cam, light)}/visualizations'
        if var == 'ref':
            return f'{vis_dir}/reference.png'
        elif var == 'rec':
            return f'{vis_dir}/reconstruction.png'
        elif var == 'acc':
            return f'{vis_dir}/accuracy.png'
        elif var == 'surf_acc':
            return f'{vis_dir}/surf_accuracy.png'
        elif var == 'comp':
            return f'{vis_dir}/completeness.png'

    def vis_meta(self, method, version, cam, light=None):
        return f'{self.eval_dir(method, version, cam, light)}/visualizations/visualizations_meta.yaml'

    def dist_figs(self, version):
        return f'{self.eval_root}/figures/{version}/{self.scene_name}/distances.pdf'

    def all_stats(self):
        return f'{self.eval_root}/all_stats.db'
