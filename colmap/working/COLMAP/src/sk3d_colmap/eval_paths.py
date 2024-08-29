from skrgbd.evaluation.eval_paths import EvalPaths as BaseEvalPaths


method = 'colmap'


class EvalPaths(BaseEvalPaths):
    # Reconstruction
    # --------------
    def rec_subsystem_dir(self, version, cam, light):
        return f'{self.rec_dir(method, version, cam, light)}/subsystem'

    def rec_precomp_sparse_dir(self, version, cam, light):
        return f'{self.rec_subsystem_dir(version, cam, light)}/precomp_sparse'

    def rec_sparse_dir(self, version, cam, light):
        return f'{self.rec_subsystem_dir(version, cam, light)}/sparse/0'

    def rec_dense_dir(self, version, cam, light):
        return f'{self.rec_subsystem_dir(version, cam, light)}/dense'
