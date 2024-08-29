import torch

from .optimizer import BatchOptimizer
from .utils import check_convergence, strong_wolfe


class BatchBFGS(BatchOptimizer):
    def __init__(
            self, params, lr=None, line_search_fn=None,
            check_convergence=check_convergence, f_rtol=None, f_atol=None, p_rtol=None, p_atol=None
    ):
        r"""BFGS with strong Wolfe line search for optimal step size.

        Parameters
        ----------
        params : iterable of torch.Tensor
            each of shape [batch_size, **]
        lr : float
            Required for None line_search_fn
        line_search_fn : {None, ‘strong_wolfe’}
        """
        super().__init__(params)
        self.line_search_fn = {None: None, 'strong_wolfe': strong_wolfe}[line_search_fn]
        self.lr = lr
        self.check_convergence = check_convergence
        self.f_rtol = f_rtol if (f_rtol is not None) else torch.finfo(self.params[0].dtype).eps * 10
        self.f_atol = f_atol
        self.p_rtol = p_rtol
        self.p_atol = p_atol

        self.f0 = None
        self.g0 = None
        self.hessinv0 = None

    @torch.no_grad()
    def step(self, closure, line_search_max_iters_n=25):
        r"""

        Parameters
        ----------
        closure : callable
            that reevaluates the model and returns the loss batch, with signature
            closure(not_converged_ids) -> loss
                not_converged_ids : torch.LongTensor
                    of shape [nnz_n] with indices of sub-problems that have not converged, sorted in ascending order
                loss : torch.Tensor
                    of shape [nnz_n]

        Returns
        -------
        loss : torch.Tensor or None
            of shape [nnz_n], the value of loss function at the start of the step,
            or None if all sub-problems have converged.
        """
        # 1. Stop if the all sub-problems have converged
        if len(self.not_converged_ids) == 0:
            return

        # 2. Initialize stuff
        closure_orig = torch.enable_grad()(closure)
        closure = lambda: closure_orig(self.not_converged_ids).detach()
        if self.f0 is None:
            self.f0 = closure()
            self.g0 = self._gather_flat_grad()
            self.hessinv0 = torch.eye(self.params_n).to(self.f0)
            self.hessinv0 = self.hessinv0.unsqueeze(0).expand(len(self.f0), -1, -1).contiguous()

        # 3. Calculate update dir
        upd_dir = -(self.hessinv0 @ self.g0.unsqueeze(-1)).squeeze(-1)

        # 4. Find step size
        if self.line_search_fn is None:
            upd = upd_dir * self.lr
            f1, g1 = None, None
        else:
            p0 = self._gather_flat_params()

            def line_search_closure(t):
                self._update_params(upd_dir * t.unsqueeze(1))
                f1 = closure()
                g1 = self._gather_flat_grad()
                self._set_params(p0)
                return f1, g1

            t, f1, g1 = self.line_search_fn(
                line_search_closure, upd_dir, t_max=self.lr, max_iters_n=line_search_max_iters_n); del p0
            upd = upd_dir.mul_(t.unsqueeze(1)); del t
        del upd_dir

        # 5. Update parameters
        self._update_params(upd)
        if f1 is None:
            f1 = closure()
            g1 = self._gather_flat_grad()

        # 6. Update convergence status
        not_converged = slice(None)
        if self.check_convergence:
            p1 = self._gather_flat_params()
            not_converged = self.check_convergence(self.f0, f1, p1, upd, self.f_rtol, self.f_atol, self.p_rtol, self.p_atol)
            self.not_converged_ids = self.not_converged_ids[not_converged]

        # 7. Update memory
        f0, self.f0 = self.f0, f1[not_converged]; del f1
        g0 = self.g0[not_converged]
        g1 = g1[not_converged]
        self.g0 = g1

        # 8. Update inverse Hessian
        y0 = g1 - g0; del g0, g1
        upd = upd[not_converged]
        hessinv0 = self.hessinv0[not_converged]; del not_converged
        hessinv1 = hessinv0.clone()

        curv = y0.unsqueeze(1) @ upd.unsqueeze(2)
        upd = upd.div_(-curv.squeeze(1))

        hessinv1.add_(upd.unsqueeze(2) @ (y0.unsqueeze(1) @ hessinv0))
        hess_dot_y = hessinv0 @ y0.unsqueeze(2); del hessinv0
        hessinv1.add_(hess_dot_y @ upd.unsqueeze(1))

        k = y0.unsqueeze(1) @ hess_dot_y; del hess_dot_y, y0
        k = k.add_(curv); del curv
        k = (upd.unsqueeze(2) @ upd.unsqueeze(1)).mul_(k); del upd
        hessinv1.add_(k); del k

        self.hessinv0 = hessinv1
        return f0
