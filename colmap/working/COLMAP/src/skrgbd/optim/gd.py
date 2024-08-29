import torch

from .optimizer import BatchOptimizer
from .utils import check_convergence, strong_wolfe


class BatchGD(BatchOptimizer):
    def __init__(
            self, params, lr=None, line_search_fn=None,
            check_convergence=check_convergence, f_rtol=None, f_atol=None, p_rtol=None, p_atol=None
    ):
        r"""Gradient descent with strong Wolfe line search for optimal step size.

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
        self.f_rtol = f_rtol
        self.f_atol = f_atol
        self.p_rtol = p_rtol
        self.p_atol = p_atol

    @torch.no_grad()
    def step(self, closure):
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
        f0 = closure()
        g0 = self._gather_flat_grad()

        # 3. Calculate update dir
        upd_dir = -g0; del g0

        # 4. Find step size
        if self.line_search_fn is None:
            upd = upd_dir * self.lr
            f1 = None
        else:
            p0 = self._gather_flat_params()

            def line_search_closure(t):
                self._update_params(upd_dir * t.unsqueeze(1))
                f1 = closure()
                g1 = self._gather_flat_grad()
                self._set_params(p0)
                return f1, g1

            t, f1, g1 = self.line_search_fn(line_search_closure, upd_dir, t_max=self.lr); del g1, p0
            upd = upd_dir.mul_(t.unsqueeze(1)); del t
        del upd_dir

        # 5. Update parameters
        self._update_params(upd)

        # 6. Update convergence status
        if self.check_convergence:
            p1 = self._gather_flat_params()
            f1 = f1 if (f1 is not None) else closure()
            not_converged = self.check_convergence(f0, f1, p1, upd, self.f_rtol, self.f_atol, self.p_rtol, self.p_atol)
            self.not_converged_ids = self.not_converged_ids[not_converged]

        return f0
