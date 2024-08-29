import torch


class BatchOptimizer:
    def __init__(self, params):
        r"""

        Parameters
        ----------
        params : iterable of torch.Tensor
            each of shape [batch_size, **]
        """
        self.params = list(params)
        self.batch_size = len(self.params[0])
        for p in self.params:
            if len(p) != self.batch_size:
                raise ValueError('All parameters should have the same batch_size.')
        self.params_n = sum(p[0].numel() for p in self.params)
        self.not_converged_ids = torch.arange(self.batch_size, device=self.params[0].device)

    def zero_grad(self, set_to_none=False):
        r"""Mimics torch.optim.Optimizer.zero_grad."""
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()

    def _gather_flat_params(self):
        params = self.params[0].new_empty([len(self.not_converged_ids), self.params_n])
        offset = 0
        for p in self.params:
            p = p.data.view(self.batch_size, -1)
            p_size = p.shape[1]
            torch.index_select(p, 0, self.not_converged_ids, out=params[:, offset: offset + p_size])
            offset = offset + p_size
        return params

    def _gather_flat_grad(self):
        grad = self.params[0].new_empty([len(self.not_converged_ids), self.params_n])
        offset = 0
        for p in self.params:
            p_grad = p.grad.data.view(self.batch_size, -1)
            p_size = p_grad.shape[1]
            torch.index_select(p_grad, 0, self.not_converged_ids, out=grad[:, offset: offset + p_size])
            offset = offset + p_size
        return grad

    def _set_params(self, value):
        offset = 0
        for p in self.params:
            p = p.data.view(self.batch_size, -1)
            p_size = p.shape[1]
            p.index_copy_(0, self.not_converged_ids, value[:, offset: offset + p_size])
            offset = offset + p_size

    def _update_params(self, upd):
        offset = 0
        for p in self.params:
            p = p.data.view(self.batch_size, -1)
            p_size = p.shape[1]
            p.index_add_(0, self.not_converged_ids, upd[:, offset: offset + p_size])
            offset = offset + p_size
