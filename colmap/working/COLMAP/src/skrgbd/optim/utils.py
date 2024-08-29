import numpy as np
import torch


@torch.no_grad()
def strong_wolfe(closure, upd_dir, t_max=None, c1=1e-4, c2=.9, max_iters_n=25, t_atol=1e-9):
    r"""
    Parameters
    ----------
    upd_dir : torch.Tensor
        of shape [batch_size, params_n]
    c1 : float
    c2 : float
    """
    t_max = t_max if (t_max is not None) else np.inf
    closure = torch.enable_grad()(closure)

    t0 = torch.zeros_like(upd_dir[:, 0])
    f0, g0 = closure(t0)
    gtd0 = (upd_dir * g0).sum(1)
    rhs_armijo = gtd0 * c1
    rhs_wolfe = -gtd0 * c2

    t_low, f_low, g_low, gtd_low = t0.clone(), f0.clone(), g0.clone(), gtd0.clone()
    t_high = torch.full_like(upd_dir[:, 0], np.inf)
    f_high, g_high, gtd_high = f0.clone(), g0.clone(), gtd0.clone(); del gtd0

    def set_low(mask, t, f, g, gtd):
        nonlocal t_low, f_low, g_low, gtd_low
        t_low = t.where(mask, t_low)
        f_low = f.where(mask, f_low)
        g_low = g.where(mask.unsqueeze(1).expand_as(g_prev), g_low)
        gtd_low = gtd.where(mask, gtd_low)

    def set_high(mask, t, f, g, gtd):
        nonlocal t_high, f_high, g_high, gtd_high
        t_high = t.where(mask, t_high)
        f_high = f.where(mask, f_high)
        g_high = g.where(mask.unsqueeze(1).expand_as(g_prev), g_high)
        gtd_high = gtd.where(mask, gtd_high)

    # Bracket phase
    t_prev, f_prev, g_prev, gtd_prev = t_low.clone(), f_low.clone(), g_low.clone(), gtd_low.clone()
    t_new = torch.full_like(t_low, min(1, t_max))
    bracket_not_found = torch.ones_like(t_new, dtype=torch.bool)

    iter_i = 0
    while iter_i < max_iters_n:
        if bracket_not_found.any().logical_not():
            break
        f_new, g_new = closure(t_new)
        gtd_new = (upd_dir * g_new).sum(1).abs()

        armijo_violated = (f_new - f0).gt(rhs_armijo * t_new)
        f_increasing = f_new.ge(f_prev) if (iter_i > 0) else armijo_violated.new_tensor(False)
        cond_1 = armijo_violated.logical_or_(f_increasing).logical_and_(bracket_not_found); del armijo_violated, f_increasing
        set_low(cond_1, t_prev, f_prev, g_prev, gtd_prev)
        set_high(cond_1, t_new, f_new, g_new, gtd_new)
        bracket_not_found = bracket_not_found.logical_and_(cond_1.logical_not_()); del cond_1

        wolfe_satisfied = gtd_new.abs().le(rhs_wolfe).logical_and_(bracket_not_found)
        set_low(wolfe_satisfied, t_new, f_new, g_new, gtd_new)
        set_high(wolfe_satisfied, t_new, f_new, g_new, gtd_new)
        bracket_not_found = bracket_not_found.logical_and_(wolfe_satisfied.logical_not_()); del wolfe_satisfied

        ascending = gtd_new.ge(0).logical_and_(bracket_not_found)
        set_low(ascending, t_new, f_new, g_new, gtd_new)
        set_high(ascending, t_new, f_new, g_new, gtd_new)
        bracket_not_found = bracket_not_found.logical_and_(ascending.logical_not_()); del ascending

        tmin = (t_new - t_prev).mul_(1e-2).add_(t_new).clamp_(max=t_max)
        tmax = (t_new * 10).clamp_(max=t_max)
        old_t_new = t_new
        t_new = _cubic_interpolate(t_prev, f_prev, gtd_prev, t_new, f_new, gtd_new, tmin, tmax); del tmin, tmax
        t_new = t_new.where(bracket_not_found, old_t_new)
        t_prev, f_prev, g_prev, gtd_prev = old_t_new, f_new, g_new, gtd_new; del old_t_new, f_new, g_new, gtd_new
        iter_i += 1
    if iter_i == max_iters_n:
        set_low(bracket_not_found, t0, f0, g0, gtd_low)
        set_high(bracket_not_found, t_prev, f_prev, g_prev, gtd_high)
    del t0, g0

    _ = f_low < f_high
    t_low, t_high = t_low.where(_, t_high), t_high.where(_, t_low)
    f_low, f_high = f_low.where(_, f_high), f_high.where(_, f_low)
    g_low, g_high = g_low.where(_.unsqueeze(1).expand_as(g_low), g_high), g_high.where(_.unsqueeze(1).expand_as(g_low), g_low)
    if iter_i == max_iters_n:
        return t_low, f_low, g_low
    gtd_low, gtd_high = gtd_low.where(_, gtd_high), gtd_high.where(_, gtd_low); del _

    # Zoom phase
    t_not_found = (t_high - t_low).abs_().gt(t_atol)
    while iter_i < max_iters_n:
        t_new = _cubic_interpolate(t_low, f_low, gtd_low, t_high, f_high, gtd_high)
        f_new, g_new = closure(t_new)
        gtd_new = (upd_dir * g_new).sum(1).abs()

        armijo_violated = (f_new - f0).gt(rhs_armijo * t_new)
        f_increasing = f_new.ge(f_low)
        cond_1 = armijo_violated.logical_or_(f_increasing); del armijo_violated, f_increasing
        set_high(cond_1, t_new, f_new, g_new, gtd_new)

        wolfe_violated = gtd_new.abs().gt(rhs_wolfe)
        flip = (t_high - t_low).mul_(gtd_new).ge(0)
        cond_2 = flip.logical_and_(wolfe_violated); del flip
        set_high(cond_2, t_low, f_low, g_low, gtd_low); del cond_2

        new_found = cond_1.logical_not().logical_and_(t_not_found)
        set_low(new_found, t_new, f_new, g_new, gtd_new); del new_found, t_new, f_new, g_new, gtd_new

        t_not_found = t_not_found.logical_and_(cond_1.logical_or_(wolfe_violated)); del wolfe_violated
        t_not_found = t_not_found.logical_and_((t_high - t_low).abs_().gt(t_atol))
        iter_i += 1
        if t_not_found.any().logical_not():
            break
    return t_low, f_low, g_low


def _cubic_interpolate(x1, f1, g1, x2, f2, g2, xmin=None, xmax=None):
    # ported from https://github.com/pytorch/pytorch/blob/master/torch/optim/lbfgs.py
    # assert (x1 <= x2).all()
    _ = x1 <= x2
    x1, x2 = x1.where(_, x2), x2.where(_, x1)
    f1, f2 = f1.where(_, f2), f2.where(_, f1)
    g1, g2 = g1.where(_, g2), g2.where(_, g1); del _
    if xmin is None:
        xmin = x1
        xmax = x2
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1 ** 2 - g1 * g2
    d2 = d2_square.sqrt()
    min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
    min_pos = xmax.min(xmin.max(min_pos)).where(d2_square >= 0, (x1 + x2) / 2)
    return min_pos


def check_convergence(f0, f1, p, upd, f_rtol, f_atol, p_rtol, p_atol):
    assert any([f_rtol is not None, f_atol is not None, p_rtol is not None, p_atol is not None])
    not_converged = f0.new_tensor(False, dtype=torch.bool)
    if (f_atol is not None) or (f_rtol is not None):
        f_delta = (f1 - f0).abs()
        f_not_converged = f0.new_tensor(True, dtype=torch.bool)
        if f_rtol is not None:
            f_not_converged = (f_delta > f0.abs().mul_(f_rtol)).logical_and_(f_not_converged)
        if f_atol is not None:
            f_not_converged = (f_delta > f_atol).logical_and_(f_not_converged)
        del f_delta
        not_converged = f_not_converged.logical_or_(not_converged); del f_not_converged
    if (p_atol is not None) or (p_rtol is not None):
        p_delta = upd.abs()
        p_not_converged = f0.new_tensor(True, dtype=torch.bool)
        if p_rtol is not None:
            p_not_converged = (p_delta > p.abs().mul_(p_rtol)).any(1).logical_and_(p_not_converged)
        if p_atol is not None:
            p_not_converged = (p_delta > p_atol).any(1).logical_and_(p_not_converged)
        del p_delta
        not_converged = p_not_converged.logical_or_(not_converged); del p_not_converged
    return not_converged
