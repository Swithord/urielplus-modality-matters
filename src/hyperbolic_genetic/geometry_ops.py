# src/geometry_ops.py
import torch


def mobius_add(u: torch.Tensor, v: torch.Tensor, eps: float = torch.finfo(torch.float64).eps) -> torch.Tensor:
    uv = (u * v).sum(-1, keepdim=True)
    uu = (u * u).sum(-1, keepdim=True)
    vv = (v * v).sum(-1, keepdim=True)
    denom = torch.clamp(1 + 2 * uv + uu * vv, min=eps)
    return ((1 + 2 * uv + vv) * u + (1 - uu) * v) / denom


def poincare_dist(U: torch.Tensor, V: torch.Tensor, eps: float = torch.finfo(torch.float64).eps) -> torch.Tensor:
    if U.dim() == 2 and V.dim() == 3:
        U = U.unsqueeze(1)
    uu = (U * U).sum(-1, keepdim=True)
    vv = (V * V).sum(-1, keepdim=True)
    diff = ((U - V) ** 2).sum(-1, keepdim=True)
    denom = torch.clamp((1 - uu) * (1 - vv), min=torch.finfo(torch.float64).eps)
    arg = 1 + 2 * diff / denom
    arg = torch.clamp(arg, min=1 + eps)
    return torch.acosh(arg).squeeze(-1)


def expmap_poincare(x: torch.Tensor, d: torch.Tensor, eps: float = torch.finfo(torch.float64).eps) -> torch.Tensor:
    n2 = (x * x).sum(-1, keepdim=True)
    lam = 2 / (1 - n2)
    nd = d.norm(dim=-1, keepdim=True)
    condition = nd > torch.finfo(torch.float64).eps
    val_gt_eps = torch.tanh(lam * nd / 2) / nd
    val_le_eps = lam / 2
    coef = torch.where(condition, val_gt_eps, val_le_eps)
    y = mobius_add(x, coef * d)
    r2 = (y * y).sum(-1, keepdim=True)
    r = torch.sqrt(torch.clamp(r2, min=torch.finfo(torch.float64).eps))
    return torch.where(r < 1, y, y * ((1-eps) / r))


def euclidean_dist(U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    if U.dim() == 2 and V.dim() == 3:
        U = U.unsqueeze(1)
    return (U - V).norm(dim=-1)


@torch.no_grad()
def sgd(param: torch.Tensor, grad: torch.Tensor, lr: float) -> None:
    if grad is None:
        return
    eps_check = torch.finfo(grad.dtype).eps
    touched_mask = grad.abs().sum(dim=-1) > eps_check
    if not touched_mask.any():
        return
    touched_indices = touched_mask.nonzero(as_tuple=False).squeeze(1)
    if touched_indices.numel() == 0:
        return
    param.index_add_(0, touched_indices, -lr * grad[touched_mask])


@torch.no_grad()
def rsgd(param: torch.Tensor, grad: torch.Tensor, lr: float) -> None:
    if grad is None:
        return
    eps_check = torch.finfo(grad.dtype).eps
    touched_mask = grad.abs().sum(dim=-1) > eps_check
    if not touched_mask.any():
        return
    touched_indices = touched_mask.nonzero(as_tuple=False).squeeze(1)
    if touched_indices.numel() == 0:
        return
    x = param[touched_indices]
    g = grad[touched_indices]
    n2 = (x * x).sum(-1, keepdim=True)
    step = -lr * ((1 - n2).square() / 4) * g
    param.index_copy_(0, touched_indices, expmap_poincare(x, step))


def lorentz_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2 and y.dim() == 3:
        x = x.unsqueeze(1)
    elif x.dim() == 3 and y.dim() == 2:
        y = y.unsqueeze(1)
    return -x[..., :1] * y[..., :1] + (x[..., 1:] * y[..., 1:]).sum(-1, keepdim=True)


def project_to_hyperboloid(x: torch.Tensor, max_spatial_norm: float = 1e10, eps: float = torch.finfo(torch.float64).eps) -> torch.Tensor:
    spatial_components = x[..., 1:]
    spatial_norms = torch.norm(spatial_components, p=2, dim=-1, keepdim=True)
    if spatial_norms.dtype != x.dtype:
        spatial_norms = spatial_norms.to(x.dtype)
    scale_factor = torch.clamp_max(max_spatial_norm / (spatial_norms + eps), 1.0)
    clamped_spatial_components = spatial_components * scale_factor
    space_sq = (clamped_spatial_components ** 2).sum(-1, keepdim=True)
    time_arg = torch.clamp(space_sq + 1.0, min=eps)
    time = torch.sqrt(time_arg)
    return torch.cat([time, clamped_spatial_components], dim=-1)


def hyperboloid_dist(u: torch.Tensor, v: torch.Tensor, eps_dist: float = torch.finfo(torch.float64).eps) -> torch.Tensor:
    if u.dim() == 2 and v.dim() == 3:
        u = u.unsqueeze(1)
    elif u.dim() == 3 and v.dim() == 2:
        v = v.unsqueeze(1)
    z = torch.clamp(-lorentz_product(u, v), min=1.0 + eps_dist)
    return torch.acosh(z).squeeze(-1)


def expmap_lorentz(
    x: torch.Tensor,
    u: torch.Tensor,
    eps: float = torch.finfo(torch.float64).eps,
    max_norm_val: float = 50.0
) -> torch.Tensor:
    tan_sq   = torch.clamp(lorentz_product(u, u), min=0.0)
    tan_norm = torch.sqrt(tan_sq + eps)
    scale    = torch.clamp_max(max_norm_val / (tan_norm + eps), 1.0)
    u        = u * scale
    tan_norm = tan_norm * scale 
    cosh     = torch.cosh(tan_norm)
    sinh_div = torch.where(
        tan_norm < eps,
        torch.ones_like(tan_norm),
        torch.sinh(tan_norm) / tan_norm
    )
    return cosh * x + sinh_div * u


def riemannian_gradient(x: torch.Tensor, g_euc: torch.Tensor) -> torch.Tensor:
    g_pseudo = g_euc.clone()
    g_pseudo[..., :1] *= -1.0
    return g_pseudo + lorentz_product(x, g_pseudo) * x


@torch.no_grad()
def hrsgd(param: torch.Tensor, grad: torch.Tensor, lr: float,
          max_rie_grad_norm: float = 5.0,
          max_spatial_norm: float = 1e10,
          eps: float = torch.finfo(torch.float64).eps) -> None:
    if grad is None:
        return
    touched_mask = grad.abs().sum(dim=-1) > eps
    if not touched_mask.any():
        return
    touched_indices = touched_mask.nonzero(as_tuple=False).squeeze(1)
    x = param[touched_indices]
    g_euc = grad[touched_indices]
    
    g_r = riemannian_gradient(x, g_euc)
    
    g_r_norm = torch.sqrt(torch.clamp(lorentz_product(g_r, g_r), min=0.0) + eps)
    clip_coef = torch.clamp_max(max_rie_grad_norm / (g_r_norm + eps), 1.0)
    g_r = g_r * clip_coef
    
    step = -lr * g_r
    
    new_points = expmap_lorentz(x, step, eps=eps)
    new_points = project_to_hyperboloid(new_points, max_spatial_norm=max_spatial_norm, eps=eps)

    is_invalid = torch.isnan(new_points) | torch.isinf(new_points)
    if torch.any(is_invalid):
        valid_idx = ~torch.any(is_invalid, dim=-1)
        if torch.any(valid_idx):
            param.index_copy_(0, touched_indices[valid_idx], new_points[valid_idx])
    else:
        param.index_copy_(0, touched_indices, new_points)
