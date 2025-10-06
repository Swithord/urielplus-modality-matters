import math
import random
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from .model import GeometryEmbedding
from .geometry_ops import (
    euclidean_dist, hrsgd, hyperboloid_dist,
    poincare_dist, project_to_hyperboloid, rsgd,
    sgd
)


def train_embeddings(
    tree: Dict[int, List[int]],
    geometry: str = "poincare",
    dim: int = 2,
    lr: float = 0.01,
    epochs: int = 60,
    K: int = 10,
    batch_size: int = 128,
    seed: int = 2025,
    hyperboloid_max_rie_grad_norm: float = 15,
    hyperboloid_max_spatial_norm_points: float = 1e10,
    warmup_epochs = 200,
    device: str | torch.device = "cuda",
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int], Dict[int, int], torch.Tensor]:
    torch.set_default_dtype(torch.float64)
    current_dtype = torch.double

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    all_ids = sorted(set(tree) | {c for ch in tree.values() for c in ch})
    id2i = {n: i for i, n in enumerate(all_ids)}
    i2id = {i: n for n, i in id2i.items()}
    n = len(all_ids)

    parent, children = {}, {i: set() for i in range(n)}
    for p, chs in tree.items():
        for c in chs:
            parent[id2i[c]] = id2i[p]
            children[id2i[p]].add(id2i[c])

    depth = [-1] * n
    dq = deque([(i, 0) for i in range(n) if i not in parent])
    while dq:
        i, d = dq.popleft()
        depth[i] = d
        for ch in children[i]:
            dq.append((ch, d + 1))

    ancestors = {i: set() for i in range(n)}
    for i in sorted(range(n), key=lambda j: depth[j]):
        p = parent.get(i)
        if p is not None:
            ancestors[i] = ancestors[p].copy(); ancestors[i].add(p)

    descendants = {i: set(children[i]) for i in range(n)}
    for i in sorted(range(n), key=lambda j: -depth[j]):
        for ch in children[i]:
            descendants[i].update(descendants[ch])

    pos_pairs = [(d, a) for a in range(n) for d in descendants[a]]

    ancestor_mask = torch.zeros((n, n), dtype=torch.bool, device=device)
    for i, ancs in ancestors.items():
        if ancs: ancestor_mask[i, torch.tensor(list(ancs), device=device)] = True
    ancestor_mask.fill_diagonal_(True)

    geometries = {
        "poincare": {
            "dist": poincare_dist,
            "step_fn": rsgd,
            "optim": optim.SGD,
            "hyper": False
        },
        "euclidean": {
            "dist": euclidean_dist,
            "step_fn": sgd,
            "optim": optim.SGD,
            "hyper": False
        },
        "hyperboloid": {
            "dist": hyperboloid_dist,
            "step_fn": lambda p, g, lr: hrsgd(
                p, g, lr,
                max_rie_grad_norm=hyperboloid_max_rie_grad_norm,
                max_spatial_norm=hyperboloid_max_spatial_norm_points,
                eps=torch.finfo(current_dtype).eps
            ),
            "optim": optim.SGD,
            "hyper": True
        },
        "euclidean_adam": {
            "dist": euclidean_dist,
            "step_fn": None,
            "optim": optim.Adam,
            "hyper": False
        },
    }

    if geometry not in geometries:
        raise ValueError(f"Unknown geometry or strategy: {geometry}")
    cfg = geometries[geometry]
    dist, step_fn = cfg["dist"], cfg["step_fn"]
    optimizer_cls, is_hyperboloid = cfg["optim"], cfg["hyper"]

    model_dim = dim + 1 if is_hyperboloid else dim
    model = GeometryEmbedding(n, model_dim).to(device)
    optimizer = optimizer_cls(model.parameters(), lr=lr)


    scheduler_t_max = max(1, epochs - warmup_epochs)
    scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_t_max, eta_min=lr*0.01)

    if is_hyperboloid:
        with torch.no_grad():
            model.weight.data[..., 1:].uniform_(-1e-7, 1e-7)
            model.weight.data[:, 0] = 1.0
            model.weight.copy_(project_to_hyperboloid(
                model.weight.data,
                max_spatial_norm=hyperboloid_max_spatial_norm_points,
                eps=torch.finfo(current_dtype).eps
            ))

    num_pos = len(pos_pairs)

    for epoch in range(epochs):
        random.shuffle(pos_pairs)
        total_loss = 0.0

        current_lr = 0
        if epoch < warmup_epochs:
            if warmup_epochs > 0:
                warmup_ratio = epoch / warmup_epochs
                current_lr = lr * 0.5 * (1.0 - math.cos(math.pi * warmup_ratio))
            else:
                current_lr = lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            current_lr = optimizer.param_groups[0]['lr']

        for start in range(0, num_pos, batch_size):
            batch = pos_pairs[start:start + batch_size]
            if not batch: continue
            child_idx = torch.tensor([p[0] for p in batch], dtype=torch.long, device=device)
            anc_idx   = torch.tensor([p[1] for p in batch], dtype=torch.long, device=device)
            u, v = model(child_idx), model(anc_idx)

            B = child_idx.size(0)
            neg_idx = torch.randint(0, n, (B, K), device=device)
            invalid = ancestor_mask[child_idx.unsqueeze(1), neg_idx]
            while invalid.any():
                pos = torch.nonzero(invalid, as_tuple=False)
                neg_idx[pos[:,0], pos[:,1]] = torch.randint(0, n, (pos.size(0),), device=device)
                invalid = ancestor_mask[child_idx.unsqueeze(1), neg_idx]
            w = model(neg_idx)

            d_pos = dist(u, v)
            d_neg = dist(u, w)
            energies_pos = -d_pos
            energies_neg = -d_neg
            all_energies = torch.cat([energies_pos.unsqueeze(1), energies_neg], dim=1)
            log_denominator = torch.logsumexp(all_energies, dim=1)
            loss = (d_pos + log_denominator).sum()

            if geometry == "euclidean_adam":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                loss.backward()
                with torch.no_grad():
                    step_fn(model.weight, model.weight.grad, current_lr)
                model.zero_grad(set_to_none=True)

            total_loss += loss.item()
        if epoch >= warmup_epochs:
            scheduler.step()

        if verbose and epoch % 20 == 0:
            print(f"Epoch {epoch:3d} – loss {total_loss:.6f} – lr {current_lr:.6f}")

    return model.weight.detach().cpu(), torch.tensor(depth), id2i, i2id, ancestor_mask.cpu()
