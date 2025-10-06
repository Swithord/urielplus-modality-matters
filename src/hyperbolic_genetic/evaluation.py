# src/evaluation.py
import numpy as np
import torch
import networkx as nx
from .geometry_ops import (
    poincare_dist,
    euclidean_dist,
    hyperboloid_dist,
)
from scipy.stats import spearmanr
from typing import Dict, List, Tuple

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {_DEVICE}")


def _build_graph_from_adj_list(tree: Dict[int, List[int]]) -> nx.DiGraph:
    G = nx.DiGraph()
    for parent, children in tree.items():
        G.add_node(parent)
        if children:
            for child in children:
                G.add_edge(parent, child)
    return G


@torch.no_grad()
def reconstruction_metrics_edgewise(
    outputs,
    geometry: str = "poincare",
    eps: float = 1e-7,
):
    emb, _, id2i, _, ancestor_mask = outputs

    if isinstance(emb, np.ndarray):
        emb = torch.from_numpy(emb)
    else:                                       
        emb = emb.detach()
    emb = emb.to(_DEVICE)

    anc = (
        ancestor_mask.clone().to(_DEVICE)
        if isinstance(ancestor_mask, torch.Tensor)
        else torch.as_tensor(ancestor_mask, dtype=torch.bool, device=_DEVICE)
    )
    anc.fill_diagonal_(False)

    dist_fn = {
        "poincare": lambda u: poincare_dist(u, emb, eps=eps).squeeze(0),
        "euclidean": lambda u: euclidean_dist(u, emb).squeeze(0),
        "hyperboloid": lambda u: hyperboloid_dist(u, emb, eps_dist=eps).squeeze(0),
    }[geometry]

    all_ranks, APs = [], []
    n = emb.size(0)

    for u in range(n):
        pos_mask = anc[u]
        if not pos_mask.any():
            continue

        dist_all = dist_fn(emb[u].unsqueeze(0))
        dist_all[u] = torch.inf

        order = torch.argsort(dist_all)
        hits = pos_mask[order]

        pos_idx = torch.nonzero(hits, as_tuple=True)[0]
        if pos_idx.numel() == 0:
            continue

        precisions = torch.cumsum(hits, dim=0)[pos_idx].float() / (pos_idx + 1)
        APs.append(precisions.mean().item())

        d_pos = dist_all[pos_mask]
        d_neg = dist_all[~pos_mask]
        ranks = 1 + (d_neg[:, None] < d_pos[None, :]).sum(dim=0)
        all_ranks.extend(ranks.tolist())

    if not all_ranks:
        return 0.0, 0.0

    mean_rank_edge = float(np.mean(all_ranks))
    map_macro = float(np.mean(APs))
    return mean_rank_edge, map_macro


@torch.no_grad()
def generality_rank_correlation(
    emb: torch.Tensor,
    tree: Dict[int, List[int]],
    id2i: Dict[int, int],
    i2id: Dict[int, int],
    geometry: str = "poincare",
    eps: float = 1e-9,
) -> float:
    """
    Calculates Spearman correlation between embedding distance-from-origin and 
    normalized hierarchical rank. This version is geometrically correct and
    handles disconnected graphs (forests).
    """
    if isinstance(emb, np.ndarray):
        emb = torch.from_numpy(emb)
    else:                                       
        emb = emb.detach()
    emb = emb.to(_DEVICE)
    
    if geometry == "euclidean":
        origin = torch.zeros(1, emb.shape[-1], device=emb.device, dtype=emb.dtype)
        distances = euclidean_dist(origin, emb)
    elif geometry == "poincare":
        origin = torch.zeros(1, emb.shape[-1], device=emb.device, dtype=emb.dtype)
        distances = poincare_dist(origin, emb, eps=eps)
    elif geometry == "hyperboloid":
        origin = torch.zeros(1, emb.shape[-1], device=emb.device, dtype=emb.dtype)
        origin[0, 0] = 1.0
        distances = hyperboloid_dist(origin, emb, eps_dist=eps)
    else:
        raise ValueError(f"Unknown geometry: {geometry}")

    model_generality_scores = distances.cpu().numpy()

    G = _build_graph_from_adj_list(tree)
    roots = [node for node, degree in G.in_degree() if degree == 0]
    if not roots:
        raise ValueError("Could not find any root nodes (in-degree 0).")

    aligned_scores, normalized_ranks = [], []

    for root_node in roots:
        tree_nodes = {root_node} | nx.descendants(G, root_node)
        T = G.subgraph(tree_nodes)
        
        sp = nx.shortest_path_length(T, source=root_node)
        lp = {}
        nodes_in_reverse_topo = list(reversed(list(nx.topological_sort(T))))
        for node in nodes_in_reverse_topo:
            if T.out_degree(node) == 0:
                lp[node] = 0
            else:
                lp[node] = 1 + max(lp.get(succ, -1) for succ in T.successors(node))
        
        for node_id in tree_nodes:
            idx = id2i.get(node_id)
            if idx is None: continue

            s_path = sp.get(node_id, 0)
            l_path = lp.get(node_id, 0)
            rank = s_path / (s_path + l_path) if (s_path + l_path) > 0 else 0.0
            
            normalized_ranks.append(rank)
            aligned_scores.append(model_generality_scores[idx])

    if len(aligned_scores) < 2:
        print("Warning: Not enough valid nodes to compute generality correlation.")
        return 0.0

    correlation, _ = spearmanr(aligned_scores, normalized_ranks)
    return correlation if not np.isnan(correlation) else 0.0


@torch.no_grad()
def distance_correlation(
    emb: torch.Tensor,
    tree: Dict[int, List[int]],
    id2i: Dict[int, int],
    i2id: Dict[int, int],
    geometry: str = "poincare",
    n_samples: int = 50000,
    eps: float = 1e-7,
) -> float:
    """
    Calculates Spearman correlation between true graph distances and embedding distances.
    This version is geometrically correct and computationally efficient.
    """
    G = _build_graph_from_adj_list(tree)
    path_lengths = dict(nx.all_pairs_shortest_path_length(G.to_undirected()))

    if isinstance(emb, np.ndarray):
        emb = torch.from_numpy(emb)
    else:                                       
        emb = emb.detach()
    emb = emb.to(_DEVICE)
    
    n = emb.size(0)

    i_indices = torch.randint(0, n, (n_samples,), device=_DEVICE)
    j_indices = torch.randint(0, n, (n_samples,), device=_DEVICE)
    mask = i_indices == j_indices
    while torch.any(mask):
        j_indices[mask] = torch.randint(0, n, (mask.sum(),), device=_DEVICE)
        mask = i_indices == j_indices

    true_dists, valid_i, valid_j = [], [], []
    for i, j in zip(i_indices.tolist(), j_indices.tolist()):
        id1, id2 = i2id.get(i), i2id.get(j)
        if id1 is not None and id2 is not None and id2 in path_lengths.get(id1, {}):
            true_dists.append(path_lengths[id1][id2])
            valid_i.append(i)
            valid_j.append(j)

    if len(true_dists) < 2:
        print("Warning: Not enough connected pairs in sample for distance correlation.")
        return 0.0
    
    valid_i = torch.tensor(valid_i, device=_DEVICE)
    valid_j = torch.tensor(valid_j, device=_DEVICE)

    dist_fn_map = {
        "poincare": poincare_dist,
        "euclidean": euclidean_dist,
        "hyperboloid": hyperboloid_dist,
    }
    dist_fn = dist_fn_map[geometry]
    emb_dists = dist_fn(emb[valid_i], emb[valid_j]).cpu().numpy()

    correlation, _ = spearmanr(true_dists, emb_dists)
    return correlation if not np.isnan(correlation) else 0.0
