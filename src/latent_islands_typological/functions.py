"""
Learning latent tree models from URIEL+ typological data.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from sklearn.metrics import mutual_info_score
from scipy.special import logsumexp
import pandas as pd
from tqdm import tqdm


def compute_mutual_information(data: np.ndarray) -> np.ndarray:
    """
    Compute mutual information matrix for all feature pairs.

    Args:
        data: Binary data matrix of shape (n_samples, n_features)

    Returns:
        Mutual information matrix of shape (n_features, n_features)
    """
    n_features = data.shape[1]
    mi_matrix = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                mi_matrix[i, j] = mutual_info_score(data[:, i], data[:, j])

    return mi_matrix


def compute_bic(model: Dict[str, Any], n_samples: int) -> float:
    """
    Calculate Bayesian Information Criterion (BIC) for model selection.

    Args:
        model: Dictionary containing 'log_likelihood' and 'n_params'
        n_samples: Number of samples in the dataset

    Returns:
        BIC value (lower is better)
    """
    log_likelihood = model["log_likelihood"]
    n_params = model["n_params"]
    penalty_scalar = 2

    bic = penalty_scalar * n_params ** 2 * np.log(n_samples) - 2 * log_likelihood
    return bic


def expectation_maximization(
        data: np.ndarray,
        n_states: int,
        n_iterations: int = 150,
        convergence_threshold: float = 1e-3,
        verbose: bool = False
) -> Dict[str, Any]:
    """
    Learn a Latent Class Model using Expectation-Maximization algorithm.

    Args:
        data: Binary data matrix of shape (n_samples, n_features)
        n_states: Number of latent states
        n_iterations: Maximum number of EM iterations
        convergence_threshold: Convergence threshold for log-likelihood change
        verbose: Whether to print progress messages

    Returns:
        Dictionary containing learned model parameters and metadata
    """
    n_samples, n_features = data.shape
    n_categories = 2  # Binary data

    # Initialize parameters
    prior_z = np.full(n_states, 1.0 / n_states)
    conditional_probs = [
        np.random.dirichlet(np.ones(n_categories), size=n_states)
        for _ in range(n_features)
    ]

    prev_log_likelihood = -np.inf

    if verbose:
        print(f"\nTraining with {n_states} latent states")

    for iteration in range(n_iterations):
        if verbose and iteration % 5 == 0:
            print(f"Iteration: {iteration}")

        # E-step: Compute posterior probabilities
        log_probs = np.tile(np.log(prior_z), (n_samples, 1))

        for feature_idx in range(n_features):
            feature_values = data[:, feature_idx]
            probs = conditional_probs[feature_idx][:, feature_values].T
            log_probs += np.log(probs + 1e-10)

        log_posterior = log_probs - logsumexp(log_probs, axis=1, keepdims=True)
        posterior = np.exp(log_posterior)

        # M-step: Update parameters
        prior_z = posterior.mean(axis=0)

        for feature_idx in range(n_features):
            feature_values = data[:, feature_idx]
            conditional_probs[feature_idx] = np.zeros((n_states, n_categories))

            for category in range(n_categories):
                mask = (feature_values == category)
                conditional_probs[feature_idx][:, category] = posterior[mask].sum(axis=0)
                conditional_probs[feature_idx][:, category] += 1e-10

            conditional_probs[feature_idx] /= conditional_probs[feature_idx].sum(
                axis=1, keepdims=True
            )

        # Check convergence
        current_log_likelihood = np.sum(logsumexp(log_probs, axis=1))
        if np.abs(prev_log_likelihood - current_log_likelihood) < convergence_threshold:
            break

        prev_log_likelihood = current_log_likelihood

    log_likelihood = prev_log_likelihood
    n_params = n_states * n_features

    model = {
        "P_z": prior_z,
        "P_x_given_z": conditional_probs,
        "log_likelihood": log_likelihood,
        "n_states": n_states,
        "n_params": n_params,
        "posterior": posterior,
        "increased_nodes": False,
        "bic": compute_bic({"log_likelihood": log_likelihood, "n_params": n_params}, n_samples),
        "n_vars": n_features
    }

    return model


def learn_latent_class_model(
        data: np.ndarray,
        max_states: int = 2,
        verbose: bool = False,
        n_restarts: int = 5
) -> Dict[str, Any]:
    """
    Learn optimal Latent Class Model using BIC for model selection.

    Args:
        data: Binary data matrix of shape (n_samples, n_features)
        max_states: Maximum number of latent states to consider
        verbose: Whether to print progress messages
        n_restarts: Number of random restarts for each state count

    Returns:
        Best model according to BIC
    """
    n_samples = data.shape[0]

    best_model = expectation_maximization(data, 2, verbose=verbose)
    best_bic = compute_bic(best_model, n_samples)

    for n_states in range(2, max_states + 1):
        for _ in range(n_restarts):
            candidate_model = expectation_maximization(data, n_states, verbose=verbose)
            candidate_bic = compute_bic(candidate_model, n_samples)

            if candidate_bic <= best_bic:
                best_model = candidate_model
                best_bic = candidate_bic

    return best_model


def compute_two_layer_bic(
        latent1: Dict[str, Any],
        latent2: Dict[str, Any],
        n_samples: int
) -> float:
    """
    Compute combined BIC for a two-layer latent tree model.

    Args:
        latent1: First latent model
        latent2: Second latent model
        n_samples: Number of samples in dataset

    Returns:
        Combined BIC value
    """
    bic1 = compute_bic(latent1, n_samples)
    bic2 = compute_bic(latent2, n_samples)
    return bic1 + bic2


def node_introduction(
        data: np.ndarray,
        model: Dict[str, Any],
        mutual_info: np.ndarray,
        top_k_pairs: int = 3
) -> Dict[str, Any]:
    """
    Introduce a new latent node by splitting variables into two groups.

    Args:
        data: Binary data matrix
        model: Current latent class model
        mutual_info: Mutual information matrix
        top_k_pairs: Number of high-MI pairs to consider

    Returns:
        Two-layer model with introduced node
    """
    n_samples, n_features = data.shape
    best_model = None
    best_bic = np.inf
    best_group1, best_group2 = None, None
    best_conditional = None

    all_vars = list(range(n_features))

    # Find top k pairs with highest mutual information
    pairs = [
        (mutual_info[i, j], i, j)
        for i in range(n_features)
        for j in range(i + 1, n_features)
    ]
    pairs = sorted(pairs, key=lambda x: x[0], reverse=True)[:top_k_pairs]

    for _, var_i, var_j in pairs:
        new_group = [var_i, var_j]
        main_group = [k for k in all_vars if k not in new_group]

        data_new = data[:, new_group]
        data_main = data[:, main_group]

        model_new = learn_latent_class_model(data_new)
        model_main = learn_latent_class_model(data_main)

        bic = compute_two_layer_bic(model_main, model_new, n_samples)

        if bic <= best_bic:
            best_model = (model_main, model_new)
            best_bic = bic
            best_group1, best_group2 = main_group, new_group

            # Compute conditional distribution P(Z2|Z1)
            joint = np.dot(model_main["posterior"].T, model_new["posterior"])
            best_conditional = joint / joint.sum(axis=1, keepdims=True)

    return {
        "latent": ["Z1", "Z2"],
        "group1": best_group1,
        "group2": best_group2,
        "Z1": best_model[0],
        "Z2": best_model[1],
        "P_Z2_given_Z1": best_conditional,
        "bic": best_bic,
        "increased_nodes": True,
        "structure": {
            "Z1": best_group1 + ["Z2"],
            "Z2": best_group2
        }
    }


def node_relocation(
        model: Dict[str, Any],
        group1: List[int],
        group2: List[int],
        data: np.ndarray
) -> Dict[str, Any]:
    """
    Relocate variables between groups to improve model fit.

    Args:
        model: Current two-layer model
        group1: Variables in first group
        group2: Variables in second group
        data: Binary data matrix

    Returns:
        Improved model after relocation
    """
    n_samples = data.shape[0]
    best_model = model
    best_bic = model["bic"]
    current_group1, current_group2 = group1, group2

    for element in group1:
        new_group1 = [x for x in current_group1 if x != element]
        new_group2 = current_group2 + [element]

        if not new_group1:
            continue

        data_g1 = data[:, new_group1]
        data_g2 = data[:, new_group2]

        model_g1 = learn_latent_class_model(data_g1)
        model_g2 = learn_latent_class_model(data_g2)

        joint = np.dot(model_g1["posterior"].T, model_g2["posterior"])
        conditional = joint / joint.sum(axis=1, keepdims=True)

        bic = compute_two_layer_bic(model_g1, model_g2, n_samples)

        if bic < best_bic:
            best_bic = bic
            best_model = {
                "latent": ["Z1", "Z2"],
                "group1": new_group1,
                "group2": new_group2,
                "Z1": model_g1,
                "Z2": model_g2,
                "P_Z2_given_Z1": conditional,
                "bic": best_bic,
                "increased_nodes": True,
                "structure": {
                    "Z1": new_group1,
                    "Z2": new_group2
                }
            }
            current_group1, current_group2 = new_group1, new_group2

    print(f"NR: G1={current_group1}, G2={current_group2}, "
          f"old_bic={model['bic']:.2f}, new_bic={best_bic:.2f}")

    return best_model


def learn_latent_tree_model(
        data: np.ndarray,
        mutual_info: np.ndarray,
        initial_model: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Learn a two-layer latent tree model.

    Args:
        data: Binary data matrix
        mutual_info: Mutual information matrix
        initial_model: Initial latent class model

    Returns:
        Best latent tree model
    """
    best_model = initial_model
    best_bic = initial_model["bic"]

    candidate = node_introduction(data, initial_model, mutual_info)

    if candidate["increased_nodes"]:
        candidate = node_relocation(
            candidate,
            candidate["group1"],
            candidate["group2"],
            data
        )

    print(f"(LTM) BIC comparison: NR={candidate['bic']:.2f}, LCA={best_bic:.2f}")

    if candidate["bic"] <= best_bic:
        best_model = candidate
        best_bic = candidate["bic"]

    return best_model


def bridged_islands(
        data: np.ndarray,
        delta: float = 0.001,
        mutual_info: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    """
    Learn the latent tree structure over the typological dataset.

    Args:
        data: Binary data matrix of shape (n_samples, n_features)
        delta: BIC threshold for determining when to split islands
        mutual_info: Pre-computed mutual information matrix (optional)

    Returns:
        List of island dictionaries containing group indices and models
    """
    n_samples, n_features = data.shape
    islands = []
    remaining_vars = set(range(n_features))

    if mutual_info is None:
        mutual_info = compute_mutual_information(data)

    with tqdm(total=len(remaining_vars), desc="Processing variables") as pbar:
        while remaining_vars:
            # Handle single variable case
            if len(remaining_vars) == 1:
                var_list = list(remaining_vars)
                model = learn_latent_class_model(data[:, var_list])
                islands.append({
                    "group_projected": list(range(len(var_list))),
                    "indices": var_list,
                    "model": model
                })
                remaining_vars.remove(var_list[0])
                pbar.update(1)
                break

            # Find pair with highest mutual information
            best_mi = -np.inf
            var_x, var_y = -1, -1

            for i in remaining_vars:
                for j in remaining_vars:
                    if i != j and mutual_info[i, j] > best_mi:
                        best_mi = mutual_info[i, j]
                        var_x, var_y = i, j

            current_island = {var_x, var_y}

            # Grow island by adding variables with high mutual information
            while True:
                remaining_vars -= current_island
                pbar.update(len(current_island))
                selected_var = -1
                highest_total_mi = -np.inf

                for candidate in remaining_vars:
                    total_mi = sum(mutual_info[s, candidate] for s in current_island)

                    if total_mi > highest_total_mi:
                        highest_total_mi = total_mi
                        selected_var = candidate

                # No more variables to add
                if selected_var == -1:
                    sorted_island = sorted(current_island)
                    model = learn_latent_class_model(data[:, sorted_island])
                    islands.append({
                        "group_projected": list(range(len(sorted_island))),
                        "indices": sorted_island,
                        "model": model
                    })
                    break

                current_island.add(selected_var)
                remaining_vars.remove(selected_var)

                # Check if island should be split
                sorted_island = sorted(current_island)
                island_data = data[:, sorted_island]

                lcm = learn_latent_class_model(island_data)
                island_mi = mutual_info[np.ix_(sorted_island, sorted_island)]
                ltm = learn_latent_tree_model(island_data, island_mi, lcm)

                if ltm["increased_nodes"]:
                    if ltm["bic"] < lcm["bic"] + delta:
                        group1 = ltm["group1"]
                        group1_original = [sorted_island[i] for i in group1]
                        group2 = ltm["group2"]
                        group2_original = [sorted_island[i] for i in group2]

                        # Keep larger group as island, return smaller to pool
                        if len(group2) > len(group1):
                            islands.append({
                                "group_projected": group2,
                                "indices": group2_original,
                                "model": ltm["Z2"]
                            })
                            remaining_vars |= set(group1_original)
                        else:
                            islands.append({
                                "group_projected": group1,
                                "indices": group1_original,
                                "model": ltm["Z1"]
                            })
                            remaining_vars |= set(group2_original)
                        break

    return islands


def compute_latent_probability(
        model: Dict[str, Any],
        observation: np.ndarray
) -> float:
    """
    Compute probability of latent state given observation.

    Args:
        model: Latent class model
        observation: Binary observation vector

    Returns:
        Probability of being in latent state 1
    """
    prior_z = model["P_z"]
    conditional_probs = model["P_x_given_z"]

    log_prob_z = np.log(prior_z)

    for idx, value in enumerate(observation):
        log_prob_z += np.log(conditional_probs[idx][:, value])

    log_prob_z -= logsumexp(log_prob_z)

    return np.exp(log_prob_z)[1]


def cosine_distance(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Compute cosine distance between two vectors.

    Args:
        vector1: First vector
        vector2: Second vector

    Returns:
        Cosine distance (1 - cosine similarity)
    """
    magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)

    if magnitude < 1e-10:
        return 0.0

    distance = 1 - np.dot(vector1, vector2) / magnitude
    return distance


def compute_total_distance(
        islands: List[Dict[str, Any]],
        observation1: np.ndarray,
        observation2: np.ndarray
) -> float:
    """
    Compute distance between two observations across all islands.

    Args:
        islands: List of island models
        observation1: First binary observation
        observation2: Second binary observation

    Returns:
        Total distance across all islands
    """
    obs1 = observation1.astype(int)
    obs2 = observation2.astype(int)

    probs1 = np.array([
        compute_latent_probability(island["model"], obs1[island["indices"]])
        for island in islands
    ])

    probs2 = np.array([
        compute_latent_probability(island["model"], obs2[island["indices"]])
        for island in islands
    ])

    return cosine_distance(probs1, probs2)


def compute_language_distance(
        data: np.ndarray,
        islands: List[Dict[str, Any]],
        languages: np.ndarray,
        lang1: str,
        lang2: str
) -> Optional[float]:
    """
    Compute distance between two languages by name.

    Args:
        data: Full dataset
        islands: List of island models
        languages: Array of language names
        lang1: First language name
        lang2: Second language name

    Returns:
        Distance between languages, or None if language not found
    """
    idx1 = np.where(languages == lang1)
    idx2 = np.where(languages == lang2)

    if len(idx1[0]) != 1 or len(idx2[0]) != 1:
        print(f"Invalid language name(s): {lang1}, {lang2}")
        return None

    return compute_total_distance(islands, data[idx1[0][0]], data[idx2[0][0]])


def compute_vector_distance(
        islands: List[Dict[str, Any]],
        vector1: np.ndarray,
        vector2: np.ndarray
) -> Optional[float]:
    """
    Compute distance between two feature vectors.

    Args:
        islands: List of island models
        vector1: First feature vector
        vector2: Second feature vector

    Returns:
        Distance between vectors, or None if incompatible lengths
    """
    if len(vector1) != len(vector2):
        print("Feature vectors have different lengths")
        return None

    return compute_total_distance(islands, vector1, vector2)


# Example usage
if __name__ == "__main__":
    data = pd.read_csv('../data/URIELPlus_Union_SoftImpute.csv').values

    print("Learning latent tree structure...")
    islands_result = bridged_islands(data=data)
    print(f"\nNumber of islands found: {len(islands_result)}")
    print("\nIsland details:")
    for i, island in enumerate(islands_result):
        print(f"Island {i + 1}: {island}")
