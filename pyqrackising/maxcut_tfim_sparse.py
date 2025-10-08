import math
import networkx as nx
import numpy as np
import os
from numba import njit, prange

from .maxcut_tfim_util import binary_search, get_cut, maxcut_hamming_cdf, opencl_context, to_scipy_sparse_upper_triangular


epsilon = opencl_context.epsilon


@njit
def update_repulsion_choice(G_cols, G_data, G_rows, max_edge, weights, n, used, node):
    # Select node
    used[node] = True

    # Repulsion: penalize neighbors
    for j in range(G_rows[node], G_rows[node + 1]):
        nbr = G_cols[j]
        if used[nbr]:
            continue
        weights[nbr] *= max(epsilon, 1 - G_data[j] / max_edge)

    for nbr in range(node):
        if used[nbr]:
            continue
        start = G_rows[nbr]
        end = G_rows[nbr + 1]
        j = binary_search(G_cols[start:end], node) + start
        if j < end:
            weights[nbr] *= max(epsilon, 1 - G_data[j] / max_edge)


# Written by Elara (OpenAI custom GPT) and improved by Dan Strano
@njit
def local_repulsion_choice(G_cols, G_data, G_rows, max_edge, weights, tot_init_weight, n, m):
    """
    Pick m nodes out of n with repulsion bias:
    - High-degree nodes are already less likely
    - After choosing a node, its neighbors' probabilities are further reduced
    adjacency_data, adjacency_rows: CSR-format sparse adjacency data
    weights: float32 array of shape (n,)
    """

    weights = weights.copy()
    used = np.zeros(n, dtype=np.bool_) # False = available, True = used

    # First bit:
    r = np.random.rand()
    cum = 0.0
    node = 0
    for i in range(n):
        cum += weights[i]
        if (tot_init_weight * r) < cum:
            node = i
            break

    if m == 1:
        used[node] = True
        return used

    update_repulsion_choice(G_cols, G_data, G_rows, max_edge, weights, n, used, node)

    for _ in range(1, m - 1):
        # Count available
        total_w = 0.0
        for i in range(n):
            if used[i]:
                continue
            total_w += weights[i]

        # Normalize & sample
        r = np.random.rand()
        cum = 0.0
        node = -1
        for i in range(n):
            if used[i]:
                continue
            cum += weights[i]
            if (total_w * r) < cum:
                node = i
                break

        if node == -1:
            node = 0
            while used[node]:
                node += 1

        # Update answer and weights
        update_repulsion_choice(G_cols, G_data, G_rows, max_edge, weights, n, used, node)

    # Count available
    total_w = 0.0
    for i in range(n):
        if used[i]:
            continue
        total_w += weights[i]

    # Normalize & sample
    r = np.random.rand()
    cum = 0.0
    node = -1
    for i in range(n):
        if used[i]:
            continue
        cum += weights[i]
        if (total_w * r) < cum:
            node = i
            break

    if node == -1:
        node = 0
        while used[node]:
            node += 1

    used[node] = True

    return used


@njit
def compute_energy(sample, G_data, G_rows, G_cols, n_qubits):
    energy = 0
    for u in range(n_qubits):
        for col in range(G_rows[u], G_rows[u + 1]):
            v = G_cols[col]
            val = G_data[col]
            energy += val if sample[u] == sample[v] else -val

    return energy


@njit
def compute_cut(sample, G_data, G_rows, G_cols, n_qubits):
    cut = 0
    for u in range(n_qubits):
        for col in range(G_rows[u], G_rows[u + 1]):
            v = G_cols[col]
            if sample[u] != sample[v]:
                cut += G_data[col]

    return cut


@njit(parallel=True)
def sample_for_energy(G_data, G_rows, G_cols, shots, thresholds, weights):
    shots = max(1, shots >> 1)
    n = G_rows.shape[0] - 1
    max_edge = G_data.max()
    tot_init_weight = weights.sum()

    solutions = np.empty((shots, n), dtype=np.bool_)
    energies = np.empty(shots, dtype=np.float64)

    best_solution = solutions[0]
    best_energy = float("inf")

    improved = True
    while improved:
        improved = False
        for s in prange(shots):
            # First dimension: Hamming weight
            mag_prob = np.random.random()
            m = 0
            while thresholds[m] < mag_prob:
                m += 1
            m += 1

            # Second dimension: permutation within Hamming weight
            sample = local_repulsion_choice(G_cols, G_data, G_rows, max_edge, weights, tot_init_weight, n, m)
            solutions[s] = sample
            energies[s] = compute_energy(sample, G_data, G_rows, G_cols, n)

        best_index = np.argmin(energies)
        energy = energies[best_index]
        if energy < best_energy:
            best_energy = energy
            best_solution = solutions[best_index].copy()
            improved = True

    return best_solution, compute_cut(best_solution, G_data, G_rows, G_cols, n)


@njit(parallel=True)
def sample_for_cut(G_data, G_rows, G_cols, shots, thresholds, weights):
    shots = max(1, shots >> 1)
    n = G_rows.shape[0] - 1
    max_edge = G_data.max()
    tot_init_weight = weights.sum()

    solutions = np.empty((shots, n), dtype=np.bool_)
    cuts = np.empty(shots, dtype=np.float64)

    best_solution = solutions[0]
    best_cut = -float("inf")

    improved = True
    while improved:
        improved = False
        for s in prange(shots):
            # First dimension: Hamming weight
            mag_prob = np.random.random()
            m = 0
            while thresholds[m] < mag_prob:
                m += 1
            m += 1

            # Second dimension: permutation within Hamming weight
            sample = local_repulsion_choice(G_cols, G_data, G_rows, max_edge, weights, tot_init_weight, n, m)
            solutions[s] = sample
            cuts[s] = compute_cut(sample, G_data, G_rows, G_cols, n)

        best_index = np.argmax(cuts)
        cut = cuts[best_index]
        if cut > best_cut:
            best_cut = cut
            best_solution = solutions[best_index].copy()
            improved = True

    return best_solution, best_cut


@njit(parallel=True)
def init_J_and_z(G_data, G_rows, G_cols):
    n_qubits = G_rows.shape[0] - 1
    degrees = np.zeros(n_qubits, dtype=np.uint32)
    J_eff = np.zeros(n_qubits, dtype=np.float64)
    for r in prange(n_qubits):
        # Row sum
        start = G_rows[r]
        end = G_rows[r + 1]
        degree = end - start
        val = G_data[start:end].sum()

        degrees[r] += degree
        J_eff[r] += val

        # Column sum
        for idx in range(start, end):
            c = G_cols[idx]
            degrees[c] += 1
            J_eff[c] += G_data[idx]

    for r in prange(n_qubits):
        degree = degrees[r]
        J_eff[r] = -J_eff[r] / degree if degree > 0 else 0

    return J_eff, degrees


@njit
def cpu_footer(shots, quality, n_qubits, G_data, G_rows, G_col, nodes, is_spin_glass, anneal_t, anneal_h):
    J_eff, degrees = init_J_and_z(G_data, G_rows, G_col)
    hamming_prob = maxcut_hamming_cdf(n_qubits, J_eff, degrees, quality, anneal_t, anneal_h)

    degrees = None
    J_eff = 1.0 / (1.0 + epsilon - J_eff)

    if is_spin_glass:
        best_solution, best_value = sample_for_energy(G_data, G_rows, G_col, shots, hamming_prob, J_eff)
    else:
        best_solution, best_value = sample_for_cut(G_data, G_rows, G_col, shots, hamming_prob, J_eff)

    bit_string, l, r = get_cut(best_solution, nodes)

    return bit_string, best_value, (l, r)


def maxcut_tfim_sparse(
    G,
    quality=None,
    shots=None,
    is_spin_glass=False,
    anneal_t=None,
    anneal_h=None
):
    wgs = opencl_context.work_group_size
    nodes = None
    n_qubits = 0
    G_m = None
    if isinstance(G, nx.Graph):
        nodes = list(G.nodes())
        n_qubits = len(nodes)
        G_m = to_scipy_sparse_upper_triangular(G, nodes, n_qubits)
    else:
        n_qubits = G.shape[0]
        nodes = list(range(n_qubits))
        G_m = G

    if n_qubits < 3:
        if n_qubits == 0:
            return "", 0, ([], [])

        if n_qubits == 1:
            return "0", 0, (nodes, [])

        if n_qubits == 2:
            weight = G_m[0, 1]
            if weight < 0.0:
                return "00", 0, (nodes, [])

            return "01", weight, ([nodes[0]], [nodes[1]])

    if quality is None:
        quality = 8

    if shots is None:
        # Number of measurement shots
        shots = n_qubits << quality

    if anneal_t is None:
        anneal_t = 8.0

    if anneal_h is None:
        anneal_h = 8.0

    return cpu_footer(shots, quality, n_qubits, G_m.data, G_m.indptr, G_m.indices, nodes, is_spin_glass, anneal_t, anneal_h)
