import math
import networkx as nx
import numpy as np
import os
from numba import njit, prange

from .maxcut_tfim_util import binary_search, get_cut, get_cut_base, maxcut_hamming_cdf, opencl_context, sample_mag, bit_pick, init_bit_pick, to_scipy_sparse_upper_triangular


epsilon = opencl_context.epsilon
dtype = opencl_context.dtype


@njit
def update_repulsion_choice(G_cols, G_data, G_rows, max_edge, weights, n, used, node, repulsion_base):
    # Select node
    used[node] = True

    # Repulsion: penalize neighbors
    for j in range(G_rows[node], G_rows[node + 1]):
        nbr = G_cols[j]
        if used[nbr]:
            continue
        weights[nbr] *= repulsion_base ** (-G_data[j] / max_edge)

    for nbr in range(node):
        if used[nbr]:
            continue
        start = G_rows[nbr]
        end = G_rows[nbr + 1]
        j = binary_search(G_cols[start:end], node) + start
        if j < end:
            weights[nbr] *= repulsion_base ** (-G_data[j] / max_edge)


# Written by Elara (OpenAI custom GPT) and improved by Dan Strano
@njit
def local_repulsion_choice(G_cols, G_data, G_rows, max_edge, weights, tot_init_weight, repulsion_base, n, m):
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
    node = init_bit_pick(weights, tot_init_weight, n)

    if m == 1:
        used[node] = True
        return used

    update_repulsion_choice(G_cols, G_data, G_rows, max_edge, weights, n, used, node, repulsion_base)

    for _ in range(1, m - 1):
        node = bit_pick(weights, used, n)

        # Update answer and weights
        update_repulsion_choice(G_cols, G_data, G_rows, max_edge, weights, n, used, node, repulsion_base)

    node = bit_pick(weights, used, n)

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

    return -energy


@njit
def compute_cut(sample, G_data, G_rows, G_cols, n_qubits):
    l, _ = get_cut_base(sample)
    cut = 0
    for u in l:
        for col in range(G_rows[u], G_rows[u + 1]):
            v = G_cols[col]
            if sample[u] != sample[v]:
                cut += G_data[col]

    return cut


@njit(parallel=True)
def sample_measurement(max_edge, G_data, G_rows, G_cols, shots, thresholds, weights, repulsion_base, is_spin_glass):
    shots = max(1, shots >> 1)
    n = G_rows.shape[0] - 1
    tot_init_weight = weights.sum()

    solutions = np.empty((shots, n), dtype=np.bool_)
    energies = np.empty(shots, dtype=dtype)

    best_solution = solutions[0]
    best_energy = -float("inf")

    improved = True
    while improved:
        improved = False
        for s in prange(shots):
            # First dimension: Hamming weight
            m = sample_mag(thresholds)

            # Second dimension: permutation within Hamming weight
            sample = local_repulsion_choice(G_cols, G_data, G_rows, max_edge, weights, tot_init_weight, repulsion_base, n, m)
            solutions[s] = sample
            energies[s] = compute_energy(sample, G_data, G_rows, G_cols, n) if is_spin_glass else compute_cut(sample, G_data, G_rows, G_cols, n)

        best_index = np.argmax(energies)
        energy = energies[best_index]
        if energy > best_energy:
            best_energy = energy
            best_solution = solutions[best_index].copy()
            improved = True

    if is_spin_glass:
        best_energy = compute_cut(best_solution, G_data, G_rows, G_cols, n)

    return best_solution, best_energy


def init_J_and_z(G_m):
    G_min = G_m.min()
    n_qubits = G_m.shape[0]
    degrees = np.empty(n_qubits, dtype=np.uint32)
    J_eff = np.empty(n_qubits, dtype=np.float64)
    G_max = -float("inf")
    for n in prange(n_qubits):
        degree = 0
        J = 0.0
        for m in range(n_qubits):
            val = G_m[n, m]
            if val > G_max:
                G_max = val
            val -= G_min
            if val != 0.0:
                degree += 1
                J += val
                val = abs(val)
        J = -J / degree if degree > 0 else 0
        degrees[n] = degree
        J_eff[n] = J

    G_min = abs(G_min)
    G_max = abs(G_max)
    if G_min > G_max:
        G_max = G_min

    return J_eff, degrees, G_max


@njit
def cpu_footer(J_eff, degrees, shots, quality, n_qubits, G_max, G_data, G_rows, G_col, nodes, is_spin_glass, anneal_t, anneal_h, repulsion_base):
    hamming_prob = maxcut_hamming_cdf(n_qubits, J_eff, degrees, quality, anneal_t, anneal_h)

    degrees = None
    J_eff = 1.0 / (1.0 + epsilon - J_eff)

    best_solution, best_value = sample_measurement(G_max, G_data, G_rows, G_col, shots, hamming_prob, J_eff, repulsion_base, is_spin_glass)

    bit_string, l, r = get_cut(best_solution, nodes)

    return bit_string, best_value, (l, r)


def maxcut_tfim_sparse(
    G,
    quality=None,
    shots=None,
    is_spin_glass=False,
    anneal_t=None,
    anneal_h=None,
    repulsion_base=None
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
        quality = 6

    if shots is None:
        # Number of measurement shots
        shots = n_qubits << quality

    if anneal_t is None:
        anneal_t = 8.0

    if anneal_h is None:
        anneal_h = 8.0

    if repulsion_base is None:
        repulsion_base = 8.0

    J_eff, degrees, G_max = init_J_and_z(G_m)

    bit_string, best_value, partition = cpu_footer(J_eff, degrees, shots, quality, n_qubits, G_max, G_m.data, G_m.indptr, G_m.indices, nodes, is_spin_glass, anneal_t, anneal_h, repulsion_base)

    if best_value < 0.0:
        # Best cut is trivial partition, all/empty
        return '0' * n_qubits, 0.0, (nodes, [])

    return bit_string, best_value, partition
