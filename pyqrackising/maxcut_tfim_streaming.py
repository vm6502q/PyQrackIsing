import math
import networkx as nx
import numpy as np
import os
from numba import njit, prange

from .maxcut_tfim_util import get_cut, get_cut_base, maxcut_hamming_cdf, opencl_context, sample_mag, bit_pick, init_bit_pick


epsilon = opencl_context.epsilon
dtype = opencl_context.dtype


@njit
def update_repulsion_choice(G_func, nodes, max_edge, weights, n, used, node, repulsion_base):
    # Select node
    used[node] = True

    # Repulsion: penalize neighbors
    for nbr in range(n):
        if used[nbr]:
            continue
        weights[nbr] *= repulsion_base ** (-G_func(nodes[node], nodes[nbr]) / max_edge)


# Written by Elara (OpenAI custom GPT) and improved by Dan Strano
@njit
def local_repulsion_choice(G_func, nodes, max_edge, weights, tot_init_weight, repulsion_base, n, m):
    """
    Pick m nodes out of n with repulsion bias:
    - High-degree nodes are already less likely
    - After choosing a node, its neighbors' probabilities are further reduced
    adjacency_data, adjacency_rows: CSR-format sparse adjacency data
    weights: float array of shape (n,)
    """

    weights = weights.copy()
    used = np.zeros(n, dtype=np.bool_) # False = available, True = used

    # First bit:
    node = init_bit_pick(weights, tot_init_weight, n)

    if m == 1:
        used[node] = True
        return used

    update_repulsion_choice(G_func, nodes, max_edge, weights, n, used, node, repulsion_base)

    for _ in range(1, m - 1):
        node = bit_pick(weights, used, n)

        # Update answer and weights
        update_repulsion_choice(G_func, nodes, max_edge, weights, n, used, node, repulsion_base)

    node = bit_pick(weights, used, n)

    # Select node
    used[node] = True

    return used


@njit
def compute_energy(sample, G_func, nodes, n_qubits):
    energy = 0
    for u in range(n_qubits):
        u_bit = sample[u]
        for v in range(u + 1, n_qubits):
            val = G_func(nodes[u], nodes[v])
            energy += val if u_bit == sample[v] else -val

    return -energy


@njit
def compute_cut(sample, G_func, nodes, n_qubits):
    l, r = get_cut_base(sample, n_qubits)
    cut = 0
    for u in l:
        for v in r:
            cut += G_func(nodes[u], nodes[v])

    return cut


@njit(parallel=True)
def sample_measurement(G_func, nodes, max_edge, shots, thresholds, weights, n, repulsion_base, is_spin_glass):
    shots = max(1, shots >> 1)
    tot_init_weight = weights.sum()

    solutions = np.empty((shots, n), dtype=np.bool_)
    energies = np.empty(shots, dtype=dtype)

    best_solution = solutions[0]
    best_energy = -float("inf")

    improved = True
    while improved:
        improved = False
        if is_spin_glass:
            for s in prange(shots):
                # First dimension: Hamming weight
                m = sample_mag(thresholds)

                # Second dimension: permutation within Hamming weight
                sample = local_repulsion_choice(G_func, nodes, max_edge, weights, tot_init_weight, repulsion_base, n, m)
                solutions[s] = sample
                energies[s] = compute_energy(sample, G_func, nodes, n)
        else:
            for s in prange(shots):
                # First dimension: Hamming weight
                m = sample_mag(thresholds)

                # Second dimension: permutation within Hamming weight
                sample = local_repulsion_choice(G_func, nodes, max_edge, weights, tot_init_weight, repulsion_base, n, m)
                solutions[s] = sample
                energies[s] = compute_cut(sample, G_func, nodes, n)

        best_index = np.argmax(energies)
        energy = energies[best_index]
        if energy > best_energy:
            best_energy = energy
            best_solution = solutions[best_index].copy()
            improved = True

    if is_spin_glass:
        best_energy = compute_cut(best_solution, G_func, nodes, n)

    return best_solution, best_energy


@njit(parallel=True)
def init_J_and_z(G_func, nodes, G_min):
    n_qubits = len(nodes)
    degrees = np.empty(n_qubits, dtype=np.uint32)
    J_eff = np.empty(n_qubits, dtype=np.float64)
    G_max = -float("inf")
    for n in prange(n_qubits):
        degree = 0
        J = 0.0
        for m in range(n_qubits):
            val = G_func(nodes[n], nodes[m])
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

    return J_eff, degrees, G_max

@njit
def find_G_min(G_func, nodes, n_nodes):
    G_min = float("inf")
    for i in range(n_nodes):
        u = nodes[i]
        for j in range(i + 1, n_nodes):
            v = nodes[j]
            val = G_func(u, v)
            if val < G_min:
                G_min = val

    return G_min


@njit
def cpu_footer(shots, quality, n_qubits, G_min, G_func, nodes, is_spin_glass, anneal_t, anneal_h, repulsion_base):
    J_eff, degrees, max_edge = init_J_and_z(G_func, nodes, G_min)
    hamming_prob = maxcut_hamming_cdf(n_qubits, J_eff, degrees, quality, anneal_t, anneal_h)

    degrees = None
    J_eff = repulsion_base ** J_eff

    best_solution, best_value = sample_measurement(G_func, nodes, max_edge, shots, hamming_prob, J_eff, n_qubits, repulsion_base, is_spin_glass)

    bit_string, l, r = get_cut(best_solution, nodes, n_qubits)

    return bit_string, best_value, (l, r)


def maxcut_tfim_streaming(
    G_func,
    nodes,
    quality=None,
    shots=None,
    is_spin_glass=False,
    anneal_t=None,
    anneal_h=None,
    repulsion_base=None
):
    wgs = opencl_context.work_group_size
    n_qubits = len(nodes)

    if n_qubits < 3:
        if n_qubits == 0:
            return "", 0, ([], [])

        if n_qubits == 1:
            return "0", 0, (nodes, [])

        if n_qubits == 2:
            weight = G_func(nodes[0], nodes[1])
            if weight < 0.0:
                return "00", 0, (nodes, [])

            return "01", weight, ([nodes[0]], [nodes[1]])

    if quality is None:
        quality = 5

    if shots is None:
        # Number of measurement shots
        shots = n_qubits << quality

    if anneal_t is None:
        anneal_t = 8.0

    if anneal_h is None:
        anneal_h = 8.0

    if repulsion_base is None:
        repulsion_base = 8.0

    G_min = find_G_min(G_func, nodes, n_qubits)

    bit_string, best_value, partition = cpu_footer(shots, quality, n_qubits, G_min, G_func, nodes, is_spin_glass, anneal_t, anneal_h, repulsion_base)

    if best_value < 0.0:
        # Best cut is trivial partition, all/empty
        return '0' * n_qubits, 0.0, (nodes, [])

    return bit_string, best_value, partition
