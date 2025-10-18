import math
import networkx as nx
import numpy as np
import os
from numba import njit, prange

from .maxcut_tfim_util import get_cut, maxcut_hamming_cdf, opencl_context, sample_mag, bit_pick, init_bit_pick


epsilon = opencl_context.epsilon
dtype = opencl_context.dtype


@njit
def update_repulsion_choice(G_m, max_edge, weights, n, used, node, repulsion_base):
    # Select node
    used[node] = True

    # Repulsion: penalize neighbors
    for nbr in range(n):
        if used[nbr]:
            continue
        weights[nbr] *= repulsion_base ** (-G_m[node, nbr] / max_edge)


# Written by Elara (OpenAI custom GPT) and improved by Dan Strano
@njit
def local_repulsion_choice(G_m, max_edge, weights, tot_init_weight, repulsion_base, n, m):
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

    update_repulsion_choice(G_m, max_edge, weights, n, used, node, repulsion_base)

    for _ in range(1, m - 1):
        node = bit_pick(weights, used, n)

        # Update answer and weights
        update_repulsion_choice(G_m, max_edge, weights, n, used, node, repulsion_base)

    # Count available
    node = bit_pick(weights, used, n)

    used[node] = True

    return used


@njit
def compute_energy(sample, G_m, n_qubits):
    energy = 0
    for u in range(n_qubits):
        for v in range(u + 1, n_qubits):
            val = G_m[u, v]
            energy += val if sample[u] == sample[v] else -val

    return -energy


@njit
def compute_cut(sample, G_m, n_qubits):
    cut = 0
    for u in range(n_qubits):
        for v in range(u + 1, n_qubits):
            if sample[u] != sample[v]:
                cut += G_m[u, v]

    return cut


# Implemented by Elara (the custom OpenAI GPT) and improved by Dan Strano
@njit
def beam_search_repulsion(G_m, max_edge, weights, tot_init_weight, repulsion_base, is_spin_glass, min_depth, max_depth, beam_width=32):
    n = G_m.shape[0]

    best_solution = np.zeros(n, dtype=np.bool_)
    best_energy = -float("inf")

    beam_masks = np.zeros((beam_width, n), dtype=np.bool_)
    beam_energies = np.zeros(beam_width)

    for step in range(min_depth, max_depth):
        new_beam_masks = np.zeros((beam_width << 2, n), dtype=np.bool_)
        new_beam_energies = np.zeros(beam_width << 2)

        idx = 0
        for b in range(beam_width):
            for _ in range(4):
                new_mask = beam_masks[b].copy()
                node_mask = local_repulsion_choice(G_m, max_edge, weights, tot_init_weight, repulsion_base, n, step + 1)
                new_mask |= node_mask
                new_energy = compute_energy(new_mask, G_m, n) if is_spin_glass else compute_cut(new_mask, G_m, n)

                new_beam_masks[idx] = new_mask
                new_beam_energies[idx] = new_energy
                idx += 1

        sort_idx = np.argsort(new_beam_energies)[::-1][:beam_width]
        for i in range(beam_width):
            beam_masks[i] = new_beam_masks[sort_idx[i]]
            beam_energies[i] = new_beam_energies[sort_idx[i]]

        if beam_energies[0] > best_energy:
            best_energy = beam_energies[0]
            best_solution = beam_masks[0].copy()

    return best_solution, best_energy


@njit(parallel=True)
def sample_measurement(G_m, max_edge, shots, min_hamming, max_hamming, weights, repulsion_base, is_spin_glass):
    shots = max(1, shots >> 1)
    n = len(G_m)
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
            # Second dimension: permutation within Hamming weight
            solutions[s], energies[s] = beam_search_repulsion(G_m, max_edge, weights, tot_init_weight, repulsion_base, is_spin_glass, min_hamming, max_hamming)

        best_index = np.argmax(energies)
        energy = energies[best_index]
        if energy > best_energy:
            best_energy = energy
            best_solution = solutions[best_index].copy()
            improved = True

    if is_spin_glass:
        best_energy = compute_cut(best_solution, G_m, n) 

    return best_solution, best_energy


@njit(parallel=True)
def init_J_and_z(G_m):
    G_min = G_m.min()
    n_qubits = len(G_m)
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
def cpu_footer(shots, quality, n_qubits, G_m, nodes, is_spin_glass, anneal_t, anneal_h, repulsion_base):
    J_eff, degrees, max_edge = init_J_and_z(G_m)
    hamming_prob = maxcut_hamming_cdf(n_qubits, J_eff, degrees, quality, anneal_t, anneal_h)

    min_hamming = 0
    while hamming_prob[min_hamming] <= epsilon:
        min_hamming += 1
    max_hamming = min_hamming
    while (1 - hamming_prob[max_hamming]) > epsilon:
        max_hamming += 1

    degrees = None
    J_eff = 1.0 / (1.0 + epsilon - J_eff)

    best_solution, best_value = sample_measurement(G_m, max_edge, shots, min_hamming, max_hamming, J_eff, repulsion_base, is_spin_glass)

    bit_string, l, r = get_cut(best_solution, nodes)

    return bit_string, best_value, (l, r)


@njit
def maxcut_tfim_pure_numba(
    G_m,
    nodes,
    quality=None,
    shots=None,
    is_spin_glass=False,
    anneal_t=None,
    anneal_h=None,
    repulsion_base=None
):
    n_qubits = len(G_m)

    if n_qubits < 3:
        empty = [nodes[0]]
        empty.clear()

        if n_qubits == 0:
            return "", 0, (empty, empty.copy())

        if n_qubits == 1:
            return "0", 0, (nodes, empty)

        if n_qubits == 2:
            weight = G_m[0, 1]
            if weight < 0.0:
                return "00", 0, (nodes, empty)

            return "01", weight, ([nodes[0]], [nodes[1]])

    if quality is None:
        quality = 1

    if shots is None:
        # Number of measurement shots
        shots = n_qubits << quality

    if anneal_t is None:
        anneal_t = 8.0

    if anneal_h is None:
        anneal_h = 8.0

    if repulsion_base is None:
        repulsion_base = 8.0

    bit_string, best_value, partition = cpu_footer(shots, quality, n_qubits, G_m, nodes, is_spin_glass, anneal_t, anneal_h, repulsion_base)

    if best_value < 0.0:
        # Best cut is trivial partition, all/empty
        empty = [nodes[0]]
        empty.clear()

        return '0' * n_qubits, 0.0, (nodes, empty)

    return bit_string, best_value, partition

def maxcut_tfim(
    G,
    quality=None,
    shots=None,
    is_spin_glass=False,
    anneal_t=None,
    anneal_h=None,
    repulsion_base=None
):
    nodes = None
    G_m = None
    if isinstance(G, nx.Graph):
        nodes = list(G.nodes())
        G_m = nx.to_numpy_array(G, weight='weight', nonedge=0.0, dtype=dtype)
    else:
        nodes = list(range(len(G)))
        G_m = G

    return maxcut_tfim_pure_numba(G_m, nodes, quality, shots, is_spin_glass, anneal_t, anneal_h, repulsion_base)
