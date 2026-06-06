import networkx as nx
import numpy as np
from numba import njit, prange
import os

from .maxcut_tfim_util import (
    compute_cut,
    compute_energy,
    compute_cut_diff_between,
    get_cut,
    heuristic_threshold,
    init_thresholds,
    maxcut_hamming_cdf,
    opencl_context,
)

from .kawasaki_chain_sampler import sample_measurement_kawasaki_dense


epsilon = opencl_context.epsilon
dtype = opencl_context.dtype


@njit(parallel=True, cache=True)
def init_J_and_z(G_m):
    G_min = G_m.min()
    n_qubits = len(G_m)
    degrees = np.empty(n_qubits, dtype=np.uint32)
    J_eff = np.empty(n_qubits, dtype=np.float64)
    for n in prange(n_qubits):
        degree = 0
        J = 0.0
        for m in range(n_qubits):
            val = G_m[n, m] - G_min
            if val <= epsilon:
                continue
            degree += 1
            J += val
        if degree > 0:
            J = -J / degree
        degrees[n] = degree
        J_eff[n] = J

    return J_eff, degrees


@njit(cache=True)
def exact_maxcut(G):
    """Brute-force exact MAXCUT solver using Numba JIT."""
    n = G.shape[0]
    max_cut = -1.0
    best_mask = 0

    # Enumerate all 2^n possible bitstrings
    for mask in range(1 << n):
        cut = 0.0
        for i in range(n):
            bi = (mask >> i) & 1
            for j in range(i + 1, n):
                if bi != ((mask >> j) & 1):
                    cut += G[i, j]
        if cut > max_cut:
            max_cut = cut
            best_mask = mask

    # Reconstruct best bitstring
    best_bits = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        best_bits[i] = (best_mask >> i) & 1

    return best_bits, max_cut


@njit(cache=True)
def exact_spin_glass(G):
    """Brute-force exact spin-glass solver using Numba JIT."""
    n = G.shape[0]
    max_cut = -1.0
    best_mask = 0

    # Enumerate all 2^n possible bitstrings
    for mask in range(1 << n):
        cut = 0.0
        for i in range(n):
            bi = (mask >> i) & 1
            for j in range(i + 1, n):
                val = G[i, j]
                cut += val if bi == ((mask >> j) & 1) else -val
        if cut > max_cut:
            max_cut = cut
            best_mask = mask

    # Reconstruct best bitstring
    best_bits = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        best_bits[i] = (best_mask >> i) & 1

    return best_bits, max_cut


def maxcut_tfim(
    G,
    quality=None,
    shots=None,
    is_spin_glass=False,
    anneal_t=None,
    anneal_h=None,
    repulsion_base=None,
):
    nodes = None
    G_m = None
    if isinstance(G, nx.Graph):
        nodes = list(G.nodes())
        G_m = nx.to_numpy_array(G, weight="weight", nonedge=0.0, dtype=dtype)
    else:
        nodes = list(range(len(G)))
        G_m = G

    n_qubits = len(G_m)

    if n_qubits < heuristic_threshold:
        best_solution, best_value = exact_spin_glass(G_m) if is_spin_glass else exact_maxcut(G_m)
        bit_string, l, r = get_cut(best_solution, nodes, n_qubits)

        if best_value < 0.0:
            # Best cut is trivial partition, all/empty
            return "0" * n_qubits, 0.0, (nodes, [])

        return bit_string, best_value, (l, r)

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
        repulsion_base = 5.0

    J_eff, degrees = init_J_and_z(G_m)
    cum_prob = maxcut_hamming_cdf(init_thresholds(n_qubits), n_qubits, J_eff, degrees, quality, anneal_t, anneal_h)

    degrees = None
    J_eff = None

    thread_count = os.cpu_count() ** 2
    thinning = 1
    best_solution, best_value = sample_measurement_kawasaki_dense(G_m, shots, thread_count, cum_prob, repulsion_base, thinning, is_spin_glass)

    bit_string, l, r = get_cut(best_solution, nodes, n_qubits)

    if best_value < 0.0:
        # Best cut is trivial partition, all/empty
        return "0" * n_qubits, 0.0, (nodes, [])

    return bit_string, best_value, (l, r)
