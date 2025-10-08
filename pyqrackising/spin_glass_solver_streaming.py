from .maxcut_tfim_streaming import maxcut_tfim_streaming
from .maxcut_tfim_util import opencl_context
from .spin_glass_solver_util import get_cut_from_bit_array, int_to_bitstring
import itertools
import networkx as nx
import numpy as np
from numba import njit, prange
import os


epsilon = opencl_context.epsilon


@njit
def evaluate_cut_edges(theta_bits, G_func, nodes):
    n_qubits = len(nodes)
    cut = 0
    for u in range(n_qubits):
        for v in range(u + 1, n_qubits):
            if theta_bits[u] != theta_bits[v]:
                cut += G_func(nodes[u], nodes[v])

    return cut


@njit
def compute_energy(theta_bits, G_func, nodes):
    n_qubits = len(nodes)
    energy = 0
    for u in range(n_qubits):
        for v in range(u + 1, n_qubits):
            val = G_func(nodes[u], nodes[v])
            energy += (val if theta_bits[u] == theta_bits[v] else -val)

    return energy


@njit
def bootstrap_worker(theta, G_func, nodes, indices):
    local_theta = theta.copy()
    for i in indices:
        local_theta[i] = not local_theta[i]
    energy = compute_energy(local_theta, G_func, nodes)

    return energy


@njit(parallel=True)
def bootstrap(best_theta, G_func, nodes, indices_array, k, min_energy, dtype):
    n = len(indices_array) // k
    energies = np.empty(n, dtype=dtype)
    for i in prange(n):
        j = i * k
        energies[i] = bootstrap_worker(best_theta, G_func, nodes, indices_array[j : j + k])

    energy = energies.min()
    if energy < min_energy:
        atol = dtype(epsilon)
        rtol = dtype(0)
        index_match = np.random.choice(np.where(np.isclose(energies, energy, atol=atol, rtol=rtol))[0])
        indices = indices_array[(index_match * k) : ((index_match + 1) * k)]
        min_energy = energies[index_match]
        for i in indices:
            best_theta[i] = not best_theta[i]

    return min_energy


def spin_glass_solver_streaming(
    G_func,
    nodes,
    quality=None,
    shots=None,
    best_guess=None,
    is_spin_glass=True,
    anneal_t=None,
    anneal_h=None
):
    dtype = opencl_context.dtype
    n_qubits = len(nodes)

    if n_qubits < 3:
        if n_qubits == 0:
            return "", 0, ([], []), 0

        if n_qubits == 1:
            return "0", 0, (nodes, []), 0

        if n_qubits == 2:
            weight = G_func(nodes[0], nodes[1])
            if weight < 0.0:
                return "00", 0, (nodes, []), weight

            return "01", weight, ([nodes[0]], [nodes[1]]), -weight

    bitstring = ""
    if isinstance(best_guess, str):
        bitstring = best_guess
    elif isinstance(best_guess, int):
        bitstring = int_to_bitstring(best_guess, n_qubits)
    elif isinstance(best_guess, list):
        bitstring = "".join(["1" if b else "0" for b in best_guess])
    else:
        bitstring, _, _ = maxcut_tfim_streaming(G_func, nodes, quality=quality, shots=shots, is_spin_glass=is_spin_glass, anneal_t=anneal_t, anneal_h=anneal_h)
    best_theta = np.array([b == "1" for b in list(bitstring)], dtype=np.bool_)

    min_energy = compute_energy(best_theta, G_func, nodes)
    improved = True
    correction_quality = 1
    combos_list = []
    while improved:
        improved = False
        k = 1
        while k <= correction_quality:
            if n_qubits < k:
                break

            combos = []
            if len(combos_list) < k:
                combos = np.array(list(
                    item for sublist in itertools.combinations(range(n_qubits), k) for item in sublist
                ))
                combos_list.append(combos)
            else:
                combos = combos_list[k - 1]

            energy = bootstrap(best_theta, G_func, nodes, combos, k, min_energy, dtype)

            if energy < min_energy:
                min_energy = energy
                improved = True
                if correction_quality < (k + 1):
                    correction_quality = k + 1
                break

            k = k + 1

    bitstring, l, r = get_cut_from_bit_array(best_theta, nodes)
    cut_value = evaluate_cut_edges(best_theta, G_func, nodes)

    return bitstring, float(cut_value), (l, r), float(min_energy)
