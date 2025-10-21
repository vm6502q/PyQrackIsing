from .maxcut_tfim_streaming import maxcut_tfim_streaming
from .maxcut_tfim_util import get_cut_base, opencl_context
from .spin_glass_solver_util import get_cut_from_bit_array, int_to_bitstring
import itertools
import networkx as nx
import numpy as np
from numba import njit, prange
import os
import random


epsilon = opencl_context.epsilon


@njit
def evaluate_cut_edges(theta_bits, G_func, nodes, n):
    l, r = get_cut_base(theta_bits, n)
    cut = 0
    for u in l:
        for v in r:
            cut += G_func(nodes[u], nodes[v])

    return cut


@njit
def compute_energy(theta_bits, G_func, nodes, n):
    energy = 0
    for u in range(n):
        u_bit = theta_bits[u]
        for v in range(u + 1, n):
            val = G_func(nodes[u], nodes[v])
            energy += (val if u_bit == theta_bits[v] else -val)

    return energy


@njit
def bootstrap_worker(theta, G_func, nodes, indices, is_spin_glass, n):
    local_theta = theta.copy()
    for i in indices:
        local_theta[i] = not local_theta[i]
    energy = compute_energy(local_theta, G_func, nodes, n) if is_spin_glass else -evaluate_cut_edges(local_theta, G_func, nodes, n)

    return energy


@njit(parallel=True)
def bootstrap(best_theta, G_func, nodes, indices_array, k, min_energy, dtype, is_spin_glass, n):
    energies = np.empty(n, dtype=dtype)
    for i in prange(n):
        j = i * k
        energies[i] = bootstrap_worker(best_theta, G_func, nodes, indices_array[j : j + k], is_spin_glass, n)

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


def log_intermediate(theta, G_func, nodes, min_energy, is_spin_glass, n):
    bitstring, l, r = get_cut_from_bit_array(theta, nodes)
    if is_spin_glass:
        cut_value = evaluate_cut_edges(theta, G_func, nodes, n)
    else:
        cut_value = -min_energy
        min_energy = compute_energy(theta, G_func, nodes, n)
    print(bitstring, float(cut_value), (l, r), float(min_energy))


def spin_glass_solver_streaming(
    G_func,
    nodes,
    quality=None,
    shots=None,
    best_guess=None,
    is_spin_glass=True,
    anneal_t=None,
    anneal_h=None,
    repulsion_base=None,
    min_order=2,
    max_order=2,
    is_log=False,
    reheat_tries=0
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
        bitstring, _, _ = maxcut_tfim_streaming(G_func, nodes, quality=quality, shots=shots, is_spin_glass=is_spin_glass, anneal_t=anneal_t, anneal_h=anneal_h, repulsion_base=repulsion_base)
    best_theta = np.array([b == "1" for b in list(bitstring)], dtype=np.bool_)

    if max_order is None:
        max_order = n_qubits

    min_energy = compute_energy(best_theta, G_func, nodes, n_qubits) if is_spin_glass else -evaluate_cut_edges(best_theta, G_func, nodes, n_qubits)

    if is_log:
        log_intermediate(best_theta, G_func, nodes, min_energy, is_spin_glass, n_qubits)

    combos_list = []
    reheat_theta = best_theta.copy()
    reheat_min_energy = min_energy
    for reheat_round in range(reheat_tries + 1):
        improved = True
        correction_quality = min_order
        while improved:
            improved = False
            k = 1
            while (k <= correction_quality) and (k <= max_order):
                if n_qubits < k:
                    break

                if len(combos_list) < k:
                    combos = np.array(list(
                        item for sublist in itertools.combinations(range(n_qubits), k) for item in sublist
                    ))
                    combos_list.append(combos)
                else:
                    combos = combos_list[k - 1]

                energy = bootstrap(best_theta, G_func, nodes, combos, k, min_energy, dtype, is_spin_glass, n_qubits)

                if energy < reheat_min_energy:
                    reheat_min_energy = energy
                    improved = True
                    if correction_quality < (k + 1):
                        correction_quality = k + 1

                    if is_log:
                        log_intermediate(reheat_theta, G_func, nodes, min_energy, is_spin_glass, n_qubits)

                    break

                k = k + 1

        if min_energy < reheat_min_energy:
            reheat_theta = best_theta.copy()
        else:
            best_theta = reheat_theta.copy()
            min_energy = reheat_min_energy

        if reheat_round < reheat_tries:
            num_to_flip = int(np.round(np.log2(n_qubits)))
            bits_to_flip = random.sample(list(range(n_qubits)), num_to_flip)
            for bit in bits_to_flip:
                reheat_theta[bit] = not reheat_theta[bit]
            reheat_min_energy = compute_energy(reheat_theta, G_func, nodes, n_qubits) if is_spin_glass else -evaluate_cut_edges(reheat_theta, G_func, nodes, n_qubits)

    bitstring, l, r = get_cut_from_bit_array(best_theta, nodes)
    if is_spin_glass:
        cut_value = evaluate_cut_edges(best_theta, G_func, nodes, n_qubits)
    else:
        cut_value = -min_energy
        min_energy = compute_energy(best_theta, G_func, nodes, n_qubits)

    return bitstring, float(cut_value), (l, r), float(min_energy)
