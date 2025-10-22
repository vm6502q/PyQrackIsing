from .maxcut_tfim_streaming import maxcut_tfim_streaming
from .maxcut_tfim_util import compute_cut_streaming, compute_energy_streaming, get_cut, get_cut_base, int_to_bitstring, opencl_context
import itertools
import networkx as nx
import numpy as np
from numba import njit, prange
import random


epsilon = opencl_context.epsilon


@njit
def bootstrap_worker(theta, G_func, nodes, indices, is_spin_glass, n):
    local_theta = theta.copy()
    for i in indices:
        local_theta[i] = not local_theta[i]
    energy = compute_energy_streaming(local_theta, G_func, nodes, n) if is_spin_glass else compute_cut_streaming(local_theta, G_func, nodes, n)

    return energy


@njit(parallel=True)
def bootstrap(best_theta, G_func, nodes, indices_array, k, max_energy, dtype, is_spin_glass, n):
    energies = np.empty(n, dtype=dtype)
    for i in prange(n):
        j = i * k
        energies[i] = bootstrap_worker(best_theta, G_func, nodes, indices_array[j : j + k], is_spin_glass, n)

    energy = energies.min()
    if energy > max_energy:
        atol = dtype(epsilon)
        rtol = dtype(0)
        index_match = np.random.choice(np.where(np.isclose(energies, energy, atol=atol, rtol=rtol))[0])
        indices = indices_array[(index_match * k) : ((index_match + 1) * k)]
        max_energy = energies[index_match]
        for i in indices:
            best_theta[i] = not best_theta[i]

    return max_energy


def log_intermediate(theta, G_func, nodes, max_energy, is_spin_glass, n):
    bitstring, l, r = get_cut(theta, nodes, n)
    if is_spin_glass:
        cut_value = compute_cut_streaming(theta, G_func, nodes, n)
        min_energy = -max_energy
    else:
        cut_value = max_energy
        min_energy = compute_energy_streaming(theta, G_func, nodes, n)
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

    max_energy = compute_energy_streaming(best_theta, G_func, nodes, n_qubits) if is_spin_glass else compute_cut_streaming(best_theta, G_func, nodes, n_qubits)

    if is_log:
        log_intermediate(best_theta, G_func, nodes, max_energy, is_spin_glass, n_qubits)

    combos_list = []
    reheat_theta = best_theta.copy()
    reheat_max_energy = max_energy
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

                energy = bootstrap(best_theta, G_func, nodes, combos, k, max_energy, dtype, is_spin_glass, n_qubits)

                if energy > reheat_max_energy:
                    reheat_max_energy = energy
                    improved = True
                    if correction_quality < (k + 1):
                        correction_quality = k + 1

                    if is_log:
                        log_intermediate(reheat_theta, G_func, nodes, max_energy, is_spin_glass, n_qubits)

                    break

                k = k + 1

        if max_energy > reheat_max_energy:
            reheat_theta = best_theta.copy()
        else:
            best_theta = reheat_theta.copy()
            max_energy = reheat_max_energy

        if reheat_round < reheat_tries:
            num_to_flip = int(np.round(np.log2(n_qubits)))
            bits_to_flip = random.sample(range(n_qubits), num_to_flip)
            for bit in bits_to_flip:
                reheat_theta[bit] = not reheat_theta[bit]
            reheat_max_energy = compute_energy_streaming(reheat_theta, G_func, nodes, n_qubits) if is_spin_glass else compute_cut_streaming(reheat_theta, G_func, nodes, n_qubits)

    bitstring, l, r = get_cut(best_theta, nodes, n_qubits)
    if is_spin_glass:
        cut_value = compute_cut_streaming(best_theta, G_func, nodes, n_qubits)
        min_energy = -max_energy
    else:
        cut_value = max_energy
        min_energy = compute_energy_streaming(best_theta, G_func, nodes, n_qubits)

    return bitstring, float(cut_value), (l, r), float(min_energy)
