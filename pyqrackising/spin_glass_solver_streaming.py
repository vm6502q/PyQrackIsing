from .maxcut_tfim_streaming import maxcut_tfim_streaming
from .maxcut_tfim_util import compute_cut_streaming, compute_energy_streaming, get_cut, get_cut_base, gray_code_next, gray_mutation, heuristic_threshold, int_to_bitstring, opencl_context
import networkx as nx
import numpy as np
from numba import njit, prange
import os


dtype = opencl_context.dtype


@njit(parallel=True)
def run_single_bit_flips(best_theta, is_spin_glass, G_func, nodes):
    n = len(best_theta)

    energies = np.empty(n, dtype=dtype)

    if is_spin_glass:
        for i in prange(n):
            state = best_theta.copy()
            state[i] = not state[i]
            energies[i] = compute_energy_streaming(state, G_func, nodes, n)
    else:
        for i in prange(n):
            state = best_theta.copy()
            state[i] = not state[i]
            energies[i] = compute_cut_streaming(state, G_func, nodes, n)

    best_index = np.argmax(energies)
    best_energy = energies[best_index]
    best_state = best_theta.copy()
    best_state[best_index] = not best_state[best_index]

    return best_energy, best_state


@njit(parallel=True)
def run_double_bit_flips(best_theta, is_spin_glass, G_func, nodes, thread_count):
    n = len(best_theta)
    combo_count = (n * (n - 1)) // 2
    thread_batch = (combo_count + thread_count - 1) // thread_count

    states = np.empty((thread_count, n), dtype=np.bool_)
    energies = np.empty(thread_count, dtype=dtype)

    if is_spin_glass:
        for t in prange(thread_count):
            s = int(t)
            while s < combo_count:
                c = int(s)
                i = int(0)
                lcv = int(n - 1)
                while c >= lcv:
                    c -= lcv
                    i += 1
                    lcv -= 1
                    if not lcv:
                        break
                j = int(c + i + 1)

                state = best_theta.copy()
                state[i] = not state[i]
                state[j] = not state[j]

                states[t], energies[t] = state, compute_energy_streaming(state, G_func, nodes, n)

                s += thread_batch
    else:
        for t in prange(thread_count):
            s = int(t)
            while s < combo_count:
                c = int(s)
                i = int(0)
                lcv = int(n - 1)
                while c >= lcv:
                    c -= lcv
                    i += 1
                    lcv -= 1
                    if not lcv:
                        break
                j = int(c + i + 1)

                state = best_theta.copy()
                state[i] = not state[i]
                state[j] = not state[j]

                states[t], energies[t] = state, compute_cut_streaming(state, G_func, nodes, n)

                s += thread_batch

    best_index = np.argmax(energies)
    best_energy = energies[best_index]
    best_state = states[best_index]

    return best_energy, best_state


@njit(parallel=True)
def pick_gray_seeds(best_theta, thread_count, gray_seed_multiple, G_func, nodes, n, is_spin_glass):
    blocks = (n + 63) // 64
    block_size = thread_count * gray_seed_multiple
    seed_count = block_size * blocks

    seeds = np.empty((seed_count, n), dtype=np.bool_)
    energies = np.empty(seed_count, dtype=dtype)

    if is_spin_glass:
        for s in prange(seed_count):
            i = s % block_size
            offset = (s // block_size) * 64
            seed = gray_mutation(i, best_theta, offset)
            energies[s] = compute_energy_streaming(seed, G_func, nodes, n)
            seeds[s] = seed
    else:
        for s in prange(seed_count):
            i = s % block_size
            offset = (s // block_size) * 64
            seed = gray_mutation(i, best_theta, offset)
            energies[s] = compute_cut_streaming(seed, G_func, nodes, n)
            seeds[s] = seed

    indices = np.argsort(energies)[::-1]

    best_seeds = np.empty((thread_count, n), dtype=np.bool_)
    best_energies = np.empty(thread_count, dtype=dtype)

    for i in prange(thread_count):
        idx = indices[i]
        best_seeds[i] = seeds[idx]
        best_energies[i] = energies[idx]

    return best_seeds, best_energies

@njit(parallel=True)
def run_gray_optimization(best_theta, iterators, energies, gray_iterations, thread_count, is_spin_glass, G_func, nodes):
    n = len(best_theta)
    thread_iterations = (gray_iterations + thread_count - 1) // thread_count
    blocks = (n + 63) // 64

    states = np.empty((thread_count, n), dtype=np.bool_)

    if is_spin_glass:
        for i in prange(thread_count):
            iterator = iterators[i].copy()
            best_energy, best_iterator = energies[i], iterator.copy()
            for curr_idx in range(thread_iterations):
                for block in range(blocks):
                    gray_code_next(iterator, curr_idx, block * 64)
                    energy = compute_energy_streaming(iterator, G_func, nodes, n)
                    if energy > best_energy:
                        best_iterator, best_energy = iterator.copy(), energy
                    else:
                        iterator = best_iterator.copy()
                if best_energy > energies[i]:
                    states[i], energies[i] = best_iterator.copy(), best_energy
    else:
        for i in prange(thread_count):
            iterator = iterators[i].copy()
            best_energy, best_iterator = energies[i], iterator.copy()
            for curr_idx in range(thread_iterations):
                for block in range(blocks):
                    gray_code_next(iterator, curr_idx, block * 64)
                    energy = compute_cut_streaming(iterator, G_func, nodes, n)
                    if energy > best_energy:
                        best_iterator, best_energy = iterator.copy(), energy
                    else:
                        iterator = best_iterator.copy()
                if best_energy > energies[i]:
                    states[i], energies[i] = best_iterator.copy(), best_energy

    best_index = np.argmax(energies)
    best_energy = energies[best_index]
    best_state = states[best_index]

    return best_energy, best_state


def spin_glass_solver_streaming(
    G_func,
    nodes,
    quality=None,
    shots=None,
    best_guess=None,
    is_maxcut_gpu=True,
    is_spin_glass=True,
    anneal_t=None,
    anneal_h=None,
    repulsion_base=None,
    is_log=False,
    gray_iterations=None,
    gray_seed_multiple=None
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

    if n_qubits < heuristic_threshold:
        best_guess = None

    bitstring = ""
    if isinstance(best_guess, str):
        bitstring = best_guess
    elif isinstance(best_guess, int):
        bitstring = int_to_bitstring(best_guess, n_qubits)
    elif isinstance(best_guess, list):
        bitstring = "".join(["1" if b else "0" for b in best_guess])
    else:
        bitstring, cut_value, _ = maxcut_tfim_streaming(G_func, nodes, quality=quality, shots=shots, is_spin_glass=is_spin_glass, anneal_t=anneal_t, anneal_h=anneal_h, repulsion_base=repulsion_base)

    best_theta = np.array([b == "1" for b in list(bitstring)], dtype=np.bool_)
    max_energy = compute_energy(best_theta, G_m, n_qubits) if is_spin_glass else cut_value

    if n_qubits < heuristic_threshold:
        bitstring, l, r = get_cut(best_theta, nodes, n_qubits)
        if is_spin_glass:
            cut_value = compute_cut_streaming(best_theta, G_func, nodes, n_qubits)
            min_energy = -max_energy
        else:
            cut_value = max_energy
            min_energy = compute_energy_streaming(best_theta, G_func, nodes, n_qubits)

        return bitstring, float(cut_value), (l, r), float(min_energy)

    if gray_iterations is None:
        gray_iterations = n_qubits * n_qubits

    if gray_seed_multiple is None:
        gray_seed_multiple = os.cpu_count()

    thread_count = os.cpu_count() ** 2
    improved = True
    while improved:
        improved = False

        # Single bit flips with O(n^2)
        energy, state = run_single_bit_flips(best_theta, is_spin_glass, G_func, nodes)
        if energy > max_energy:
            max_energy = energy
            best_theta = state
            improved = True
            continue

        # Double bit flips with O(n^3)
        energy, state = run_double_bit_flips(best_theta, is_spin_glass, G_func, nodes, thread_count)
        if energy > max_energy:
            max_energy = energy
            best_theta = state
            improved = True
            continue

        # Gray code with default O(n^3)
        iterators, energies = pick_gray_seeds(best_theta, thread_count, gray_seed_multiple, G_func, nodes, n_qubits, is_spin_glass)
        energy, state = energies[0], iterators[0]
        if energy > max_energy:
            max_energy = energy
            best_theta = state
            improved = True
            continue

        energy, state = run_gray_optimization(best_theta, iterators, energies, gray_iterations, thread_count, is_spin_glass, G_func, nodes)
        if energy > max_energy:
            max_energy = energy
            best_theta = state
            improved = True
            continue

        # Post-reheat phase
        reheat_theta = state

        # Single bit flips with O(n^2)
        energy, state = run_single_bit_flips(reheat_theta, is_spin_glass, G_func, nodes)
        if energy > max_energy:
            max_energy = energy
            best_theta = state
            improved = True
            continue

        # Double bit flips with O(n^3)
        energy, state = run_double_bit_flips(reheat_theta, is_spin_glass, G_func, nodes, thread_count)
        if energy > max_energy:
            max_energy = energy
            best_theta = state
            improved = True

    bitstring, l, r = get_cut(best_theta, nodes, n_qubits)
    if is_spin_glass:
        cut_value = compute_cut_streaming(best_theta, G_func, nodes, n_qubits)
        min_energy = -max_energy
    else:
        cut_value = max_energy
        min_energy = compute_energy_streaming(best_theta, G_func, nodes, n_qubits)

    return bitstring, float(cut_value), (l, r), float(min_energy)
