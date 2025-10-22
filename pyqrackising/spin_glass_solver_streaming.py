from .maxcut_tfim_streaming import maxcut_tfim_streaming
from .maxcut_tfim_util import compute_cut_streaming, compute_energy_streaming, get_cut, get_cut_base, gray_code_next, gray_mutation, int_to_bitstring, opencl_context
import itertools
import networkx as nx
import numpy as np
from numba import njit, prange
import os
import random


dtype = opencl_context.dtype


@njit(parallel=True)
def run_single_bit_flips(best_theta, is_spin_glass, G_func, nodes):
    n = len(best_theta)

    states = np.empty((n, n), dtype=np.bool_)
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
    best_state = states[best_index]

    return best_energy, best_state


@njit(parallel=True)
def run_double_bit_flips(best_theta, is_spin_glass, G_func, nodes):
    n = len(best_theta)

    states = np.empty((n, n), dtype=np.bool_)
    energies = np.zeros(n, dtype=dtype)

    if is_spin_glass:
        for i in prange(n):
            state = best_theta.copy()
            state[i] = not state[i]
            for j in range(n):
                if i == j:
                    continue
                state[j] = not state[j]
                energy = compute_energy_streaming(state, G_func, nodes, n)
                if energy > energies[i]:
                    states[i], energies[i] = state, energy
    else:
        for i in prange(n):
            state = best_theta.copy()
            state[i] = not state[i]
            for j in range(n):
                if i == j:
                    continue
                state[j] = not state[j]
                energy = compute_cut_streaming(state, G_func, nodes, n)
                if energy > energies[i]:
                    states[i], energies[i] = state, energy

    best_index = np.argmax(energies)
    best_energy = energies[best_index]
    best_state = states[best_index]

    return best_energy, best_state


@njit(parallel=True)
def run_gray_optimization(best_theta, iterators, gray_iterations, thread_count, is_spin_glass, G_func, nodes):
    n = len(best_theta)
    thread_iterations = gray_iterations // thread_count
    gray_iterations = (gray_iterations + thread_count - 1) // thread_count

    states = np.empty((thread_count, n), dtype=np.bool_)
    energies = np.zeros(thread_count, dtype=dtype)

    if is_spin_glass:
        for i in prange(thread_count):
            i_offset = i * thread_iterations
            iterator = iterators[i]
            for curr_idx in range(thread_iterations):
                state = gray_code_next(iterator, curr_idx)
                energy = compute_energy_streaming(state, G_func, nodes, n)
                if energy > energies[i]:
                    states[i], energies[i] = state, energy
    else:
        for i in prange(thread_count):
            i_offset = i * thread_iterations
            iterator = iterators[i]
            for curr_idx in range(thread_iterations):
                state = gray_code_next(iterator, curr_idx)
                energy = compute_cut_streaming(state, G_func, nodes, n)
                if energy > energies[i]:
                    states[i], energies[i] = state, energy

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
    gray_iterations=None
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
        bitstring, cut_value, _ = maxcut_tfim_streaming(G_func, nodes, quality=quality, shots=shots, is_spin_glass=is_spin_glass, anneal_t=anneal_t, anneal_h=anneal_h, repulsion_base=repulsion_base)
    best_theta = np.array([b == "1" for b in list(bitstring)], dtype=np.bool_)

    if gray_iterations is None:
        gray_iterations = n_qubits * os.cpu_count()

    max_energy = compute_energy_streaming(best_theta, G_func, nodes, n_qubits) if is_spin_glass else cut_value

    # Single bit flips
    improved = True
    while improved:
        improved = False
        energy, state = run_single_bit_flips(best_theta, is_spin_glass, G_func, nodes)
        if energy > max_energy:
            max_energy = energy
            best_theta = state
            improved = True
            break
        energy, state = run_double_bit_flips(best_theta, is_spin_glass, G_func, nodes)
        if energy > max_energy:
            max_energy = energy
            best_theta = state
            improved = True
            break

    # Gray code
    thread_count = os.cpu_count() ** 2
    iterators = np.array([gray_mutation(i, best_theta) for i in range(thread_count)])
    energy, state = run_gray_optimization(best_theta, iterators, gray_iterations, thread_count, is_spin_glass, G_func, nodes)
    if energy > max_energy:
        max_energy = energy
        best_theta = state

    bitstring, l, r = get_cut(best_theta, nodes, n_qubits)
    if is_spin_glass:
        cut_value = compute_cut_streaming(best_theta, G_func, nodes, n_qubits)
        min_energy = -max_energy
    else:
        cut_value = max_energy
        min_energy = compute_energy_streaming(best_theta, G_func, nodes, n_qubits)

    return bitstring, float(cut_value), (l, r), float(min_energy)
