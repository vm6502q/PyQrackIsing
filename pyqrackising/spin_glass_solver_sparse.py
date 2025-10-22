from .maxcut_tfim_sparse import maxcut_tfim_sparse
from .maxcut_tfim_util import compute_cut_sparse, compute_energy_sparse, get_cut, gray_code_next, gray_mutation, int_to_bitstring, opencl_context, setup_opencl, to_scipy_sparse_upper_triangular
import itertools
import networkx as nx
import numpy as np
from numba import njit, prange
import os
import random
from scipy.sparse import lil_matrix, csr_matrix


IS_OPENCL_AVAILABLE = True
try:
    import pyopencl as cl
except ImportError:
    IS_OPENCL_AVAILABLE = False


dtype = opencl_context.dtype


@njit(parallel=True)
def run_single_bit_flips(best_theta, is_spin_glass, G_data, G_rows, G_cols):
    n = len(best_theta)

    energies = np.empty(n, dtype=dtype)

    if is_spin_glass:
        for i in prange(n):
            state = best_theta.copy()
            state[i] = not state[i]
            energies[i] = compute_energy_sparse(state, G_data, G_rows, G_cols, n)
    else:
        for i in prange(n):
            state = best_theta.copy()
            state[i] = not state[i]
            energies[i] = compute_cut_sparse(state, G_data, G_rows, G_cols, n)

    best_index = np.argmax(energies)
    best_energy = energies[best_index]
    best_state = best_theta.copy()
    best_state[best_index] = not best_state[best_index]

    return best_energy, best_state


@njit(parallel=True)
def run_double_bit_flips(best_theta, is_spin_glass, G_data, G_rows, G_cols):
    n = len(best_theta)

    states = np.empty((n, n), dtype=np.bool_)
    energies = np.full(n, np.finfo(dtype).min, dtype=dtype)

    if is_spin_glass:
        for i in prange(n):
            state = best_theta.copy()
            state[i] = not state[i]
            for j in range(n):
                if i == j:
                    continue
                state[j] = not state[j]
                energy = compute_energy_sparse(state, G_data, G_rows, G_cols, n)
                if energy > energies[i]:
                    states[i], energies[i] = state.copy(), energy
                state[j] = not state[j]
    else:
        for i in prange(n):
            state = best_theta.copy()
            state[i] = not state[i]
            for j in range(n):
                if i == j:
                    continue
                state[j] = not state[j]
                energy = compute_cut_sparse(state, G_data, G_rows, G_cols, n)
                if energy > energies[i]:
                    states[i], energies[i] = state.copy(), energy
                state[j] = not state[j]

    best_index = np.argmax(energies)
    best_energy = energies[best_index]
    best_state = states[best_index]

    return best_energy, best_state


@njit(parallel=True)
def pick_gray_seeds(best_theta, thread_count, gray_seed_multiple, G_data, G_rows, G_cols, n, is_spin_glass):
    seed_count = thread_count * gray_seed_multiple

    seeds = np.empty((seed_count, n), dtype=np.bool_)
    energies = np.empty(seed_count, dtype=dtype)

    if is_spin_glass:
        for i in prange(seed_count):
            seed = gray_mutation(i, best_theta)
            energies[i] = compute_energy_sparse(seed, G_data, G_rows, G_cols, n)
            seeds[i] = seed
    else:
        for i in prange(seed_count):
            seed = gray_mutation(i, best_theta)
            energies[i] = compute_cut_sparse(seed, G_data, G_rows, G_cols, n)
            seeds[i] = seed

    indices = np.argsort(energies)[::-1]

    best_seeds = np.empty((thread_count, n), dtype=np.bool_)
    best_energies = np.empty(thread_count, dtype=dtype)

    for i in prange(thread_count):
        idx = indices[i]
        best_seeds[i] = seeds[idx]
        best_energies[i] = energies[idx]

    return best_seeds, best_energies

@njit(parallel=True)
def run_gray_optimization(best_theta, iterators, gray_iterations, thread_count, is_spin_glass, G_data, G_rows, G_cols):
    n = len(best_theta)
    thread_iterations = (gray_iterations + thread_count - 1) // thread_count

    states = np.empty((thread_count, n), dtype=np.bool_)
    energies = np.full(thread_count, np.finfo(dtype).min, dtype=dtype)

    if is_spin_glass:
        for i in prange(thread_count):
            iterator = iterators[i]
            for curr_idx in range(thread_iterations):
                gray_code_next(iterator, curr_idx)
                energy = compute_energy_sparse(iterator, G_data, G_rows, G_cols, n)
                if energy > energies[i]:
                    states[i], energies[i] = iterator.copy(), energy
    else:
        for i in prange(thread_count):
            iterator = iterators[i]
            for curr_idx in range(thread_iterations):
                gray_code_next(iterator, curr_idx)
                energy = compute_cut_sparse(iterator, G_data, G_rows, G_cols, n)
                if energy > energies[i]:
                    states[i], energies[i] = iterator.copy(), energy

    best_index = np.argmax(energies)
    best_energy = energies[best_index]
    best_state = states[best_index]

    return best_energy, best_state


def spin_glass_solver_sparse(
    G,
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
            return "", 0, ([], []), 0

        if n_qubits == 1:
            return "0", 0, (nodes, []), 0

        if n_qubits == 2:
            weight = G_m[0, 1]
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
        bitstring, cut_value, _ = maxcut_tfim_sparse(G_m, quality=quality, shots=shots, is_spin_glass=is_spin_glass, anneal_t=anneal_t, anneal_h=anneal_h, repulsion_base=repulsion_base, is_maxcut_gpu=is_maxcut_gpu, is_nested=True)
    best_theta = np.array([b == "1" for b in list(bitstring)], dtype=np.bool_)

    if gray_iterations is None:
        gray_iterations = n_qubits * os.cpu_count()

    if gray_seed_multiple is None:
        gray_seed_multiple = os.cpu_count()

    max_energy = compute_energy_sparse(best_theta, G_m.data, G_m.indptr, G_m.indices, n_qubits) if is_spin_glass else cut_value

    thread_count = os.cpu_count() ** 2
    improved = True
    while improved:
        improved = False

        # Single bit flips with O(n^2)
        energy, state = run_single_bit_flips(best_theta, is_spin_glass, G_m.data, G_m.indptr, G_m.indices)
        if energy > max_energy:
            max_energy = energy
            best_theta = state
            improved = True
            continue

        # Double bit flips with O(n^3)
        energy, state = run_double_bit_flips(best_theta, is_spin_glass, G_m.data, G_m.indptr, G_m.indices)
        if energy > max_energy:
            max_energy = energy
            best_theta = state
            improved = True
            continue

        # Gray code with default O(n^3)
        iterators, energies = pick_gray_seeds(best_theta, thread_count, gray_seed_multiple, G_m.data, G_m.indptr, G_m.indices, n_qubits, is_spin_glass)
        best_i = np.argmax(energies)
        energy, state = energies[best_i], iterators[best_i]
        if energy > max_energy:
            max_energy = energy
            best_theta = state
            improved = True
            continue

        energy, state = run_gray_optimization(best_theta, iterators, gray_iterations, thread_count, is_spin_glass, G_m.data, G_m.indptr, G_m.indices)
        if energy > max_energy:
            max_energy = energy
            best_theta = state
            improved = True

    bitstring, l, r = get_cut(best_theta, nodes, n_qubits)
    if is_spin_glass:
        cut_value = compute_cut_sparse(best_theta, G_m.data, G_m.indptr, G_m.indices, n_qubits)
        min_energy = -max_energy
    else:
        cut_value = max_energy
        min_energy = compute_energy_sparse(best_theta, G_m.data, G_m.indptr, G_m.indices, n_qubits)

    return bitstring, float(cut_value), (l, r), float(min_energy)
