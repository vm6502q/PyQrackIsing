from .maxcut_tfim_sparse import maxcut_tfim_sparse
from .maxcut_tfim_util import compute_cut_sparse, compute_energy_sparse, get_cut, gray_code_next, gray_mutation, int_to_bitstring, make_G_m_csr_buf, make_best_theta_buf, opencl_context, setup_opencl, to_scipy_sparse_upper_triangular
import networkx as nx
import numpy as np
from numba import njit, prange
import os


IS_OPENCL_AVAILABLE = True
try:
    import pyopencl as cl
except ImportError:
    IS_OPENCL_AVAILABLE = False


dtype = opencl_context.dtype
wgs = opencl_context.work_group_size


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
def run_double_bit_flips(best_theta, is_spin_glass, G_data, G_rows, G_cols, thread_count):
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

                states[t], energies[t] = state, compute_energy_sparse(state, G_data, G_rows, G_cols, n)

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

                states[t], energies[t] = state, compute_cut_sparse(state, G_data, G_rows, G_cols, n)

                s += thread_batch

    best_index = np.argmax(energies)
    best_energy = energies[best_index]
    best_state = states[best_index]

    return best_energy, best_state


@njit(parallel=True)
def pick_gray_seeds(best_theta, thread_count, gray_seed_multiple, G_data, G_rows, G_cols, n, is_spin_glass):
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
            energies[s] = compute_energy_sparse(seed, G_data, G_rows, G_cols, n)
            seeds[s] = seed
    else:
        for s in prange(seed_count):
            i = s % block_size
            offset = (s // block_size) * 64
            seed = gray_mutation(i, best_theta, offset)
            energies[s] = compute_cut_sparse(seed, G_data, G_rows, G_cols, n)
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
def run_gray_optimization(best_theta, iterators, energies, gray_iterations, thread_count, is_spin_glass, G_data, G_rows, G_cols):
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
                    energy = compute_energy_sparse(iterator, G_data, G_rows, G_cols, n)
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
                    energy = compute_cut_sparse(iterator, G_data, G_rows, G_cols, n)
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


def run_bit_flips_opencl(is_double, n, kernel, best_energy, theta, theta_buf, G_data_buf, G_rows_buf, G_cols_buf, is_segmented, local_size, global_size, args_buf, local_energy_buf, local_index_buf, max_energy_host, max_index_host, max_energy_buf, max_index_buf):
    queue = opencl_context.queue

    # Set kernel args
    if is_segmented:
        kernel.set_args(
            G_data_buf[0],
            G_data_buf[1],
            G_data_buf[2],
            G_data_buf[3],
            G_rows_buf,
            G_cols_buf,
            theta_buf,
            args_buf,
            max_energy_buf,
            max_index_buf,
            local_energy_buf,
            local_index_buf
        )
    else:
        kernel.set_args(
            G_data_buf,
            G_rows_buf,
            G_cols_buf,
            theta_buf,
            args_buf,
            max_energy_buf,
            max_index_buf,
            local_energy_buf,
            local_index_buf
        )

    cl.enqueue_nd_range_kernel(queue, kernel, (global_size,), (local_size,))

    # Read results
    cl.enqueue_copy(queue, max_energy_host, max_energy_buf)
    queue.finish()

    # Queue read for results we might not need
    cl.enqueue_copy(queue, max_index_host, max_index_buf)

    # Find global minimum
    best_x = np.argmax(max_energy_host)
    energy = max_energy_host[best_x]

    if energy <= best_energy:
        # No improvement: we can exit early
        return best_energy, theta

    # We need the best index
    queue.finish()

    s = max_index_host[best_x]
    theta = theta.copy()

    if is_double:
        c = s
        i = 0
        lcv = n - 1
        while c >= lcv:
            c -= lcv
            i += 1
            lcv -= 1
            if not lcv:
                break
        j = c + i + 1

        theta = theta.copy()
        theta[i] = not theta[i]
        theta[j] = not theta[j]
    else:
        theta[s] = not theta[s]

    return energy, theta


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

    is_opencl = is_maxcut_gpu and IS_OPENCL_AVAILABLE

    if is_opencl:
        segment_size = (G_m.data.shape[0] + 3) >> 2
        is_segmented = (G_m.data.nbytes << 1) > opencl_context.max_alloc
        G_data_buf, G_rows_buf, G_cols_buf = make_G_m_csr_buf(G_m, is_segmented, segment_size)

        local_work_group_size = min(wgs, n_qubits)
        global_work_group_size = n_qubits
        opencl_args = setup_opencl(local_work_group_size, global_work_group_size, np.array([n_qubits, is_spin_glass, segment_size]))

        if is_segmented:
            single_bit_flips_kernel = opencl_context.single_bit_sparse_flips_segmented_kernel
            double_bit_flips_kernel = opencl_context.double_bit_sparse_flips_segmented_kernel
        else:
            single_bit_flips_kernel = opencl_context.single_bit_flips_sparse_kernel
            double_bit_flips_kernel = opencl_context.double_bit_flips_sparse_kernel

    thread_count = os.cpu_count() ** 2
    improved = True
    while improved:
        improved = False

        # Single bit flips with O(n^2)
        if is_opencl:
            theta_buf = make_best_theta_buf(best_theta)
            energy, state = run_bit_flips_opencl(False, n_qubits, single_bit_flips_kernel, max_energy, best_theta, theta_buf, G_data_buf, G_rows_buf, G_cols_buf, is_segmented, *opencl_args)
        else:
            energy, state = run_single_bit_flips(best_theta, is_spin_glass, G_m.data, G_m.indptr, G_m.indices)
        if energy > max_energy:
            max_energy = energy
            best_theta = state
            improved = True
            continue

        # Double bit flips with O(n^3)
        if is_opencl:
            # theta_buf has not changed
            energy, state = run_bit_flips_opencl(True, n_qubits, double_bit_flips_kernel, max_energy, best_theta, theta_buf, G_data_buf, G_rows_buf, G_cols_buf, is_segmented, *opencl_args)
        else:
            energy, state = run_double_bit_flips(best_theta, is_spin_glass, G_m.data, G_m.indptr, G_m.indices, thread_count)
        if energy > max_energy:
            max_energy = energy
            best_theta = state
            improved = True
            continue

        # Gray code with default O(n^3)
        iterators, energies = pick_gray_seeds(best_theta, thread_count, gray_seed_multiple, G_m.data, G_m.indptr, G_m.indices, n_qubits, is_spin_glass)
        energy, state = energies[0], iterators[0]
        if energy > max_energy:
            max_energy = energy
            best_theta = state
            improved = True
            continue

        energy, state = run_gray_optimization(best_theta, iterators, energies, gray_iterations, thread_count, is_spin_glass, G_m.data, G_m.indptr, G_m.indices)
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
