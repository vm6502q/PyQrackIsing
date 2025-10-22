import networkx as nx
import numpy as np
from numba import njit, prange
import os

from .maxcut_tfim_util import binary_search, compute_cut_sparse, compute_energy_sparse, convert_bool_to_uint, get_cut, get_cut_base, make_G_m_csr_buf, make_theta_buf, maxcut_hamming_cdf, opencl_context, sample_mag, setup_opencl, bit_pick, to_scipy_sparse_upper_triangular

IS_OPENCL_AVAILABLE = True
try:
    import pyopencl as cl
except ImportError:
    IS_OPENCL_AVAILABLE = False


epsilon = opencl_context.epsilon
dtype = opencl_context.dtype
wgs = opencl_context.work_group_size


@njit
def update_repulsion_choice(G_data, G_rows, G_cols, weights, n, used, node, repulsion_base):
    # Select node
    used[node] = True

    # Repulsion: penalize neighbors
    for j in range(G_rows[node], G_rows[node + 1]):
        nbr = G_cols[j]
        if used[nbr]:
            continue
        weights[nbr] *= repulsion_base ** (-G_data[j])

    for nbr in range(node):
        if used[nbr]:
            continue
        start = G_rows[nbr]
        end = G_rows[nbr + 1]
        j = binary_search(G_cols[start:end], node) + start
        if j < end:
            weights[nbr] *= repulsion_base ** (-G_data[j])


# Written by Elara (OpenAI custom GPT) and improved by Dan Strano
@njit
def local_repulsion_choice(G_data, G_rows, G_cols, repulsion_base, n, m, s):
    """
    Pick m nodes out of n with repulsion bias:
    - High-degree nodes are already less likely
    - After choosing a node, its neighbors' probabilities are further reduced
    """

    used = np.zeros(n, dtype=np.bool_) # False = available, True = used

    # First bit:
    node = s % n
    if m == 1:
        used[node] = True
        return used

    weights = np.ones(n, dtype=np.float64)
    update_repulsion_choice(G_data, G_rows, G_cols, weights, n, used, node, repulsion_base)

    for _ in range(1, m - 1):
        node = bit_pick(weights, used, n)
        update_repulsion_choice(G_data, G_rows, G_cols, weights, n, used, node, repulsion_base)

    node = bit_pick(weights, used, n)
    used[node] = True

    return used


@njit(parallel=True)
def sample_measurement(G_data, G_rows, G_cols, shots, thread_count, thresholds, repulsion_base, is_spin_glass):
    shot_segment = (max(1, shots >> 1) + thread_count - 1) // thread_count
    shots = shot_segment * thread_count
    n = G_rows.shape[0] - 1

    solutions = np.empty((thread_count, n), dtype=np.bool_)
    energies = np.full(thread_count, np.finfo(dtype).min, dtype=dtype)

    best_solution = solutions[0]
    best_energy = -float("inf")

    improved = True
    while improved:
        improved = False
        if is_spin_glass:
            for i in prange(thread_count):
                s_offset = i * shot_segment
                for j in range(shot_segment):
                    s = s_offset + j

                    # First dimension: Hamming weight
                    m = sample_mag(thresholds)

                    # Second dimension: permutation within Hamming weight
                    sample = local_repulsion_choice(G_data, G_rows, G_cols, repulsion_base, n, m, s)
                    energy = compute_energy_sparse(sample, G_data, G_rows, G_cols, n)

                    if energy > energies[i]:
                        solutions[i], energies[i] = sample, energy
        else:
            for i in prange(thread_count):
                s_offset = i * shot_segment
                for j in range(shot_segment):
                    s = s_offset + j

                    # First dimension: Hamming weight
                    m = sample_mag(thresholds)

                    # Second dimension: permutation within Hamming weight
                    sample = local_repulsion_choice(G_data, G_rows, G_cols, repulsion_base, n, m, s)
                    energy = compute_cut_sparse(sample, G_data, G_rows, G_cols, n)

                    if energy > energies[i]:
                        solutions[i], energies[i] = sample, energy

        best_index = np.argmax(energies)
        energy = energies[best_index]
        if energy > best_energy:
            best_energy = energy
            best_solution = solutions[best_index].copy()
            improved = True

    if is_spin_glass:
        best_energy = compute_cut_sparse(sample, G_data, G_rows, G_cols, n)

    return best_solution, best_energy


@njit(parallel=True)
def shot_loop(G_data, G_rows, G_cols, thresholds, repulsion_base, n, shots, solutions):
    for s in prange(shots):
        # First dimension: Hamming weight
        m = sample_mag(thresholds)
        # Second dimension: permutation within Hamming weight
        solutions[s] = local_repulsion_choice(G_data, G_rows, G_cols, repulsion_base, n, m, s)


def sample_for_opencl(G_data, G_rows, G_cols, G_data_buf, G_rows_buf, G_cols_buf, shots, thresholds, repulsion_base, is_spin_glass, is_segmented, segment_size, theta_segment_size):
    shots = ((max(1, shots >> 1) + 31) >> 5) << 5
    n = G_rows.shape[0] - 1

    solutions = np.empty((shots, n), dtype=np.bool_)

    best_solution = solutions[0]
    best_energy = -float("inf")

    opencl_args = setup_opencl(shots, shots, np.array([n, shots, is_spin_glass, segment_size, theta_segment_size], dtype=np.int32))

    improved = True
    while improved:
        improved = False
        shot_loop(G_data, G_rows, G_cols, thresholds, repulsion_base, n, shots, solutions)
        solution, energy = run_cut_opencl(best_energy, solutions, G_data_buf, G_rows_buf, G_cols_buf, is_segmented, segment_size, is_spin_glass, *opencl_args)
        if energy > best_energy:
            best_energy = energy
            best_solution = solution.copy()
            improved = True

    if is_spin_glass:
        best_energy = compute_cut_sparse(best_solution, G_data, G_rows, G_cols, n) 

    return best_solution, float(best_energy)


def init_J_and_z(G_m, repulsion_base):
    G_min = G_m.min()
    n_qubits = G_m.shape[0]
    degrees = np.empty(n_qubits, dtype=np.uint32)
    J_eff = np.empty(n_qubits, dtype=np.float64)
    for n in prange(n_qubits):
        degree = 0
        J = 0.0
        weight = 0.0
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


@njit
def cpu_footer(J_eff, degrees, shots, thread_count, quality, n_qubits, G_data, G_rows, G_cols, nodes, is_spin_glass, anneal_t, anneal_h, repulsion_base):
    hamming_prob = maxcut_hamming_cdf(n_qubits, J_eff, degrees, quality, anneal_t, anneal_h)

    best_solution, best_value = sample_measurement(G_data, G_rows, G_cols, shots, thread_count, hamming_prob, repulsion_base, is_spin_glass)

    bit_string, l, r = get_cut(best_solution, nodes, n_qubits)

    return bit_string, best_value, (l, r)


def run_cut_opencl(best_energy, samples, G_data_buf, G_rows_buf, G_cols_buf, is_segmented, segment_size, is_spin_glass, local_size, global_size, args_buf, local_energy_buf, local_index_buf, max_energy_host, max_index_host, max_energy_buf, max_index_buf):
    queue = opencl_context.queue
    calculate_cut_kernel = opencl_context.calculate_cut_sparse_segmented_kernel if is_segmented else opencl_context.calculate_cut_sparse_kernel

    shots = samples.shape[0]
    n = samples.shape[1]

    # Buffers
    mf = cl.mem_flags
    theta_buf = make_theta_buf(convert_bool_to_uint(samples), is_segmented, shots, n)

    # Set kernel args
    if is_segmented:
        calculate_cut_kernel.set_args(
            G_data_buf[0],
            G_data_buf[1],
            G_data_buf[2],
            G_data_buf[3],
            G_rows_buf,
            G_cols_buf,
            theta_buf[0],
            theta_buf[1],
            theta_buf[2],
            theta_buf[3],
            args_buf,
            max_energy_buf,
            max_index_buf,
            local_energy_buf,
            local_index_buf
        )
    else:
        calculate_cut_kernel.set_args(
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

    cl.enqueue_nd_range_kernel(queue, calculate_cut_kernel, (global_size,), (local_size,))

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
        return samples[0], max_energy_host[0]

    # We need the best index
    queue.finish()

    return samples[max_index_host[best_x]], energy



def maxcut_tfim_sparse(
    G,
    quality=None,
    shots=None,
    is_spin_glass=False,
    anneal_t=None,
    anneal_h=None,
    repulsion_base=None,
    is_maxcut_gpu=True,
    is_nested=False
):
    wgs = opencl_context.work_group_size
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
            return "", 0, ([], [])

        if n_qubits == 1:
            return "0", 0, (nodes, [])

        if n_qubits == 2:
            weight = G_m[0, 1]
            if weight < 0.0:
                return "00", 0, (nodes, [])

            return "01", weight, ([nodes[0]], [nodes[1]])

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

    J_eff, degrees = init_J_and_z(G_m, repulsion_base)

    n_qubits = G_m.shape[0]

    is_opencl = is_maxcut_gpu and IS_OPENCL_AVAILABLE

    if not is_opencl:
        thread_count = os.cpu_count() ** 2

        bit_string, best_value, partition = cpu_footer(J_eff, degrees, shots, thread_count, quality, n_qubits, G_m.data, G_m.indptr, G_m.indices, nodes, is_spin_glass, anneal_t, anneal_h, repulsion_base)

        if best_value < 0.0:
            # Best cut is trivial partition, all/empty
            return '0' * n_qubits, 0.0, (nodes, [])

        return bit_string, best_value, partition

    segment_size = (G_m.data.shape[0] + 3) >> 2
    theta_segment_size = (((n_qubits + 31) >> 5) * (((shots + wgs - 1) // wgs) * wgs) + 3) >> 2
    is_segmented = (G_m.data.nbytes << 1) > opencl_context.max_alloc or ((theta_segment_size << 3) > opencl_context.max_alloc)

    G_data_buf, G_rows_buf, G_cols_buf = make_G_m_csr_buf(G_m, is_segmented, segment_size)

    hamming_prob = maxcut_hamming_cdf(n_qubits, J_eff, degrees, quality, anneal_t, anneal_h)

    degrees = None
    J_eff = None

    best_solution, best_value = sample_for_opencl(G_m.data, G_m.indptr, G_m.indices, G_data_buf, G_rows_buf, G_cols_buf, shots, hamming_prob, repulsion_base, is_spin_glass, is_segmented, segment_size, theta_segment_size)

    bit_string, l, r = get_cut(best_solution, nodes, n_qubits)

    if best_value < 0.0:
        # Best cut is trivial partition, all/empty
        return '0' * n_qubits, 0.0, (nodes, [])

    return bit_string, best_value, (l, r)
