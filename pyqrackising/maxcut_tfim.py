import math
import networkx as nx
import numpy as np
import os
from numba import njit, prange

from .maxcut_tfim_util import convert_bool_to_uint, get_cut, get_cut_base, make_G_m_buf, make_theta_buf, maxcut_hamming_cdf, opencl_context, sample_mag, setup_opencl, bit_pick, init_bit_pick

IS_OPENCL_AVAILABLE = True
try:
    import pyopencl as cl
except ImportError:
    IS_OPENCL_AVAILABLE = False


epsilon = opencl_context.epsilon
dtype = opencl_context.dtype
wgs = opencl_context.work_group_size


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
    l, r = get_cut_base(sample, n_qubits)
    cut = 0
    for u in l:
        for v in r:
            cut += G_m[u, v]

    return cut


@njit(parallel=True)
def sample_measurement(G_m, max_edge, shots, thresholds, weights, repulsion_base, is_spin_glass):
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
        if is_spin_glass:
            for s in prange(shots):
                # First dimension: Hamming weight
                m = sample_mag(thresholds)

                # Second dimension: permutation within Hamming weight
                sample = local_repulsion_choice(G_m, max_edge, weights, tot_init_weight, repulsion_base, n, m)
                solutions[s] = sample
                energies[s] = compute_energy(sample, G_m, n)
        else:
            for s in prange(shots):
                # First dimension: Hamming weight
                m = sample_mag(thresholds)

                # Second dimension: permutation within Hamming weight
                sample = local_repulsion_choice(G_m, max_edge, weights, tot_init_weight, repulsion_base, n, m)
                solutions[s] = sample
                energies[s] = compute_cut(sample, G_m, n)

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
def shot_loop(G_m, max_edge, thresholds, weights, tot_init_weight, repulsion_base, n, shots, solutions):
    for s in prange(shots):
        # First dimension: Hamming weight
        m = sample_mag(thresholds)
        # Second dimension: permutation within Hamming weight
        solutions[s] = local_repulsion_choice(G_m, max_edge, weights, tot_init_weight, repulsion_base, n, m)


def sample_for_opencl(G_m, G_m_buf, max_edge, shots, thresholds, weights, repulsion_base, is_spin_glass, is_segmented, segment_size, theta_segment_size):
    shots = ((max(1, shots >> 1) + 31) >> 5) << 5
    n = len(G_m)
    tot_init_weight = weights.sum()

    solutions = np.empty((shots, n), dtype=np.bool_)

    best_solution = solutions[0]
    best_energy = -float("inf")

    opencl_args = setup_opencl(shots, shots, np.array([n, shots, is_spin_glass, segment_size, theta_segment_size], dtype=np.int32))

    improved = True
    while improved:
        improved = False
        shot_loop(G_m, max_edge, thresholds, weights, tot_init_weight, repulsion_base, n, shots, solutions)
        solution, energy = run_cut_opencl(solutions, G_m_buf, is_segmented, segment_size, is_spin_glass, *opencl_args)
        if energy > best_energy:
            best_energy = energy
            best_solution = solution.copy()
            improved = True

    if is_spin_glass:
        best_energy = compute_cut(best_solution, G_m, n)

    return best_solution, float(best_energy)


@njit(parallel=True)
def init_J_and_z(G_m, repulsion_base):
    G_min = G_m.min()
    n_qubits = len(G_m)
    degrees = np.empty(n_qubits, dtype=np.uint32)
    J_eff = np.empty(n_qubits, dtype=np.float64)
    weights = np.empty(n_qubits, dtype=np.float64)
    G_max = -float("inf")
    for n in prange(n_qubits):
        degree = 0
        J = 0.0
        weight = 0.0
        for m in range(n_qubits):
            val = G_m[n, m]
            weight += val
            if val > G_max:
                G_max = val
            val -= G_min
            if val != 0.0:
                degree += 1
                J += val
        if degree > 0:
            J = -J / degree
        weight = -weight / n_qubits
        degrees[n] = degree
        J_eff[n] = J
        weights[n] = weight

    weights = repulsion_base ** weights

    G_min = abs(G_min)
    G_max = abs(G_max)
    if G_min > G_max:
        G_max = G_min

    return J_eff, degrees, weights, G_max


@njit
def cpu_footer(shots, quality, n_qubits, G_m, nodes, is_spin_glass, anneal_t, anneal_h, repulsion_base):
    J_eff, degrees, weights, max_edge = init_J_and_z(G_m, repulsion_base)
    hamming_prob = maxcut_hamming_cdf(n_qubits, J_eff, degrees, quality, anneal_t, anneal_h)

    degrees = None
    J_eff = None

    best_solution, best_value = sample_measurement(G_m, max_edge, shots, hamming_prob, weights, repulsion_base, is_spin_glass)

    bit_string, l, r = get_cut(best_solution, nodes, n_qubits)

    return bit_string, best_value, (l, r)


def run_cut_opencl(samples, G_m_buf, is_segmented, segment_size, is_spin_glass, local_size, global_size, args_buf, local_energy_buf, local_index_buf, max_energy_host, max_index_host, max_energy_buf, max_index_buf):
    queue = opencl_context.queue
    calculate_cut_kernel = opencl_context.calculate_cut_segmented_kernel if is_segmented else opencl_context.calculate_cut_kernel

    shots = samples.shape[0]
    n = samples.shape[1]

    # Buffers
    mf = cl.mem_flags
    theta_buf = make_theta_buf(convert_bool_to_uint(samples), is_segmented, shots, n)

    # Set kernel args
    if is_segmented:
        calculate_cut_kernel.set_args(
            G_m_buf[0],
            G_m_buf[1],
            G_m_buf[2],
            G_m_buf[3],
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
            G_m_buf,
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
    cl.enqueue_copy(queue, max_index_host, max_index_buf)
    queue.finish()

    # Find global minimum
    best_x = np.argmax(max_energy_host)
    best_i = max_index_host[best_x]

    return samples[best_i], max_energy_host[best_x]

def maxcut_tfim(
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
    nodes = None
    G_m = None
    if isinstance(G, nx.Graph):
        nodes = list(G.nodes())
        G_m = nx.to_numpy_array(G, weight='weight', nonedge=0.0, dtype=dtype)
    else:
        nodes = list(range(len(G)))
        G_m = G

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
        quality = 5

    if shots is None:
        # Number of measurement shots
        shots = n_qubits << quality

    if anneal_t is None:
        anneal_t = 8.0

    if anneal_h is None:
        anneal_h = 8.0

    if repulsion_base is None:
        repulsion_base = 8.0

    is_opencl = is_maxcut_gpu and IS_OPENCL_AVAILABLE

    if not is_opencl:
        bit_string, best_value, partition = cpu_footer(shots, quality, n_qubits, G_m, nodes, is_spin_glass, anneal_t, anneal_h, repulsion_base)

        if best_value < 0.0:
            # Best cut is trivial partition, all/empty
            return '0' * n_qubits, 0.0, (nodes, [])

        return bit_string, best_value, partition

    segment_size = (G_m.shape[0] * G_m.shape[1] + 3) >> 2
    theta_segment_size = (((n_qubits + 31) >> 5) * (((shots + wgs - 1) // wgs) * wgs) + 3) >> 2
    is_segmented = ((G_m.nbytes << 1) > opencl_context.max_alloc) or ((theta_segment_size << 3) > opencl_context.max_alloc)
    G_m_buf = make_G_m_buf(G_m, is_segmented, segment_size)

    J_eff, degrees, weights, max_edge = init_J_and_z(G_m, repulsion_base)
    hamming_prob = maxcut_hamming_cdf(n_qubits, J_eff, degrees, quality, anneal_t, anneal_h)

    degrees = None
    J_eff = None

    best_solution, best_value = sample_for_opencl(G_m, G_m_buf, max_edge, shots, hamming_prob, weights, repulsion_base, is_spin_glass, is_segmented, segment_size, theta_segment_size)

    bit_string, l, r = get_cut(best_solution, nodes, n_qubits)

    if best_value < 0.0:
        # Best cut is trivial partition, all/empty
        return '0' * n_qubits, 0.0, (nodes, [])

    return bit_string, best_value, (l, r)
