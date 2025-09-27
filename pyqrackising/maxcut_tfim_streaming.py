import math
import networkx as nx
import numpy as np
import os
from numba import njit, prange

from .maxcut_tfim_util import fix_cdf, get_cut, init_thresholds, maxcut_hamming_cdf, opencl_context, probability_by_hamming_weight

IS_OPENCL_AVAILABLE = True
try:
    import pyopencl as cl
except ImportError:
    IS_OPENCL_AVAILABLE = False


@njit
def update_repulsion_choice(G_func, G_func_args_tuple, nodes, max_weight, weights, n, used, node):
    # Select node
    used[node] = True

    # Repulsion: penalize neighbors
    for nbr in range(n):
        if used[nbr]:
            continue
        weights[nbr] *= max(2e-8, 1 - G_func((nodes[node], nodes[nbr]), G_func_args_tuple) / max_weight)


# Written by Elara (OpenAI custom GPT) and improved by Dan Strano
@njit
def local_repulsion_choice(G_func, G_func_args_tuple, nodes, max_weight, weights, n, m, shot):
    """
    Pick m nodes out of n with repulsion bias:
    - High-degree nodes are already less likely
    - After choosing a node, its neighbors' probabilities are further reduced
    adjacency_data, adjacency_rows: CSR-format sparse adjacency data
    weights: float64 array of shape (n,)
    """

    weights = weights.copy()
    used = np.zeros(n, dtype=np.bool_) # False = available, True = used

    # Update answer and weights
    update_repulsion_choice(G_func, G_func_args_tuple, nodes, max_weight, weights, n, used, shot % n)

    for _ in range(1, m):
        # Count available
        total_w = 0.0
        for i in range(n):
            if used[i]:
                continue
            total_w += weights[i]

        # Normalize & sample
        r = np.random.rand()
        cum = 0.0
        node = -1
        for i in range(n):
            if used[i]:
                continue
            cum += weights[i]
            if (total_w * r) < cum:
                node = i
                break

        if node == -1:
            node = 0
            while used[node]:
                node += 1

        # Select node
        used[node] = True

        # Update answer and weights
        update_repulsion_choice(G_func, G_func_args_tuple, nodes, max_weight, weights, n, used, node)

    return used


@njit
def compute_energy(sample, G_func, G_func_args_tuple, nodes, n_qubits):
    energy = 0
    for u in range(n_qubits):
        for v in range(u + 1, n_qubits):
            energy += G_func((nodes[u], nodes[v]), G_func_args_tuple) * (1 if sample[u] == sample[v] else -1)

    return energy


@njit(parallel=True)
def sample_for_solution(G_func, G_func_args_tuple, nodes, max_weight, shots, thresholds, degrees_sum, weights, n):
    solutions = np.empty((shots, n), dtype=np.bool_)
    energies = np.empty(shots, dtype=np.float32)

    for s in prange(shots):
        # First dimension: Hamming weight
        mag_prob = np.random.random()
        m = 0
        while thresholds[m] < mag_prob:
            m += 1
        m += 1

        # Second dimension: permutation within Hamming weight
        sample = local_repulsion_choice(G_func, G_func_args_tuple, nodes, max_weight, weights, n, m, s)

        solutions[s] = sample
        energies[s] = compute_energy(sample, G_func, G_func_args_tuple, nodes, n)

    best_solution = solutions[np.argmin(energies)]

    best_value = 0.0
    for u in range(n):
        for v in range(u + 1, n):
            if best_solution[u] != best_solution[v]:
                best_value += G_func((nodes[u], nodes[v]), G_func_args_tuple)

    return best_solution, best_value


@njit(parallel=True)
def init_J_and_z(G_func, G_func_args_tuple, nodes):
    n_qubits = len(nodes)
    degrees = np.empty(n_qubits, dtype=np.uint32)
    J_eff = np.empty(n_qubits, dtype=np.float32)
    J_max = -float("inf")
    G_max = -float("inf")
    for n in prange(n_qubits):
        degree = 0
        J = 0.0
        for m in range(n_qubits):
            val = G_func((nodes[n], nodes[m]), G_func_args_tuple)
            if val != 0.0:
                degree += 1
            J += val
            G_max = max(val, G_max)
        J = -J / degree if degree > 0 else 0
        degrees[n] = degree
        J_eff[n] = J
        J_abs = abs(J)
        J_max = max(J_abs, J_max)
    J_eff /= J_max

    return J_eff, degrees, G_max


@njit
def cpu_footer(shots, quality, n_qubits, G_func, G_func_args_tuple, nodes):
    J_eff, degrees, G_max = init_J_and_z(G_func, G_func_args_tuple, nodes)
    hamming_prob = init_thresholds(n_qubits)

    maxcut_hamming_cdf(n_qubits, J_eff, degrees, quality, hamming_prob)

    max_weight = degrees.sum()
    degrees = None
    J_eff = 1.0 / (1.0 + (2e-52) - J_eff)
    weights = J_eff.astype(np.float64)
    J_eff = None

    best_solution, best_value = sample_for_solution(G_func, G_func_args_tuple, nodes, G_max, shots, hamming_prob, max_weight, weights, n_qubits)

    bit_string, l, r = get_cut(best_solution, nodes)

    return bit_string, best_value, (l, r)


@njit
def gpu_footer(shots, n_qubits, G_func, G_func_args_tuple, nodes, G_max, weights, degrees, hamming_prob, max_weight):
    fix_cdf(hamming_prob)

    best_solution, best_value = sample_for_solution(G_func, G_func_args_tuple, nodes, G_max, shots, hamming_prob, max_weight, weights, n_qubits)

    bit_string, l, r = get_cut(best_solution, nodes)

    return bit_string, best_value, (l, r)


def maxcut_tfim_streaming(
    G_func,
    nodes,
    G_func_args_tuple=None,
    quality=None,
    shots=None,
    is_base_maxcut_gpu=True
):
    n_qubits = len(nodes)

    if n_qubits < 3:
        if n_qubits == 0:
            return "", 0, ([], [])

        if n_qubits == 1:
            return "0", 0, (nodes, [])

        if n_qubits == 2:
            weight = G_func((nodes[0], nodes[1]), G_func_args_tuple)
            if weight < 0.0:
                return "00", 0, (nodes, [])

            return "01", weight, ([nodes[0]], [nodes[1]])

    if quality is None:
        quality = 2

    if shots is None:
        # Number of measurement shots
        shots = n_qubits << quality

    n_steps = 2 << quality
    grid_size = n_steps * n_qubits

    if (not is_base_maxcut_gpu) or not (IS_OPENCL_AVAILABLE and grid_size >= 128):
        return cpu_footer(shots, quality, n_qubits, G_func, G_func_args_tuple, nodes)

    J_eff, degrees, G_max = init_J_and_z(G_func, G_func_args_tuple, nodes)

    delta_t = 1.0 / n_steps
    tot_t = 2.0 * n_steps * delta_t
    h_mult = 2.0 / tot_t

    args = np.empty(3, dtype=np.float32)
    args[0] = delta_t
    args[1] = tot_t
    args[2] = h_mult

    # Move to GPU
    mf = cl.mem_flags
    args_buf = cl.Buffer(opencl_context.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=args)
    J_buf = cl.Buffer(opencl_context.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=J_eff)
    deg_buf = cl.Buffer(opencl_context.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=degrees)
    theta_buf = cl.Buffer(opencl_context.ctx, mf.READ_WRITE, size=(n_qubits * 4))

    # Warp size is 32:
    group_size = min(n_qubits, 64)
    global_size = ((n_qubits + group_size - 1) // group_size) * group_size

    opencl_context.init_theta_kernel(
        opencl_context.queue, (global_size,), (group_size,),
        args_buf, np.int32(n_qubits), J_buf, deg_buf, theta_buf
    )

    hamming_prob = init_thresholds(n_qubits)

    # Warp size is 32:
    group_size = n_qubits - 1
    if group_size > 256:
        group_size = 256
    grid_dim = n_steps * n_qubits * group_size

    # Move to GPU
    ham_buf = cl.Buffer(opencl_context.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=hamming_prob)

    # Kernel execution
    opencl_context.maxcut_hamming_cdf_kernel(
        opencl_context.queue, (grid_dim,), (group_size,),
        np.int32(n_qubits), deg_buf, args_buf, J_buf, theta_buf, ham_buf
    )

    # Fetch results
    cl.enqueue_copy(opencl_context.queue, hamming_prob, ham_buf)
    opencl_context.queue.finish()

    args_buf.release()
    J_buf.release()
    deg_buf.release()
    theta_buf.release()

    args_buf = None
    J_buf = None
    deg_buf = None
    theta_buf = None

    max_weight = degrees.sum()
    degrees = None
    J_eff = 1.0 / (1.0 + (2e-52) - J_eff)
    weights = J_eff.astype(np.float64)
    J_eff = None

    return gpu_footer(shots, n_qubits, G_func, G_func_args_tuple, nodes, G_max, weights, degrees, hamming_prob, max_weight)
