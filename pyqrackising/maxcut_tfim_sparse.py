import math
import networkx as nx
import numpy as np
import os
from numba import njit, prange

from .maxcut_tfim_util import get_cut, init_theta, init_thresholds, low_width, maxcut_hamming_cdf, opencl_context, probability_by_hamming_weight

IS_OPENCL_AVAILABLE = True
try:
    import pyopencl as cl
except ImportError:
    IS_OPENCL_AVAILABLE = False


@njit
def binary_search(l, t):
  left = 0
  right = len(l) - 1

  while left <= right:
    mid = (left + right) >> 1

    if l[mid] == t:
      return mid

    if l[mid] < t:
      left = mid + 1
    else:
      right = mid - 1

  return len(l)


# Written by Elara (OpenAI custom GPT) and improved by Dan Strano
@njit
def local_repulsion_choice(G_cols, G_data, G_rows, max_weight, weights, n, m):
    """
    Pick m nodes out of n with repulsion bias:
    - High-degree nodes are already less likely
    - After choosing a node, its neighbors' probabilities are further reduced
    adjacency_data, adjacency_rows: CSR-format sparse adjacency data
    weights: float64 array of shape (n,)
    """

    weights = weights.copy()
    chosen = np.zeros(m, dtype=np.int32)   # store chosen indices
    available = np.ones(n, dtype=np.bool_) # True = available, False = not
    mask = np.zeros(n, dtype=np.bool_)
    chosen_count = 0

    for _ in range(m):
        # Count available
        total_w = 0.0
        for i in range(n):
            if available[i]:
                total_w += weights[i]
        if total_w <= 0:
            break

        # Normalize & sample
        r = np.random.rand()
        cum = 0.0
        node = -1
        for i in range(n):
            if available[i]:
                cum += weights[i] / total_w
                if r < cum:
                    node = i
                    break

        if node == -1:
            continue

        # Select node
        chosen[chosen_count] = node
        chosen_count += 1
        available[node] = False
        mask[node] = True

        # Repulsion: penalize neighbors
        for j in range(G_rows[node], G_rows[node + 1]):
            nbr = G_cols[j]
            if available[nbr]:
                weights[nbr] *= 0.5 ** (G_data[j] / max_weight)  # tunable penalty factor

        for nbr in range(node):
            if not available[nbr]:
                continue
            start = G_rows[nbr]
            end = G_rows[nbr + 1]
            j = binary_search(G_cols[start:end], node) + start
            if j < end:
                weights[nbr] *= 0.5 ** (G_data[j] / max_weight)  # tunable penalty factor

    return mask


@njit
def compute_energy(sample, G_data, G_rows, G_cols):
    n_qubits = G_rows.shape[0] - 1
    energy = 0
    for u in range(n_qubits):
        for col in range(G_rows[u], G_rows[u + 1]):
            v = G_cols[col]
            energy += G_data[col] * (1 if sample[u] == sample[v] else -1)

    return energy


@njit(parallel=True)
def sample_for_solution(G_data, G_rows, G_cols, shots, thresholds, J_eff):
    n = G_rows.shape[0] - 1
    max_weight = G_data.max()
    weights = 1.0 / (1.0 + (2 ** -52) - J_eff)

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
        sample = local_repulsion_choice(G_cols, G_data, G_rows, max_weight, weights, n, m)
        solutions[s] = sample
        energies[s] = compute_energy(sample, G_data, G_rows, G_cols)

    best_solution = solutions[np.argmin(energies)]

    best_value = 0.0
    for u in range(n):
        for col in range(G_rows[u], G_rows[u + 1]):
            v = G_cols[col]
            if best_solution[u] != best_solution[v]:
                best_value += G_data[col]

    return best_solution, best_value


@njit(parallel=True)
def init_J_and_z(G_data, G_rows, G_cols):
    n_qubits = G_rows.shape[0] - 1
    degrees = np.empty(n_qubits, dtype=np.uint32)
    J_eff = np.empty(n_qubits, dtype=np.float32)
    for r in prange(n_qubits):
        # Row sum
        start = G_rows[r]
        end = G_rows[r + 1]
        degree = end - start
        val = G_data[start:end].sum()

        degrees[r] += degree
        J_eff[r] += val

        # Column sum
        for idx in range(start, end):
            c = G_cols[idx]
            degrees[c] += 1
            J_eff[c] += G_data[idx]

    J_max = -float("inf")
    for r in prange(n_qubits):
        J = J_eff[r]
        degree = degrees[r]
        J_eff[r] = -J / degree if degree > 0 else 0
        J_abs = abs(J)
        J_max = max(J_abs, J_max)
    J_eff /= J_max

    return J_eff, degrees


@njit
def cpu_footer(shots, quality, n_qubits, G_data, G_rows, G_cols, nodes):
    J_eff, degrees = init_J_and_z(G_data, G_rows, G_cols)
    hamming_prob = init_thresholds(n_qubits)

    maxcut_hamming_cdf(n_qubits, J_eff, degrees, quality, hamming_prob)

    best_solution, best_value = sample_for_solution(G_data, G_rows, G_cols, shots, hamming_prob, J_eff)

    bit_string, l, r = get_cut(best_solution, nodes)

    return bit_string, best_value, (l, r)


@njit
def gpu_footer(shots, n_qubits, G_data, G_rows, G_cols, J_eff, hamming_prob, nodes):
    hamming_prob /= hamming_prob.sum()
    tot_prob = 0.0
    for i in range(n_qubits - 1):
        tot_prob += hamming_prob[i]
        hamming_prob[i] = tot_prob
    hamming_prob[-1] = 2.0

    best_solution, best_value = sample_for_solution(G_data, G_rows, G_cols, shots, hamming_prob, J_eff)

    bit_string, l, r = get_cut(best_solution, nodes)

    return bit_string, best_value, (l, r)


def to_scipy_sparse_upper_triangular(G, nodes, n_nodes):
    lil = lil_matrix((n_nodes, n_nodes), dtype=np.float32)
    for u in range(n_nodes):
        u_node = nodes[u]
        for v in range(u + 1, n_nodes):
            v_node = nodes[v]
            if G.has_edge(u_node, v_node):
                lil[u, v] = G[u_node][v_node].get('weight', 1.0)

    return lil.tocsr()


def maxcut_tfim_sparse(
    G,
    quality=None,
    shots=None,
):
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
        return low_width(G_m, nodes, n_qubits)

    if quality is None:
        quality = 8

    if shots is None:
        # Number of measurement shots
        shots = n_qubits << quality

    n_steps = 1 << quality
    grid_size = n_steps * n_qubits

    if not (IS_OPENCL_AVAILABLE and grid_size >= 128):
        return cpu_footer(shots, quality, n_qubits, G_m.data, G_m.indptr, G_m.indices, nodes)

    J_eff, degrees = init_J_and_z(G_m.data, G_m.indptr, G_m.indices)
    hamming_prob = init_thresholds(n_qubits)

    delta_t = 1.0 / n_steps
    tot_t = 2.0 * n_steps * delta_t
    h_mult = 2.0 / tot_t
    theta = init_theta(delta_t, tot_t, h_mult, n_qubits, J_eff, degrees)
    args = np.empty(3, dtype=np.float32)
    args[0] = delta_t
    args[1] = tot_t
    args[2] = h_mult

    # Warp size is 32:
    group_size = n_qubits - 1
    if group_size > 256:
        group_size = 256
    grid_dim = n_steps * n_qubits * group_size

    # Move to GPU
    mf = cl.mem_flags
    args_buf = cl.Buffer(opencl_context.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=args)
    J_buf = cl.Buffer(opencl_context.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=J_eff)
    deg_buf = cl.Buffer(opencl_context.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=degrees)
    theta_buf = cl.Buffer(opencl_context.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=theta)
    ham_buf = cl.Buffer(opencl_context.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=hamming_prob)

    # Kernel execution
    opencl_context.maxcut_hamming_cdf_kernel(
        opencl_context.queue, (grid_dim,), (group_size,),
        np.int32(n_qubits), deg_buf, args_buf, J_buf, theta_buf, ham_buf
    )

    # Fetch results
    cl.enqueue_copy(opencl_context.queue, hamming_prob, ham_buf)

    return gpu_footer(shots, n_qubits, G_m.data, G_m.indptr, G_m.indices, J_eff, hamming_prob, nodes)
