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
def update_repulsion_choice(G_m, max_weight, weights, n, used, node):
    # Select node
    used[node] = True

    # Repulsion: penalize neighbors
    for nbr in range(n):
        if used[nbr]:
            continue
        weights[nbr] *= max(2e-8, 1 - G_m[node, nbr] / max_weight)


# Written by Elara (OpenAI custom GPT) and improved by Dan Strano
@njit
def local_repulsion_choice(G_m, max_weight, weights, n, m, shot):
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
    update_repulsion_choice(G_m, max_weight, weights, n, used, shot % n)

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

        # Update answer and weights
        update_repulsion_choice(G_m, max_weight, weights, n, used, node)

    return used


@njit
def compute_energy(sample, G_m, n_qubits):
    energy = 0
    for u in range(n_qubits):
        for v in range(u + 1, n_qubits):
            energy += G_m[u, v] * (1 if sample[u] == sample[v] else -1)

    return energy


@njit(parallel=True)
def sample_for_solution(G_m, shots, thresholds, weights):
    n = len(G_m)
    max_weight = G_m.max()

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
        sample = local_repulsion_choice(G_m, max_weight, weights, n, m, s)

        solutions[s] = sample
        energies[s] = compute_energy(sample, G_m, n)

    best_solution = solutions[np.argmin(energies)]

    best_value = 0.0
    for u in range(n):
        for v in range(u + 1, n):
            if best_solution[u] != best_solution[v]:
                best_value += G_m[u, v]

    return best_solution, best_value


@njit(parallel=True)
def init_J_and_z(G_m):
    n_qubits = len(G_m)
    degrees = np.empty(n_qubits, dtype=np.uint32)
    J_eff = np.empty(n_qubits, dtype=np.float32)
    J_max = -float("inf")
    for n in prange(n_qubits):
        degree = sum(G_m[n] != 0.0)
        J = (-G_m[n].sum() / degree) if degree > 0 else 0
        degrees[n] = degree
        J_eff[n] = J
        J_abs = abs(J)
        J_max = max(J_abs, J_max)
    J_eff /= J_max

    return J_eff, degrees


def run_sampling_opencl(G_m_np, thresholds_np, shots, n, is_g_buf_reused):
    ctx = opencl_context.ctx
    queue = opencl_context.queue
    kernel = opencl_context.sample_for_solution_best_bitset_kernel

    max_local_size = 64  # tune
    max_global_size = ((opencl_context.MAX_GPU_PROC_ELEM + max_local_size - 1) // max_local_size) * max_local_size  # corresponds to MAX_PROC_ELEM macro in OpenCL kernel program
    global_size = min(((shots + max_local_size - 1) // max_local_size) * max_local_size, max_global_size)
    local_size = max_local_size
    num_groups = global_size // local_size

    # Bit-packing params
    words = (n + 31) // 32

    # Random seeds (host)
    rng_seeds_np = np.random.randint(1, 2**31-1, size=global_size, dtype=np.uint32)

    # Device buffers
    mf = cl.mem_flags
    G_m_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=G_m_np)
    thresholds_buf = cl.Buffer(opencl_context.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=thresholds_np)
    rng_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=rng_seeds_np)

    # Solutions buffer: each work-item writes a candidate bitset
    solutions_buf = cl.Buffer(ctx, mf.READ_WRITE, size=num_groups * words * np.uint32().nbytes)

    # Best energies and solutions (per group)
    # We'll reuse solutions_buf to hold them after reduction (kernel copies winners into group slices)
    best_energies_np = np.empty(num_groups, dtype=np.float32)
    best_energies_buf = cl.Buffer(ctx, mf.WRITE_ONLY, best_energies_np.nbytes)

    # Local memory buffers
    local_energy_buf = cl.LocalMemory(np.dtype(np.float32).itemsize * local_size)
    local_index_buf = cl.LocalMemory(np.dtype(np.int32).itemsize * local_size)

    # Set kernel args
    kernel.set_args(
        G_m_buf,
        thresholds_buf,
        np.int32(n),
        np.int32(shots),
        np.float64(G_m_np.max()),
        rng_buf,
        solutions_buf,
        best_energies_buf,
        local_energy_buf,
        local_index_buf
    )

    # Launch
    cl.enqueue_nd_range_kernel(queue, kernel, (global_size,), (local_size,))

    # Copy back
    cl.enqueue_copy(queue, best_energies_np, best_energies_buf)
    solutions_np = np.empty(num_groups * words, dtype=np.uint32)
    cl.enqueue_copy(queue, solutions_np, solutions_buf)  # all candidates, includes per-group winners
    queue.finish()

    # Host reduction
    best_group = np.argmax(best_energies_np)  # max cut
    best_energy = float(best_energies_np[best_group])
    best_solution_bits = solutions_np[best_group * words : (best_group + 1) * words]

    # Unpack bitset into boolean vector
    best_solution = np.zeros(n, dtype=np.bool_)
    for u in range(n):
        w = u >> 5
        b = u & 31
        best_solution[u] = (best_solution_bits[w] >> b) & 1

    if is_g_buf_reused:
        opencl_context.G_m_buf = G_m_buf

    return best_solution, best_energy


@njit
def cpu_footer(shots, quality, n_qubits, G_m, nodes):
    J_eff, degrees = init_J_and_z(G_m)
    hamming_prob = init_thresholds(n_qubits)

    maxcut_hamming_cdf(n_qubits, J_eff, degrees, quality, hamming_prob)

    degrees = None
    J_eff = 1.0 / (1.0 + (2e-52) - J_eff)
    weights = J_eff.astype(np.float64)
    J_eff = None

    best_solution, best_value = sample_for_solution(G_m, shots, hamming_prob, weights)

    bit_string, l, r = get_cut(best_solution, nodes)

    return bit_string, best_value, (l, r)


@njit
def gpu_footer(shots, n_qubits, G_m, weights, hamming_prob, nodes):
    fix_cdf(hamming_prob)

    best_solution, best_value = sample_for_solution(G_m, shots, hamming_prob, weights)

    bit_string, l, r = get_cut(best_solution, nodes)

    return bit_string, best_value, (l, r)


def maxcut_tfim(
    G,
    quality=None,
    shots=None,
    is_alt_gpu_sampling=False,
    is_g_buf_reused=False,
    is_base_maxcut_gpu=True
):
    nodes = None
    n_qubits = 0
    G_m = None
    if isinstance(G, nx.Graph):
        nodes = list(G.nodes())
        n_qubits = len(nodes)
        G_m = nx.to_numpy_array(G, weight='weight', nonedge=0.0)
    else:
        n_qubits = len(G)
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
        quality = 2

    if shots is None:
        # Number of measurement shots
        shots = n_qubits << quality

    n_steps = 2 << quality
    grid_size = n_steps * n_qubits

    if (not is_base_maxcut_gpu) or not (IS_OPENCL_AVAILABLE and grid_size >= 128):
        return cpu_footer(shots, quality, n_qubits, G_m, nodes)

    J_eff, degrees = init_J_and_z(G_m)

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
    theta_buf = cl.Buffer(opencl_context.ctx, mf.READ_WRITE, size=(n_qubits * np.float32().nbytes))

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

    if not is_alt_gpu_sampling:
        degrees = None
        J_eff = 1.0 / (1.0 + (2e-52) - J_eff)
        weights = J_eff.astype(np.float64)
        J_eff = None

        return gpu_footer(shots, n_qubits, G_m, weights, hamming_prob, nodes)

    fix_cdf(hamming_prob)
    best_solution, best_value = run_sampling_opencl(G_m, hamming_prob, shots, n_qubits, is_g_buf_reused)
    bit_string, l, r = get_cut(best_solution, nodes)

    return bit_string, best_value, (l, r)
