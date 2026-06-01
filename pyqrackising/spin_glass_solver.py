from .maxcut_tfim import maxcut_tfim
from .maxcut_tfim_util import (
    compute_cut,
    compute_energy,
    compute_cut_diff,
    compute_cut_diff_2,
    compute_cut_diff_between,
    get_cut,
    gray_code_next,
    gray_mutation,
    heuristic_threshold,
    int_to_bitstring,
    make_G_m_buf,
    make_best_theta_buf,
    make_best_theta_buf_64,
    opencl_context,
    setup_opencl,
)
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
gnl = opencl_context.GRAY_NODE_LIMIT


@njit(parallel=True, cache=True)
def run_single_bit_flips(best_theta, is_spin_glass, G_m):
    n = len(best_theta)
    energies = np.empty(n, dtype=dtype)
    for i in prange(n):
        state = best_theta.copy()
        state[i] = not state[i]
        energies[i] = compute_cut_diff(i, state, G_m, n)
    best_index = np.argmax(energies)
    best_energy = energies[best_index]
    if is_spin_glass:
        best_energy *= 2.0
    best_state = best_theta.copy()
    best_state[best_index] = not best_state[best_index]

    return best_energy, best_state


@njit(parallel=True, cache=True)
def run_double_bit_flips(best_theta, is_spin_glass, G_m, thread_count):
    n = len(best_theta)
    combo_count = (n * (n - 1)) // 2
    thread_batch = (combo_count + thread_count - 1) // thread_count

    states = np.empty((thread_count, n), dtype=np.bool_)
    energies = np.empty(thread_count, dtype=dtype)

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

            states[t], energies[t] = state, compute_cut_diff_2(i, j, state, G_m, n)

            s += thread_batch

    best_index = np.argmax(energies)
    best_energy = energies[best_index]
    if is_spin_glass:
        best_energy *= 2.0
    best_state = states[best_index]

    return best_energy, best_state


@njit(parallel=True, cache=True)
def pick_gray_seeds(best_theta, thread_count, gray_seed_multiple, G_m, n, is_spin_glass):
    blocks = (n + 63) >> 6
    block_size = thread_count * gray_seed_multiple
    seed_count = block_size * blocks

    seeds = np.empty((seed_count, n), dtype=np.bool_)
    energies = np.empty(seed_count, dtype=dtype)

    for s in prange(seed_count):
        i = s % block_size
        offset = (s // block_size) << 6
        seed = gray_mutation(i, best_theta, offset)
        energies[s] = compute_cut_diff_between(best_theta, seed, G_m, n)
        seeds[s] = seed

    indices = np.argpartition(energies, -thread_count)[-thread_count:]
    indices = indices[np.argsort(energies[indices])[::-1]]
    best_seeds = np.empty((thread_count, n), dtype=np.bool_)
    best_energies = np.empty(thread_count, dtype=dtype)
    for i in prange(thread_count):
        idx = indices[i]
        best_seeds[i] = seeds[idx]
        best_energies[i] = energies[idx]

    if is_spin_glass:
        best_energies *= 2.0

    return best_seeds, best_energies[0]


@njit(parallel=True, cache=True)
def run_gray_optimization(best_theta, iterators, gray_iterations, thread_count, is_spin_glass, G_m):
    n = len(best_theta)
    thread_iterations = (gray_iterations + thread_count - 1) // thread_count
    blocks = (n + 63) >> 6
    energies = np.empty(thread_count, dtype=dtype)

    for i in prange(thread_count):
        iterator = iterators[i]
        for curr_idx in range(thread_iterations):
            best_energy = 0.0
            for block in range(blocks):
                flip_bit = gray_code_next(iterator, curr_idx, block << 6)
                energy = compute_cut_diff(flip_bit, iterator, G_m, n)
                if energy > 0.0:
                    best_energy += energy
                else:
                    # Revert iterator
                    iterator[flip_bit] = not iterator[flip_bit]
            energies[i] += best_energy

    best_index = np.argmax(energies)
    best_energy = energies[best_index]
    if is_spin_glass:
        best_energy *= 2.0
    best_state = iterators[best_index]

    return best_energy, best_state


def run_bit_flips_opencl(
    is_double,
    n,
    kernel,
    theta,
    theta_buf,
    G_m_buf,
    is_segmented,
    local_size,
    global_size,
    args_buf,
    local_energy_buf,
    local_index_buf,
    max_energy_host,
    max_index_host,
    max_energy_buf,
    max_index_buf,
):
    queue = opencl_context.queue

    # Set kernel args
    if is_segmented:
        kernel.set_args(
            G_m_buf[0],
            G_m_buf[1],
            G_m_buf[2],
            G_m_buf[3],
            theta_buf,
            args_buf,
            max_energy_buf,
            max_index_buf,
            local_energy_buf,
            local_index_buf,
        )
    else:
        kernel.set_args(
            G_m_buf,
            theta_buf,
            args_buf,
            max_energy_buf,
            max_index_buf,
            local_energy_buf,
            local_index_buf,
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

    if energy <= 0.0:
        # No improvement: we can exit early
        return 0.0, theta

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

        theta[i] = not theta[i]
        theta[j] = not theta[j]
    else:
        theta[s] = not theta[s]

    return energy, theta


def run_gray_search_opencl(
    n,
    kernel,
    theta,
    theta_buf,
    G_m_buf,
    is_segmented,
    local_size,
    global_size,
    args_buf,
    max_energy_host,
    max_theta_host,
    max_energy_buf,
    max_theta_buf,
):
    queue = opencl_context.queue

    # Set kernel args
    if is_segmented:
        kernel.set_args(
            G_m_buf[0],
            G_m_buf[1],
            G_m_buf[2],
            G_m_buf[3],
            theta_buf,
            args_buf,
            max_theta_buf,
            max_energy_buf,
        )
    else:
        kernel.set_args(G_m_buf, theta_buf, args_buf, max_theta_buf, max_energy_buf)

    cl.enqueue_nd_range_kernel(queue, kernel, (global_size,), (local_size,))

    # Read results
    cl.enqueue_copy(queue, max_energy_host, max_energy_buf)
    queue.finish()

    # Queue read for results we might not need
    cl.enqueue_copy(queue, max_theta_host, max_theta_buf)

    # Find global minimum
    best_x = np.argmax(max_energy_host)
    energy = max_energy_host[best_x]

    if energy <= 0.0:
        # No improvement: we can exit early
        return 0.0, theta

    # We need the best index
    queue.finish()

    blocks = (n + 63) // 64
    offset = best_x * blocks
    for b in range(blocks):
        s = max_theta_host[offset + b]
        b_offset = b << 6
        for i in range(64):
            j = b_offset + i
            if j >= n:
                break
            theta[j] = (s >> i) & 1

    return energy, theta


# Belief propagation was added by (Anthropic) Claude
def belief_propagation_marginals_dense(G_m, n, bp_scale=1.0, damping=0.5):
    """
    Run loopy belief propagation on the dense (upper-triangular) adjacency
    matrix to produce per-node marginals for warm-starting the partition.

    The dense solver stores G_m as a full n×n matrix; we read only the
    upper triangle (j > i) to match the sparse convention, so behaviour
    is consistent across all three solver variants.

    Parameters
    ----------
    G_m : ndarray, shape (n, n)
        Dense adjacency/weight matrix. May be full or upper-triangular.
    n : int
        Number of nodes.
    bp_scale : float
        Iteration cap = int(bp_scale * n). Gives O(n * m) total work,
        <= O(n^3) for any graph density. User-controlled.
    damping : float
        Message damping factor in [0, 1). Stabilises oscillating marginals
        on dense or heavily frustrated graphs.

    Returns
    -------
    marginals : ndarray, shape (n,)
        Soft assignment in (-1, +1). Positive values favour partition 1,
        negative values favour partition 0.
    """
    max_iterations = max(1, int(bp_scale * n))

    # Collect undirected edges from upper triangle
    us = []
    vs = []
    ws = []
    for i in range(n):
        for j in range(i + 1, n):
            w = G_m[i, j]
            if abs(w) > 1e-12:
                us.append(i)
                vs.append(j)
                ws.append(w)

    m = len(us)
    if m == 0:
        return np.zeros(n, dtype=np.float64)

    us = np.array(us, dtype=np.int32)
    vs = np.array(vs, dtype=np.int32)
    ws = np.array(ws, dtype=np.float64)

    # msg_fwd[k] = message from us[k] -> vs[k]
    # msg_bwd[k] = message from vs[k] -> us[k]
    msg_fwd = np.zeros(m, dtype=np.float64)
    msg_bwd = np.zeros(m, dtype=np.float64)

    # Build neighbour index: node -> list of (edge_idx, direction)
    # direction=0: node is 'u' end, incoming = msg_bwd[k]
    # direction=1: node is 'v' end, incoming = msg_fwd[k]
    node_edges = [[] for _ in range(n)]
    for k in range(m):
        node_edges[us[k]].append((k, 0))
        node_edges[vs[k]].append((k, 1))

    for _ in range(max_iterations):
        new_fwd = np.empty(m, dtype=np.float64)
        new_bwd = np.empty(m, dtype=np.float64)

        for k in range(m):
            u = us[k]
            v = vs[k]
            w = ws[k]

            # Message u -> v: aggregate incoming to u excluding v
            h_u = 0.0
            for (ek, direction) in node_edges[u]:
                if ek == k:
                    continue
                h_u += msg_bwd[ek] if direction == 0 else msg_fwd[ek]

            tanh_w = np.tanh(w)
            product = tanh_w * np.tanh(h_u)
            product = np.clip(product, -1.0 + 1e-9, 1.0 - 1e-9)
            new_fwd[k] = damping * msg_fwd[k] + (1.0 - damping) * np.arctanh(product)

            # Message v -> u: aggregate incoming to v excluding u
            h_v = 0.0
            for (ek, direction) in node_edges[v]:
                if ek == k:
                    continue
                h_v += msg_bwd[ek] if direction == 0 else msg_fwd[ek]

            product_v = tanh_w * np.tanh(h_v)
            product_v = np.clip(product_v, -1.0 + 1e-9, 1.0 - 1e-9)
            new_bwd[k] = damping * msg_bwd[k] + (1.0 - damping) * np.arctanh(product_v)

        msg_fwd = new_fwd
        msg_bwd = new_bwd

    # Marginals: sum all incoming messages per node
    marginals = np.zeros(n, dtype=np.float64)
    for k in range(m):
        marginals[us[k]] += msg_bwd[k]
        marginals[vs[k]] += msg_fwd[k]

    return np.tanh(marginals)


def bp_warm_start_dense(G_m, n, bp_scale=1.0, damping=0.5):
    """Partition bitstring from BP marginals over the dense adjacency matrix."""
    marginals = belief_propagation_marginals_dense(G_m, n, bp_scale, damping)
    return marginals > 0.0


def spin_glass_solver(
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
    gray_seed_multiple=None,
    bp_scale=None,
    bp_damping=0.5,
):
    """
    Dense spin glass / MaxCut solver with optional belief propagation warm-start.

    Parameters
    ----------
    bp_scale : float or None
        Controls BP iteration count as int(bp_scale * n_qubits).
        Default None disables BP. Set to 1.0 to enable with O(n*m) cost
        (<=O(n^3) for any graph density). Increase for more BP iterations
        at proportionally higher cost.
    bp_damping : float
        Message damping in [0, 1). Default 0.5. Higher values help
        convergence on dense or heavily frustrated graphs.
    """
    dtype = opencl_context.dtype
    nodes = None
    n_qubits = 0
    G_m = None
    if isinstance(G, nx.Graph):
        nodes = list(G.nodes())
        n_qubits = len(nodes)
        G_m = nx.to_numpy_array(G, weight="weight", nonedge=0.0, dtype=dtype)
    else:
        n_qubits = len(G)
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

    if n_qubits < heuristic_threshold:
        best_guess = None

    bitstring = ""
    cut_value = None
    if isinstance(best_guess, str):
        bitstring = best_guess
    elif isinstance(best_guess, int):
        bitstring = int_to_bitstring(best_guess, n_qubits)
    elif isinstance(best_guess, list):
        bitstring = "".join(["1" if b else "0" for b in best_guess])
    else:
        # BP warm-start: run before the sampling heuristic if bp_scale is set,
        # so that the cascade local search begins from a sign-aware partition
        # rather than the TFIM prior alone.
        if bp_scale is not None and n_qubits >= heuristic_threshold:
            bp_theta = bp_warm_start_dense(G_m, n_qubits, bp_scale, bp_damping)
            bitstring, cut_value, _ = maxcut_tfim(
                G_m,
                quality=quality,
                shots=shots,
                is_spin_glass=is_spin_glass,
                anneal_t=anneal_t,
                anneal_h=anneal_h,
                repulsion_base=repulsion_base,
                is_maxcut_gpu=is_maxcut_gpu,
                is_nested=True,
            )
            # Keep whichever of BP or sampling gave the better cut
            bp_energy = (
                compute_energy(bp_theta, G_m, n_qubits)
                if is_spin_glass
                else compute_cut(bp_theta, G_m, n_qubits)
            )
            sample_theta = np.array([b == "1" for b in list(bitstring)], dtype=np.bool_)
            sample_energy = (
                compute_energy(sample_theta, G_m, n_qubits)
                if is_spin_glass
                else compute_cut(sample_theta, G_m, n_qubits)
            )
            if bp_energy > sample_energy:
                bitstring = "".join(["1" if b else "0" for b in bp_theta])
                cut_value = bp_energy if not is_spin_glass else None
        else:
            bitstring, cut_value, _ = maxcut_tfim(
                G_m,
                quality=quality,
                shots=shots,
                is_spin_glass=is_spin_glass,
                anneal_t=anneal_t,
                anneal_h=anneal_h,
                repulsion_base=repulsion_base,
                is_maxcut_gpu=is_maxcut_gpu,
                is_nested=True,
            )

    best_theta = np.array([b == "1" for b in list(bitstring)], dtype=np.bool_)
    if is_spin_glass:
        max_energy = compute_energy(best_theta, G_m, n_qubits)
    elif cut_value is None:
        max_energy = compute_cut(best_theta, G_m, n_qubits)
    else:
        max_energy = cut_value

    if n_qubits < heuristic_threshold:
        bitstring, l, r = get_cut(best_theta, nodes, n_qubits)
        if is_spin_glass:
            cut_value = compute_cut(best_theta, G_m, n_qubits)
            min_energy = -max_energy
        else:
            cut_value = max_energy
            min_energy = compute_energy(best_theta, G_m, n_qubits)

        return bitstring, float(cut_value), (l, r), float(min_energy)

    if gray_iterations is None:
        gray_iterations = n_qubits * n_qubits

    if gray_seed_multiple is None:
        gray_seed_multiple = os.cpu_count()

    is_opencl = is_maxcut_gpu and IS_OPENCL_AVAILABLE

    if is_opencl:
        segment_size = (G_m.shape[0] * G_m.shape[1] + 3) >> 2
        is_segmented = (G_m.nbytes << 1) > opencl_context.max_alloc
        G_m_buf = make_G_m_buf(G_m, is_segmented, segment_size)

        local_work_group_size = min(wgs, n_qubits)
        global_work_group_size = n_qubits
        opencl_args = setup_opencl(
            local_work_group_size,
            global_work_group_size,
            np.array([n_qubits, is_spin_glass, segment_size]),
        )

        if is_segmented:
            single_bit_flips_kernel = opencl_context.single_bit_flips_segmented_kernel
            double_bit_flips_kernel = opencl_context.double_bit_flips_segmented_kernel
            gray_kernel = opencl_context.gray_segmented_kernel
        else:
            single_bit_flips_kernel = opencl_context.single_bit_flips_kernel
            double_bit_flips_kernel = opencl_context.double_bit_flips_kernel
            gray_kernel = opencl_context.gray_kernel

        if n_qubits <= gnl:
            gray_work_group_size = opencl_context.MAX_GPU_PROC_ELEM
            gray_args = setup_opencl(
                1,
                gray_work_group_size,
                np.array(
                    [
                        n_qubits,
                        is_spin_glass,
                        (gray_iterations + gray_work_group_size - 1) // gray_work_group_size,
                        segment_size,
                    ]
                ),
                True,
            )
            gray_kernel = opencl_context.gray_segmented_kernel if is_segmented else opencl_context.gray_kernel

    thread_count = os.cpu_count() ** 2
    improved = True
    while improved:
        improved = False

        # Single bit flips with O(n^2)
        if is_opencl:
            theta_buf = make_best_theta_buf(best_theta)
            energy, state = run_bit_flips_opencl(
                False,
                n_qubits,
                single_bit_flips_kernel,
                best_theta,
                theta_buf,
                G_m_buf,
                is_segmented,
                *opencl_args,
            )
        else:
            energy, state = run_single_bit_flips(best_theta, is_spin_glass, G_m)
        if energy > 0.0:
            max_energy += energy
            best_theta = state
            improved = True
            continue

        # Double bit flips with O(n^3)
        if is_opencl:
            # theta_buf has not changed
            energy, state = run_bit_flips_opencl(
                True,
                n_qubits,
                double_bit_flips_kernel,
                best_theta,
                theta_buf,
                G_m_buf,
                is_segmented,
                *opencl_args,
            )
        else:
            energy, state = run_double_bit_flips(best_theta, is_spin_glass, G_m, thread_count)
        if energy > 0.0:
            max_energy += energy
            best_theta = state
            improved = True
            continue

        if is_opencl and (n_qubits <= gnl):
            theta_buf_64 = make_best_theta_buf_64(best_theta)
            energy, state = run_gray_search_opencl(n_qubits, gray_kernel, best_theta, theta_buf_64, G_m_buf, is_segmented, *gray_args)
        else:
            # Gray code with default O(n^3)
            iterators, energy = pick_gray_seeds(best_theta, thread_count, gray_seed_multiple, G_m, n_qubits, is_spin_glass)
            state = iterators[0]
            if energy > 0.0:
                max_energy += energy
                best_theta = state
                improved = True
                continue

            energy, state = run_gray_optimization(best_theta, iterators, gray_iterations, thread_count, is_spin_glass, G_m)
        if energy > 0.0:
            max_energy += energy
            best_theta = state
            improved = True
            continue

        # Post-reheat phase
        reheat_theta = state.copy()
        reheat_energy = energy

        # Single bit flips with O(n^2)
        if is_opencl:
            theta_buf = make_best_theta_buf(reheat_theta)
            energy, state = run_bit_flips_opencl(
                False,
                n_qubits,
                single_bit_flips_kernel,
                reheat_theta,
                theta_buf,
                G_m_buf,
                is_segmented,
                *opencl_args,
            )
        else:
            energy, state = run_single_bit_flips(reheat_theta, is_spin_glass, G_m)
        if (energy + reheat_energy) > 0.0:
            max_energy += reheat_energy + energy
            best_theta = state
            improved = True
            continue

        if energy > 0.0:
            reheat_theta = state.copy()
            reheat_energy += energy

        # Double bit flips with O(n^3)
        if is_opencl:
            # theta_buf has not changed
            energy, state = run_bit_flips_opencl(
                True,
                n_qubits,
                double_bit_flips_kernel,
                reheat_theta,
                theta_buf,
                G_m_buf,
                is_segmented,
                *opencl_args,
            )
        else:
            energy, state = run_double_bit_flips(reheat_theta, is_spin_glass, G_m, thread_count)
        if (energy + reheat_energy) > 0.0:
            max_energy += reheat_energy + energy
            best_theta = state
            improved = True

    bitstring, l, r = get_cut(best_theta, nodes, n_qubits)
    if is_spin_glass:
        cut_value = compute_cut(best_theta, G_m, n_qubits)
        min_energy = -max_energy
    else:
        cut_value = max_energy
        min_energy = compute_energy(best_theta, G_m, n_qubits)

    return bitstring, float(cut_value), (l, r), float(min_energy)
