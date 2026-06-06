from .maxcut_tfim_sparse import maxcut_tfim_sparse
from .maxcut_tfim_util import (
    compute_cut_sparse,
    compute_energy_sparse,
    get_cut,
    gray_code_next,
    gray_mutation,
    heuristic_threshold_sparse,
    int_to_bitstring,
    make_G_m_csr_buf,
    make_best_theta_buf,
    make_best_theta_buf_64,
    opencl_context,
    setup_opencl,
    to_scipy_sparse_upper_triangular,
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


@njit(parallel=True, cache=True)
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


@njit(parallel=True, cache=True)
def pick_gray_seeds(best_theta, thread_count, gray_seed_multiple, G_data, G_rows, G_cols, n, is_spin_glass):
    blocks = (n + 63) >> 6
    block_size = thread_count * gray_seed_multiple
    seed_count = block_size * blocks

    seeds = np.empty((seed_count, n), dtype=np.bool_)
    energies = np.empty(seed_count, dtype=dtype)

    if is_spin_glass:
        for s in prange(seed_count):
            i = s % block_size
            offset = (s // block_size) << 6
            seed = gray_mutation(i, best_theta, offset)
            energies[s] = compute_energy_sparse(seed, G_data, G_rows, G_cols, n)
            seeds[s] = seed
    else:
        for s in prange(seed_count):
            i = s % block_size
            offset = (s // block_size) << 6
            seed = gray_mutation(i, best_theta, offset)
            energies[s] = compute_cut_sparse(seed, G_data, G_rows, G_cols, n)
            seeds[s] = seed

    indices = np.argpartition(energies, -thread_count)[-thread_count:]
    indices = indices[np.argsort(energies[indices])[::-1]]
    best_seeds = np.empty((thread_count, n), dtype=np.bool_)
    best_energies = np.empty(thread_count, dtype=dtype)
    for i in prange(thread_count):
        idx = indices[i]
        best_seeds[i] = seeds[idx]
        best_energies[i] = energies[idx]

    return best_seeds, best_energies


@njit(parallel=True, cache=True)
def run_gray_optimization(
    best_theta,
    iterators,
    energies,
    gray_iterations,
    thread_count,
    is_spin_glass,
    G_data,
    G_rows,
    G_cols,
):
    n = len(best_theta)
    thread_iterations = (gray_iterations + thread_count - 1) // thread_count
    blocks = (n + 63) >> 6

    if is_spin_glass:
        for i in prange(thread_count):
            iterator = iterators[i]
            best_energy = energies[i]
            for curr_idx in range(thread_iterations):
                for block in range(blocks):
                    flip_bit = gray_code_next(iterator, curr_idx, block << 6)
                    energy = compute_energy_sparse(iterator, G_data, G_rows, G_cols, n)
                    if energy > best_energy:
                        best_energy = energy
                    else:
                        # Revert iterator
                        iterator[flip_bit] = not iterator[flip_bit]
                if best_energy > energies[i]:
                    energies[i] = best_energy
    else:
        for i in prange(thread_count):
            iterator = iterators[i]
            best_energy = energies[i]
            for curr_idx in range(thread_iterations):
                for block in range(blocks):
                    flip_bit = gray_code_next(iterator, curr_idx, block << 6)
                    energy = compute_cut_sparse(iterator, G_data, G_rows, G_cols, n)
                    if energy > best_energy:
                        best_energy = energy
                    else:
                        # Revert iterator
                        iterator[flip_bit] = not iterator[flip_bit]
                if best_energy > energies[i]:
                    energies[i] = best_energy

    best_index = np.argmax(energies)
    best_energy = energies[best_index]
    best_state = iterators[best_index]

    return best_energy, best_state


def run_bit_flips_opencl(
    is_double,
    n,
    kernel,
    best_energy,
    theta,
    theta_buf,
    G_data_buf,
    G_rows_buf,
    G_cols_buf,
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
            local_index_buf,
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

        theta[i] = not theta[i]
        theta[j] = not theta[j]
    else:
        theta[s] = not theta[s]

    return energy, theta


def run_gray_search_opencl(
    n,
    kernel,
    best_energy,
    theta,
    theta_buf,
    G_data_buf,
    G_rows_buf,
    G_cols_buf,
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
            G_data_buf[0],
            G_data_buf[1],
            G_data_buf[2],
            G_data_buf[3],
            G_rows_buf,
            G_cols_buf,
            theta_buf,
            args_buf,
            max_theta_buf,
            max_energy_buf,
        )
    else:
        kernel.set_args(G_data_buf, G_rows_buf, G_cols_buf, theta_buf, args_buf, max_theta_buf, max_energy_buf)

    cl.enqueue_nd_range_kernel(queue, kernel, (global_size,), (local_size,))

    # Read results
    cl.enqueue_copy(queue, max_energy_host, max_energy_buf)
    queue.finish()

    # Queue read for results we might not need
    cl.enqueue_copy(queue, max_theta_host, max_theta_buf)

    # Find global minimum
    best_x = np.argmax(max_energy_host)
    energy = max_energy_host[best_x]

    if energy <= best_energy:
        # No improvement: we can exit early
        return best_energy, theta

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
def belief_propagation_marginals(G_data, G_rows, G_cols, n, bp_scale=1.0, damping=0.5):
    """
    Run loopy belief propagation on the signed graph to produce per-node
    marginals suitable for biasing the initial partition assignment.

    Messages are defined on directed edges (i->j) in the MaxCut factor graph.
    Since the CSR matrix is upper-triangular, we iterate over stored edges and
    maintain both directions explicitly.

    Parameters
    ----------
    G_data : ndarray
        Edge weights from upper-triangular CSR.
    G_rows : ndarray
        CSR row pointers.
    G_cols : ndarray
        CSR column indices.
    n : int
        Number of nodes.
    bp_scale : float
        Iteration cap = int(bp_scale * n). Default 1.0 keeps total work
        O(n * m), which is <= O(n^3) for any graph density. User-controlled.
    damping : float
        Message damping factor in [0, 1). Higher values slow convergence
        but stabilise oscillating marginals on dense frustrated graphs.

    Returns
    -------
    marginals : ndarray, shape (n,)
        Soft assignment in (-1, +1). Positive values favour partition 1,
        negative values favour partition 0.
    """
    max_iterations = max(1, int(bp_scale * n))

    # Messages: msg[i, j] = current message from i to j.
    # We store all directed edges as two dicts indexed by (src, dst).
    # For memory efficiency on sparse graphs we use flat arrays parallel
    # to the CSR structure, holding both directions.

    # Number of undirected edges
    m = G_data.shape[0]

    # For each undirected edge k: (u, v) with u < v (upper triangular)
    # we store two directed messages: fwd[k] = msg u->v, bwd[k] = msg v->u
    msg_fwd = np.zeros(m, dtype=np.float64)  # u -> v
    msg_bwd = np.zeros(m, dtype=np.float64)  # v -> u

    # Build a reverse index: for each node, list of (edge_idx, direction)
    # direction=0 means node is the 'u' end (row), direction=1 means 'v' end (col)
    node_edges = [[] for _ in range(n)]
    edge_u = np.empty(m, dtype=np.int32)
    edge_v = np.empty(m, dtype=np.int32)

    k = 0
    for u in range(n):
        for ptr in range(G_rows[u], G_rows[u + 1]):
            v = G_cols[ptr]
            edge_u[k] = u
            edge_v[k] = v
            node_edges[u].append((k, 0))  # u is src of fwd message
            node_edges[v].append((k, 1))  # v is src of bwd message
            k += 1

    for _ in range(max_iterations):
        new_fwd = np.empty(m, dtype=np.float64)
        new_bwd = np.empty(m, dtype=np.float64)

        for k in range(m):
            u = edge_u[k]
            v = edge_v[k]
            w = G_data[k]

            # Message u -> v: sum of incoming messages to u from all
            # neighbours except v, passed through the edge factor.
            # For MaxCut, the factor encourages u and v to differ;
            # for negative edges it encourages them to agree.
            # The update rule: new_msg(u->v) = atanh(tanh(w) * tanh(sum_excl))
            # where sum_excl is the sum of all incoming messages to u except
            # the one from v.
            h_u = 0.0
            for (ek, direction) in node_edges[u]:
                if ek == k:
                    continue  # exclude the v->u message
                if direction == 0:
                    # u is the fwd src on edge ek, so incoming to u is bwd[ek]
                    h_u += msg_bwd[ek]
                else:
                    # u is the bwd src on edge ek, so incoming to u is fwd[ek]
                    h_u += msg_fwd[ek]

            tanh_w = np.tanh(w)
            tanh_h = np.tanh(h_u)
            product = tanh_w * tanh_h
            # Clamp to avoid atanh singularity
            product = np.clip(product, -1.0 + 1e-9, 1.0 - 1e-9)
            raw_fwd = np.arctanh(product)
            new_fwd[k] = damping * msg_fwd[k] + (1.0 - damping) * raw_fwd

            # Message v -> u (symmetric, swap roles)
            h_v = 0.0
            for (ek, direction) in node_edges[v]:
                if ek == k:
                    continue  # exclude the u->v message
                if direction == 0:
                    h_v += msg_bwd[ek]
                else:
                    h_v += msg_fwd[ek]

            tanh_h_v = np.tanh(h_v)
            product_v = tanh_w * tanh_h_v
            product_v = np.clip(product_v, -1.0 + 1e-9, 1.0 - 1e-9)
            raw_bwd = np.arctanh(product_v)
            new_bwd[k] = damping * msg_bwd[k] + (1.0 - damping) * raw_bwd

        msg_fwd = new_fwd
        msg_bwd = new_bwd

    # Compute marginals: for each node, sum all incoming messages
    marginals = np.zeros(n, dtype=np.float64)
    for k in range(m):
        u = edge_u[k]
        v = edge_v[k]
        marginals[u] += msg_bwd[k]  # incoming to u from v
        marginals[v] += msg_fwd[k]  # incoming to v from u

    return np.tanh(marginals)


def bp_warm_start(G_data, G_rows, G_cols, n, bp_scale=1.0, damping=0.5):
    """
    Produce an initial partition bitstring from BP marginals.
    Nodes with positive marginal go to partition 1, negative to partition 0.
    Falls back to all-zeros (trivial) if BP produces a degenerate result.
    """
    marginals = belief_propagation_marginals(G_data, G_rows, G_cols, n, bp_scale, damping)
    theta = marginals > 0.0
    return theta


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
    gray_seed_multiple=None,
    bp_scale=None,
    bp_damping=0.5,
):
    """
    Spin glass / MaxCut solver with optional belief propagation warm-start.

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

    if n_qubits < heuristic_threshold_sparse:
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
        if bp_scale is not None and n_qubits >= heuristic_threshold_sparse:
            bp_theta = bp_warm_start(
                G_m.data, G_m.indptr, G_m.indices, n_qubits, bp_scale, bp_damping
            )
            bp_bitstring = "".join(["1" if b else "0" for b in bp_theta])
            # Pass BP result as best_guess into maxcut_tfim_sparse so the
            # sampling heuristic can compete against it or build on it.
            bitstring, cut_value, _ = maxcut_tfim_sparse(
                G_m,
                quality=quality,
                shots=shots,
                is_spin_glass=is_spin_glass,
                anneal_t=anneal_t,
                anneal_h=anneal_h,
                repulsion_base=repulsion_base,
            )
            # Keep whichever of BP or sampling gave the better cut
            bp_energy = (
                compute_energy_sparse(bp_theta, G_m.data, G_m.indptr, G_m.indices, n_qubits)
                if is_spin_glass
                else compute_cut_sparse(bp_theta, G_m.data, G_m.indptr, G_m.indices, n_qubits)
            )
            sample_theta = np.array([b == "1" for b in list(bitstring)], dtype=np.bool_)
            sample_energy = (
                compute_energy_sparse(sample_theta, G_m.data, G_m.indptr, G_m.indices, n_qubits)
                if is_spin_glass
                else compute_cut_sparse(sample_theta, G_m.data, G_m.indptr, G_m.indices, n_qubits)
            )
            if bp_energy > sample_energy:
                bitstring = bp_bitstring
                cut_value = bp_energy if not is_spin_glass else None
        else:
            bitstring, cut_value, _ = maxcut_tfim_sparse(
                G_m,
                quality=quality,
                shots=shots,
                is_spin_glass=is_spin_glass,
                anneal_t=anneal_t,
                anneal_h=anneal_h,
                repulsion_base=repulsion_base,
            )

    best_theta = np.array([b == "1" for b in list(bitstring)], dtype=np.bool_)
    if is_spin_glass:
        max_energy = compute_energy_sparse(best_theta, G_m.data, G_m.indptr, G_m.indices, n_qubits)
    elif cut_value is None:
        max_energy = compute_cut_sparse(best_theta, G_m.data, G_m.indptr, G_m.indices, n_qubits)
    else:
        max_energy = cut_value

    if n_qubits < heuristic_threshold_sparse:
        bitstring, l, r = get_cut(best_theta, nodes, n_qubits)
        if is_spin_glass:
            cut_value = compute_cut_sparse(best_theta, G_m.data, G_m.indptr, G_m.indices, n_qubits)
            min_energy = -max_energy
        else:
            cut_value = max_energy
            min_energy = compute_energy_sparse(best_theta, G_m.data, G_m.indptr, G_m.indices, n_qubits)

        return bitstring, float(cut_value), (l, r), float(min_energy)

    if gray_iterations is None:
        gray_iterations = n_qubits * n_qubits

    if gray_seed_multiple is None:
        gray_seed_multiple = os.cpu_count()

    is_opencl = is_maxcut_gpu and IS_OPENCL_AVAILABLE

    if is_opencl:
        segment_size = (G_m.data.shape[0] + 3) >> 2
        is_segmented = (G_m.data.nbytes << 1) > opencl_context.max_alloc
        G_data_buf, G_rows_buf, G_cols_buf = make_G_m_csr_buf(G_m, is_segmented, segment_size)

        local_work_group_size = min(wgs, n_qubits)
        global_work_group_size = n_qubits
        opencl_args = setup_opencl(
            local_work_group_size,
            global_work_group_size,
            np.array([n_qubits, is_spin_glass, segment_size]),
        )

        if is_segmented:
            single_bit_flips_kernel = opencl_context.single_bit_sparse_flips_segmented_kernel
            double_bit_flips_kernel = opencl_context.double_bit_sparse_flips_segmented_kernel
        else:
            single_bit_flips_kernel = opencl_context.single_bit_flips_sparse_kernel
            double_bit_flips_kernel = opencl_context.double_bit_flips_sparse_kernel

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
            gray_kernel = opencl_context.gray_sparse_segmented_kernel if is_segmented else opencl_context.gray_sparse_kernel

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
                max_energy,
                best_theta,
                theta_buf,
                G_data_buf,
                G_rows_buf,
                G_cols_buf,
                is_segmented,
                *opencl_args,
            )
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
            energy, state = run_bit_flips_opencl(
                True,
                n_qubits,
                double_bit_flips_kernel,
                max_energy,
                best_theta,
                theta_buf,
                G_data_buf,
                G_rows_buf,
                G_cols_buf,
                is_segmented,
                *opencl_args,
            )
        else:
            energy, state = run_double_bit_flips(best_theta, is_spin_glass, G_m.data, G_m.indptr, G_m.indices, thread_count)
        if energy > max_energy:
            max_energy = energy
            best_theta = state
            improved = True
            continue

        if is_opencl and (n_qubits <= gnl):
            theta_buf_64 = make_best_theta_buf_64(best_theta)
            energy, state = run_gray_search_opencl(
                n_qubits,
                gray_kernel,
                max_energy,
                best_theta,
                theta_buf_64,
                G_data_buf,
                G_rows_buf,
                G_cols_buf,
                is_segmented,
                *gray_args,
            )
        else:
            # Gray code with default O(n^3)
            iterators, energies = pick_gray_seeds(
                best_theta,
                thread_count,
                gray_seed_multiple,
                G_m.data,
                G_m.indptr,
                G_m.indices,
                n_qubits,
                is_spin_glass,
            )
            energy, state = energies[0], iterators[0]
            if energy > max_energy:
                max_energy = energy
                best_theta = state
                improved = True
                continue

            energy, state = run_gray_optimization(
                best_theta,
                iterators,
                energies,
                gray_iterations,
                thread_count,
                is_spin_glass,
                G_m.data,
                G_m.indptr,
                G_m.indices,
            )
        if energy > max_energy:
            max_energy = energy
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
                max_energy,
                reheat_theta,
                theta_buf,
                G_data_buf,
                G_rows_buf,
                G_cols_buf,
                is_segmented,
                *opencl_args,
            )
        else:
            energy, state = run_single_bit_flips(reheat_theta, is_spin_glass, G_m.data, G_m.indptr, G_m.indices)
        if energy > max_energy:
            max_energy = energy
            best_theta = state
            improved = True
            continue

        if energy > reheat_energy:
            reheat_theta = state.copy()

        # Double bit flips with O(n^3)
        if is_opencl:
            # theta_buf has not changed
            energy, state = run_bit_flips_opencl(
                True,
                n_qubits,
                double_bit_flips_kernel,
                max_energy,
                reheat_theta,
                theta_buf,
                G_data_buf,
                G_rows_buf,
                G_cols_buf,
                is_segmented,
                *opencl_args,
            )
        else:
            energy, state = run_double_bit_flips(reheat_theta, is_spin_glass, G_m.data, G_m.indptr, G_m.indices, thread_count)
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
