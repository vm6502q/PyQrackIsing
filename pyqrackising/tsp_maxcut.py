from .maxcut_tfim_util import binary_search, opencl_context, to_scipy_sparse_upper_triangular
from .tsp import tsp_symmetric, tsp_symmetric_sparse, tsp_symmetric_streaming
from .spin_glass_solver import spin_glass_solver
from .spin_glass_solver_sparse import spin_glass_solver_sparse
from .spin_glass_solver_streaming import spin_glass_solver_streaming
import networkx as nx
from numba import njit, prange


@njit(parallel=True)
def tsp_to_maxcut_bipartition(tsp_path, weights):
    n = len(tsp_path)
    best_cut_value = -float('inf')
    best_partition_A = None
    best_partition_B = None
    direction = 0

    for offset in [-1, 0, 1]:
        mid = n // 2 + offset
        A = tsp_path[:mid]
        B = tsp_path[mid:]
        cut_value = 0.0
        for i in prange(len(A)):
            u = A[i]
            for v in B:
                cut_value += weights[u, v]
        if cut_value > best_cut_value:
            best_cut_value = cut_value
            best_partition_A = A
            best_partition_B = B
            direction = offset

    if direction == 0:
        return best_partition_A, best_partition_B, best_cut_value

    improved = True
    best_offset = direction
    while improved:
        improved = False
        offset = best_offset + direction
        mid = n // 2 + offset
        A = tsp_path[:mid]
        B = tsp_path[mid:]
        cut_value = 0.0
        for i in prange(len(A)):
            u = A[i]
            for v in B:
                cut_value += weights[u, v]
        if cut_value > best_cut_value:
            best_cut_value = cut_value
            best_partition_A = A
            best_partition_B = B
            best_offset = offset
            improved = True

    return best_partition_A, best_partition_B, best_cut_value


@njit
def get_cut(bitstring, nodes):
    solution = list(bitstring)
    l, r = [], []
    for i in range(len(solution)):
        if solution[i] == "1":
            r.append(nodes[i])
        else:
            l.append(nodes[i])

    return l, r


@njit
def get_bitstring(partition, nodes):
    bitstring = ""
    for node in nodes:
        bitstring += "0" if node in partition[0] else "1"

    return bitstring


@njit(parallel=True)
def early_exit(G_m, partition, nodes):
    n_qubits = len(nodes)
    bitstring = get_bitstring(partition, nodes)
    theta_bits = [(b == "1") for b in list(bitstring)]
    energy = 0
    for u in prange(n_qubits):
        for v in range(u + 1, n_qubits):
            val = G_m[nodes[u], nodes[v]]
            energy += (val if theta_bits[u] == theta_bits[v] else -val)

    return bitstring, energy


def tsp_maxcut(G, k_neighbors=20, is_optimized=False, is_parallel=True, **kwargs):
    dtype = opencl_context.dtype
    G_m = None
    if isinstance(G, nx.Graph):
        nodes = list(G.nodes())
        n_qubits = len(nodes)
        G_m = nx.to_numpy_array(G, weight='weight', nonedge=0.0, dtype=dtype)
    else:
        n_qubits = len(G)
        nodes = list(range(n_qubits))
        G_m = G

    path, _ = tsp_symmetric(G_m, is_cyclic=False, monte_carlo=True, k_neighbors=k_neighbors, is_parallel=is_parallel)
    l, r, cut_value = tsp_to_maxcut_bipartition(path, G_m)

    if not is_optimized:
        bitstring, energy = early_exit(G_m, (l, r), nodes)

        return bitstring, cut_value, ([nodes[x] for x in l], [nodes[x] for x in r]), energy

    bitint = 0
    for b in partition[0]:
        bitint |= 1 << b

    bitstring, cut_value, cut, energy = spin_glass_solver(G_m, best_guess=bitint, **kwargs)
    l, r = get_cut(bitstring, nodes)

    return bitstring, cut_value, ([nodes[x] for x in l], [nodes[x] for x in r]), energy


@njit(parallel=True)
def tsp_to_maxcut_bipartition_sparse(tsp_path, G_data, G_rows, G_cols):
    n = len(tsp_path)
    best_cut_value = -float('inf')
    best_partition_A = None
    best_partition_B = None
    direction = 0

    for offset in [-1, 0, 1]:
        mid = n // 2 + offset
        A = tsp_path[:mid]
        B = tsp_path[mid:]
        cut_value = 0.0
        for i in prange(len(A)):
            u = A[i]
            for v in B:
                low, high = (u, v) if u < v else (v, u)
                start = G_rows[low]
                end = G_rows[low + 1]
                i = binary_search(G_cols[start:end], high) + start
                if i < end:
                    cut_value += G_data[i]
        if cut_value > best_cut_value:
            best_cut_value = cut_value
            best_partition_A = A
            best_partition_B = B
            direction = offset

    if direction == 0:
        return best_partition_A, best_partition_B, best_cut_value

    improved = True
    best_offset = direction
    while improved:
        improved = False
        offset = best_offset + direction
        mid = n // 2 + offset
        A = tsp_path[:mid]
        B = tsp_path[mid:]
        cut_value = 0.0
        for i in prange(len(A)):
            u = A[i]
            for v in B:
                low, high = (u, v) if u < v else (v, u)
                start = G_rows[low]
                end = G_rows[low + 1]
                i = binary_search(G_cols[start:end], high) + start
                if i < end:
                    cut_value += G_data[i]
        if cut_value > best_cut_value:
            best_cut_value = cut_value
            best_partition_A = A
            best_partition_B = B
            best_offset = offset
            improved = True

    return best_partition_A, best_partition_B, best_cut_value


@njit(parallel=True)
def early_exit_sparse(G_data, G_rows, G_cols, partition, nodes):
    n_qubits = len(nodes)
    bitstring = get_bitstring(partition, nodes)
    theta_bits = [(b == "1") for b in list(bitstring)]
    energy = 0
    for u in prange(n_qubits):
        for v in range(u + 1, n_qubits):
            start = G_rows[u]
            end = G_rows[u + 1]
            i = binary_search(G_cols[start:end], v) + start
            if i < end:
                val = G_data[i]
                energy += (val if theta_bits[u] == theta_bits[v] else -val)

    return bitstring, energy


def tsp_maxcut_sparse(G, k_neighbors=20, is_optimized=False, is_parallel=True, **kwargs):
    dtype = opencl_context.dtype
    G_m = None
    if isinstance(G, nx.Graph):
        nodes = list(G.nodes())
        n_qubits = len(nodes)
        G_m = to_scipy_sparse_upper_triangular(G, nodes, n_qubits, dtype)
    else:
        n_qubits = G.shape[0]
        nodes = list(range(n_qubits))
        G_m = G

    path, _ = tsp_symmetric_sparse(G_m, k_neighbors=k_neighbors, is_parallel=is_parallel)
    l, r, cut_value = tsp_to_maxcut_bipartition_sparse(path, G_m.data, G_m.indptr, G_m.indices)

    if not is_optimized:
        bitstring, energy = early_exit_sparse(G_m.data, G_m.indptr, G_m.indices, (l, r), nodes)

        return bitstring, cut_value, ([nodes[x] for x in l], [nodes[x] for x in r]), energy

    bitint = 0
    for b in partition[0]:
        bitint |= 1 << b

    bitstring, cut_value, cut, energy = spin_glass_solver_sparse(G_m, best_guess=bitint, **kwargs)
    l, r = get_cut(bitstring, nodes)

    return bitstring, cut_value, ([nodes[x] for x in l], [nodes[x] for x in r]), energy


@njit(parallel=True)
def tsp_to_maxcut_bipartition_streaming(tsp_path, G_func):
    n = len(tsp_path)
    best_cut_value = -float('inf')
    best_partition_A = None
    best_partition_B = None
    direction = 0

    for offset in [-1, 0, 1]:
        mid = n // 2 + offset
        A = tsp_path[:mid]
        B = tsp_path[mid:]
        cut_value = 0.0
        for i in prange(len(A)):
            u = A[i]
            for v in B:
                cut_value += G_func(u, v)
        if cut_value > best_cut_value:
            best_cut_value = cut_value
            best_partition_A = A
            best_partition_B = B
            direction = offset

    if direction == 0:
        return best_partition_A, best_partition_B, best_cut_value

    improved = True
    best_offset = direction
    while improved:
        improved = False
        offset = best_offset + direction
        mid = n // 2 + offset
        A = tsp_path[:mid]
        B = tsp_path[mid:]
        cut_value = 0.0
        for i in prange(len(A)):
            u = A[i]
            for v in B:
                cut_value += G_func(u, v)
        if cut_value > best_cut_value:
            best_cut_value = cut_value
            best_partition_A = A
            best_partition_B = B
            best_offset = offset
            improved = True

    return best_partition_A, best_partition_B, best_cut_value


@njit(parallel=True)
def early_exit_streaming(G_func, partition, nodes):
    n_qubits = len(nodes)
    bitstring = get_bitstring(partition, nodes)
    theta_bits = [(b == "1") for b in list(bitstring)]
    energy = 0
    for u in prange(n_qubits):
        for v in range(u + 1, n_qubits):
            val = G_func(nodes[u], nodes[v])
            energy += (val if theta_bits[u] == theta_bits[v] else -val)

    return bitstring, energy


def tsp_maxcut_streaming(G_func, nodes, k_neighbors=20, is_optimized=False, is_parallel=True, **kwargs):
    n_qubits = len(nodes)
    path, _ = tsp_symmetric_streaming(G_func, nodes, k_neighbors=k_neighbors, is_parallel=is_parallel)
    l, r, cut_value = tsp_to_maxcut_bipartition_streaming(path, G_func)

    if not is_optimized:
        bitstring, energy = early_exit_streaming(G_func, (l, r), nodes)

        return bitstring, cut_value, ([nodes[x] for x in l], [nodes[x] for x in r]), energy

    bitint = 0
    for b in partition[0]:
        bitint |= 1 << b

    bitstring, cut_value, cut, energy = spin_glass_solver_streaming(G_func, nodes, best_guess=bitint, **kwargs)
    l, r = get_cut(bitstring, nodes)

    return bitstring, cut_value, ([nodes[x] for x in l], [nodes[x] for x in r]), energy
