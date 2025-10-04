from .maxcut_tfim_util import to_scipy_sparse_upper_triangular
from .tsp import tsp_symmetric, tsp_symmetric_sparse, tsp_symmetric_streaming
from .spin_glass_solver import spin_glass_solver
from .spin_glass_solver_sparse import spin_glass_solver_sparse
from .spin_glass_solver_streaming import spin_glass_solver_streaming
import networkx as nx


def tsp_to_maxcut_bipartition(tsp_path, weights):
    n = len(tsp_path)
    best_cut_value = -float('inf')
    best_partition = None
    direction = 0

    for offset in [-1, 0, 1]:
        mid = n // 2 + offset
        A = tsp_path[:mid]
        B = tsp_path[mid:]
        cut_value = sum(
            weights[u, v]
            for u in A
            for v in B
        )
        if cut_value > best_cut_value:
            best_cut_value = cut_value
            best_partition = (A, B)
            direction = offset

    if direction == 0:
        return best_partition, best_cut_value

    improved = True
    best_offset = direction
    while improved:
        improved = False
        offset = best_offset + direction
        mid = n // 2 + offset
        A = tsp_path[:mid]
        B = tsp_path[mid:]
        cut_value = sum(
            weights[u, v]
            for u in A
            for v in B
        )
        if cut_value > best_cut_value:
            best_cut_value = cut_value
            best_partition = (A, B)
            best_offset = offset
            improved = True

    return best_partition, best_cut_value

def get_cut(bitstring, nodes):
    solution = list(bitstring)
    l, r = [], []
    for i in range(len(solution)):
        if solution[i] == "1":
            r.append(nodes[i])
        else:
            l.append(nodes[i])

    return l, r

def tsp_maxcut(G, k_neighbors=20, **kwargs):
    G_m = None
    if isinstance(G, nx.Graph):
        nodes = list(G.nodes())
        n_qubits = len(nodes)
        G_m = nx.to_numpy_array(G, weight='weight', nonedge=0.0, dtype=dtype)
    else:
        n_qubits = len(G)
        nodes = list(range(n_qubits))
        G_m = G

    path, length = tsp_symmetric(G, is_cyclic=False, monte_carlo=True, k_neighbors=k_neighbors)
    partition, cut_value = tsp_to_maxcut_bipartition(path, G_m)

    bitint = 0
    for b in partition[0]:
        bitint |= 1 << b

    bitstring, cut_value, cut, energy = spin_glass_solver(G_m, best_guess=bitint, **kwargs)
    l, r = get_cut(bitstring, nodes)

    return bitstring, cut_value, (l, r), energy


def tsp_maxcut_sparse(G, k_neighbors=20, **kwargs):
    G_m = None
    if isinstance(G, nx.Graph):
        nodes = list(G.nodes())
        n_qubits = len(nodes)
        G_m = to_scipy_sparse_upper_triangular(G, nodes, n_qubits, dtype)
    else:
        n_qubits = G.shape[0]
        nodes = list(range(n_qubits))
        G_m = G

    path, length = tsp_symmetric_sparse(G, k_neighbors=k_neighbors)
    partition, cut_value = tsp_to_maxcut_bipartition(path, G_m)

    bitint = 0
    for b in partition[0]:
        bitint |= 1 << b

    bitstring, cut_value, cut, energy = spin_glass_solver_sparse(G_m, best_guess=bitint, **kwargs)
    l, r = get_cut(bitstring, nodes)

    return bitstring, cut_value, (l, r), energy


def tsp_to_maxcut_bipartition_streaming(tsp_path, G_func):
    n = len(tsp_path)
    best_cut_value = -float('inf')
    best_partition = None
    direction = 0

    for offset in [-1, 0, 1]:
        mid = n // 2 + offset
        A = tsp_path[:mid]
        B = tsp_path[mid:]
        cut_value = sum(
            G_func(u, v)
            for u in A
            for v in B
        )
        if cut_value > best_cut_value:
            best_cut_value = cut_value
            best_partition = (A, B)
            direction = offset

    if direction == 0:
        return best_partition, best_cut_value

    improved = True
    best_offset = direction
    while improved:
        improved = False
        offset = best_offset + direction
        mid = n // 2 + offset
        A = tsp_path[:mid]
        B = tsp_path[mid:]
        cut_value = sum(
            G_func(u, v)
            for u in A
            for v in B
        )
        if cut_value > best_cut_value:
            best_cut_value = cut_value
            best_partition = (A, B)
            best_offset = offset
            improved = True

    return best_partition, best_cut_value


def tsp_maxcut_streaming(G_func, nodes, k_neighbors=20, **kwargs):
    path, length = tsp_symmetric_streaming(G_func, nodes, k_neighbors=k_neighbors)
    partition, cut_value = tsp_to_maxcut_bipartition_streaming(path, G_func)

    bitint = 0
    for b in partition[0]:
        bitint |= 1 << b

    bitstring, cut_value, cut, energy = spin_glass_solver_streaming(G_func, nodes, best_guess=bitint, **kwargs)
    l, r = get_cut(bitstring, nodes)

    return bitstring, cut_value, (l, r), energy
