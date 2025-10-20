from .maxcut_tfim_util import binary_search, opencl_context, to_scipy_sparse_upper_triangular
from .maxcut_tfim import maxcut_tfim
from concurrent.futures import ProcessPoolExecutor
import time
import itertools
import networkx as nx
from numba import njit, prange
import numpy as np
import os
from scipy.sparse import lil_matrix


n_threads = os.cpu_count()
max_parallel_level = np.log2(n_threads)


# two_opt() and targeted_three_opt() written by Elara (OpenAI ChatGPT instance)
@njit
def path_length(path, G_m):
    tot_len = 0.0
    for i in range(len(path) - 1):
        tot_len += G_m[path[i], path[i + 1]]

    return tot_len


@njit
def one_way_two_opt(best_path, G):
    improved = True
    best_dist = path_length(best_path, G)
    path_len = len(best_path)

    while improved:
        improved = False
        for i in range(1, path_len - 1):
            for j in range(i + 2, path_len):
                new_path = best_path[:]
                new_path[i:j] = best_path[j-1:i-1:-1]
                new_dist = path_length(new_path, G)
                if new_dist < best_dist:
                    best_path, best_dist = new_path, new_dist
                    improved = True

    return best_path, best_dist


@njit(parallel=True)
def one_way_two_opt_parallel(best_path, G):
    improved = True
    best_dist = path_length(best_path, G)
    paths = [best_path] * n_threads
    dists = [best_dist] * n_threads
    path_len = len(best_path)

    while improved:
        improved = False
        for i in range(1, path_len - 1):
            for j in range(i + 2, path_len, n_threads):
                max_k = min(n_threads, path_len - j)
                for k in prange(max_k):
                    l = j + k
                    new_path = best_path[:]
                    new_path[i:l] = best_path[l-1:i-1:-1]
                    dists[k] = path_length(new_path, G)
                    paths[k] = new_path
                for k in range(max_k):
                    if dists[k] < best_dist:
                        best_path, best_dist, improved = paths[k], dists[k], True

    return best_path, best_dist


@njit
def anchored_two_opt(best_path, G):
    improved = True
    best_dist = path_length(best_path, G)
    path_len = len(best_path)

    while improved:
        improved = False
        for i in range(2, path_len - 1):
            for j in range(i + 2, path_len):
                new_path = best_path[:]
                new_path[i:j] = best_path[j-1:i-1:-1]
                new_dist = path_length(new_path, G)
                if new_dist < best_dist:
                    best_path, best_dist = new_path, new_dist
                    improved = True

    return best_path, best_dist


@njit(parallel=True)
def anchored_two_opt_parallel(best_path, G):
    improved = True
    best_dist = path_length(best_path, G)
    paths = [best_path] * n_threads
    dists = [best_dist] * n_threads
    path_len = len(best_path)

    while improved:
        improved = False
        for i in range(2, path_len - 1):
            for j in range(i + 2, path_len, n_threads):
                max_k = min(n_threads, path_len - j)
                for k in prange(max_k):
                    l = j + k
                    new_path = best_path[:]
                    new_path[i:l] = best_path[l-1:i-1:-1]
                    dists[k] = path_length(new_path, G)
                    paths[k] = new_path
                for k in range(max_k):
                    if dists[k] < best_dist:
                        best_path, best_dist, improved = paths[k], dists[k], True

    return best_path, best_dist


@njit
def two_opt(best_path, G):
    improved = True
    best_dist = path_length(best_path, G)
    path_len = len(best_path)

    while improved:
        improved = False
        for i in range(1, path_len - 2):
            for j in range(i + 2, path_len - 1):
                new_path = best_path[:]
                new_path[i:j] = best_path[j-1:i-1:-1]  # reverse segment
                new_dist = path_length(new_path, G)
                if new_dist < best_dist:
                    best_path, best_dist = new_path, new_dist
                    improved = True

    return best_path, best_dist


@njit(parallel=True)
def two_opt_parallel(best_path, G):
    improved = True
    best_dist = path_length(best_path, G)
    paths = [best_path] * n_threads
    dists = [best_dist] * n_threads
    path_len = len(best_path)

    while improved:
        improved = False
        for i in range(1, path_len - 2):
            for j in range(i + 2, path_len - 1, n_threads):
                max_k = min(n_threads, path_len - (j + 1))
                for k in prange(max_k):
                    l = j + k
                    new_path = best_path[:]
                    new_path[i:l] = best_path[l-1:i-1:-1]
                    dists[k] = path_length(new_path, G)
                    paths[k] = new_path
                for k in range(max_k):
                    if dists[k] < best_dist:
                        best_path, best_dist, improved = paths[k], dists[k], True

    return best_path, best_dist


@njit
def targeted_three_opt(path, W, neighbor_lists, k_neighbors=20):
    """
    Lin-Kernighan style 3-opt heuristic for TSP improvement.

    path: list of node indices (tour)
    W: adjacency matrix (numpy array)
    k_neighbors: restrict 3-opt to nearest neighbors for efficiency
    """

    n = len(path)
    best_path = path[:]
    best_dist = path_length(best_path, W)
    improved = True

    while improved:
        improved = False
        for i in range(n - 5):
            for j in neighbor_lists[path[i]]:
                if j <= i or j >= n-3:
                    continue
                for k in neighbor_lists[path[j]]:
                    if k <= j or k >= n-1:
                        continue

                    # 7 unique cases (same as brute force, but restricted)
                    new_path = best_path[:i+1] + best_path[i+1:j+1][::-1] + best_path[j+1:]
                    dist = path_length(new_path, W)
                    if dist < best_dist:
                        best_path, best_dist, improved = new_path, dist, True
                        break

                    new_path = best_path[:j+1] + best_path[j+1:k+1][::-1] + best_path[k+1:]
                    dist = path_length(new_path, W)
                    if dist < best_dist:
                        best_path, best_dist, improved = new_path, dist, True
                        break

                    new_path = best_path[:i+1] + best_path[i+1:j+1][::-1] + best_path[j+1:k+1][::-1] + best_path[k+1:]
                    dist = path_length(new_path, W)
                    if dist < best_dist:
                        best_path, best_dist, improved = new_path, dist, True
                        break

                    new_path = best_path[:i+1] + best_path[j+1:k+1] + best_path[i+1:j+1] + best_path[k+1:]
                    dist = path_length(new_path, W)
                    if dist < best_dist:
                        best_path, best_dist, improved = new_path, dist, True
                        break

                    new_path = best_path[:i+1] + best_path[j+1:k+1][::-1] + best_path[i+1:j+1][::-1] + best_path[k+1:]
                    dist = path_length(new_path, W)
                    if dist < best_dist:
                        best_path, best_dist, improved = new_path, dist, True
                        break

                    new_path = best_path[:i+1] + best_path[j+1:k+1] + best_path[i+1:j+1][::-1] + best_path[k+1:]
                    dist = path_length(new_path, W)
                    if dist < best_dist:
                        best_path, best_dist, improved = new_path, dist, True
                        break

                    new_path = best_path[:i+1] + best_path[j+1:k+1][::-1] + best_path[i+1:j+1] + best_path[k+1:]
                    dist = path_length(new_path, W)
                    if dist < best_dist:
                        best_path, best_dist, improved = new_path, dist, True
                        break

                if improved:
                    break

            if improved:
                break

        path = best_path[:]

    return best_path, best_dist


@njit(parallel=True)
def targeted_three_opt_parallel(path, W, neighbor_lists, k_neighbors=20):
    n = len(path)
    best_path = path[:]
    best_dist = path_length(best_path, W)
    paths = [best_path] * 7
    dists = [best_dist] * 7
    improved = True

    while improved:
        improved = False
        for i in range(n - 5):
            for _j in range(k_neighbors):
                j = neighbor_lists[path[i], _j]
                if j <= i or j >= n-3:
                    continue
                for _k in range(k_neighbors):
                    k = neighbor_lists[path[j], _k]
                    if k <= j or k >= n-1:
                        continue

                    # 7 unique cases (same as brute force, but restricted)
                    for l in prange(7):
                        match l:
                            case 0:
                                new_path = best_path[:i+1] + best_path[i+1:j+1][::-1] + best_path[j+1:]
                            case 1:
                                new_path = best_path[:j+1] + best_path[j+1:k+1][::-1] + best_path[k+1:]
                            case 2:
                                new_path = best_path[:i+1] + best_path[i+1:j+1][::-1] + best_path[j+1:k+1][::-1] + best_path[k+1:]
                            case 3:
                                new_path = best_path[:i+1] + best_path[j+1:k+1] + best_path[i+1:j+1] + best_path[k+1:]
                            case 4:
                                new_path = best_path[:i+1] + best_path[j+1:k+1][::-1] + best_path[i+1:j+1][::-1] + best_path[k+1:]
                            case 5:
                                new_path = best_path[:i+1] + best_path[j+1:k+1] + best_path[i+1:j+1][::-1] + best_path[k+1:]
                            case _:
                                new_path = best_path[:i+1] + best_path[j+1:k+1][::-1] + best_path[i+1:j+1] + best_path[k+1:]

                        dists[l] = path_length(new_path, W)
                        paths[l] = new_path

                    for l in range(7):
                        if dists[l] < best_dist:
                            best_path, best_dist, improved = paths[l], dists[l], True

                    if improved:
                        break

                if improved:
                    break

            if improved:
                break

        path = best_path[:]

    return best_path, best_dist


@njit
def init_G_a_b(G_m, a, b):
    n_a_nodes = len(a)
    n_b_nodes = len(b)
    G_a = np.zeros((n_a_nodes, n_a_nodes), dtype=np.float64)
    G_b = np.zeros((n_b_nodes, n_b_nodes), dtype=np.float64)
    for i in range(n_a_nodes):
        for j in range(n_a_nodes):
            if i == j:
                continue
            G_a[i, j] = G_m[a[i], a[j]]
    for i in range(n_b_nodes):
        for j in range(n_b_nodes):
            if i == j:
                continue
            G_b[i, j] = G_m[b[i], b[j]]

    return G_a, G_b


@njit
def stich_singlet(G_m, singlet, bulk):
    best_path = bulk.copy()
    best_weight = G_m[singlet, bulk[0]]
    weight = G_m[singlet, bulk[-1]]
    if weight < best_weight:
        best_weight = weight
        best_path += [singlet]
    else:
        best_path = [singlet] + best_path

    for i in range(1, len(bulk)):
        weight = (
            G_m[singlet, bulk[i - 1]] +
            G_m[singlet, bulk[i]] -
            G_m[bulk[i - 1], bulk[i]]
        )
        if weight < best_weight:
            best_weight = weight
            best_path = bulk.copy()
            best_path.insert(i, singlet)

    return best_path

@njit
def stitch_symmetric(G_m, path_a, path_b):
    if len(path_a) == 1:
        return stich_singlet(G_m, path_a[0], path_b)

    if len(path_b) == 1:
        return stich_singlet(G_m, path_b[0], path_a)

    terminals_a = [path_a[0], path_a[-1]]
    terminals_b = [path_b[0], path_b[-1]]

    best_connect = G_m[terminals_a[1], terminals_b[0]]
    best_path = path_b.copy()
    weight = G_m[terminals_a[0], terminals_b[1]]
    if weight < best_connect:
        best_connect = weight
        best_path += path_a
    else:
        best_path = path_a + best_path

    for _ in range(2):
        for _ in range(2):
            for i in range(1, len(path_b)):
                weight = (
                    G_m[terminals_a[0], path_b[i - 1]] +
                    G_m[terminals_a[1], path_b[i]] -
                    G_m[path_b[i - 1], path_b[i]]
                )
                if weight < best_connect:
                    best_connect = weight
                    best_path = path_b.copy()
                    best_path[i:i] = path_a
            path_a.reverse()
            terminals_a.reverse()
        path_a, path_b = path_b, path_a
        terminals_a, terminals_b = terminals_b, terminals_a

    return best_path


@njit
def stitch_asymmetric(G_m, path_a, path_b):
    if len(path_a) == 1:
        return stich_singlet(G_m, path_a[0], path_b)

    if len(path_b) == 1:
        return stich_singlet(G_m, path_b[0], path_a)

    terminals_a = [path_a[0], path_a[-1]]
    terminals_b = [path_b[0], path_b[-1]]

    best_connect = G_m[terminals_a[1], terminals_b[0]]
    best_path = path_b.copy()
    weight = G_m[terminals_a[0], terminals_b[1]]
    if weight < best_connect:
        best_connect = weight
        best_path += path_a
    else:
        best_path = path_a + best_path

    for _ in range(2):
        for _ in range(2):
            path_weight = path_length(path_a, G_m) + path_length(path_a, G_m)
            best_weight = best_connect + path_weight
            for i in range(1, len(path_b)):
                weight = (
                    G_m[terminals_a[0], path_b[i - 1]] +
                    G_m[terminals_a[1], path_b[i]] -
                    G_m[path_b[i - 1], path_b[i]]
                )
                if (weight + path_weight) < best_weight:
                    best_weight = weight + path_weight
                    best_path = path_b.copy()
                    best_path[i:i] = path_a
            path_a.reverse()
            terminals_a.reverse()
        path_a, path_b = path_b, path_a
        terminals_a, terminals_b = terminals_b, terminals_a

    return best_path


@njit
def restitch(G_m, path, is_sym):
    l = len(path)
    mid = ((l + 1) if (l & 1) and (np.random.random() < 0.5) else l) >> 1

    if mid < 4:
        return path

    path_a = restitch(G_m, path[:mid], is_sym)
    path_b = restitch(G_m, path[mid:], is_sym)

    if is_sym:
        return stitch_symmetric(G_m, path_a, path_b)

    return stitch_asymmetric(G_m, path_a, path_b)


@njit
def monte_carlo_driver(n_nodes):
    a, b = [], []
    for i in range(n_nodes):
        if np.random.random() < 0.5:
            a.append(i)
        else:
            b.append(i)

    return a, b


@njit
def monte_carlo_loop(n_nodes):
    a, b = monte_carlo_driver(n_nodes)
    while (len(a) == 0) or (len(b) == 0):
        a, b = monte_carlo_driver(n_nodes)

    return a, b


# Elara suggested replacing base-case handling with her brute-force solver
@njit
def tsp_bruteforce_cyclic(G_m, perms):
    """
    Brute-force TSP solver for small n.
    G_m : numpy.ndarray (2D adjacency/weight matrix)
    is_cyclic : bool (default=True) – whether to close the tour

    Returns:
        (best_path, best_weight)
    """
    n = len(G_m)
    best_weight = float('inf')
    best_path = None

    # Must fix node 0 at start to remove rotational symmetry in cyclic case!

    max_i = len(perms[0]) - 1

    for perm in perms:
        path = [0] + list(perm)
        weight = 0.0
        for i in range(max_i):
            weight += G_m[path[i], path[i+1]]
        weight += G_m[path[-1], path[0]]

        if weight < best_weight:
            best_weight = weight
            best_path = path

    best_path = best_path + [best_path[0]]

    return best_path, best_weight

@njit
def tsp_bruteforce_acyclic(G_m, perms):
    """
    Brute-force TSP solver for small n.
    G_m : numpy.ndarray (2D adjacency/weight matrix)
    is_cyclic : bool (default=True) – whether to close the tour

    Returns:
        (best_path, best_weight)
    """
    n = len(G_m)
    best_weight = float('inf')
    best_path = None

    # Must fix node 0 at start to remove rotational symmetry in cyclic case!

    max_i = len(perms[0]) - 1

    for path in perms:
        weight = 0.0
        for i in range(max_i):
            weight += G_m[path[i], path[i+1]]

        if weight < best_weight:
            best_weight = weight
            best_path = path

    best_path = list(best_path)

    return best_path, best_weight


@njit
def tsp_symmetric_brute_force_driver(G_m, n_nodes, nodes, is_cyclic):
    if n_nodes == 3:
        if is_cyclic:
            weight_0 = G_m[0, 1] + G_m[1, 2] + G_m[2, 0]
            weight_1 = G_m[0, 2] + G_m[2, 1] + G_m[1, 0]

            if weight_0 >= weight_1:
                return (nodes + [nodes[0]], weight_0)

            nodes.reverse()

            return ([nodes[2]] + nodes, weight_1)

        w_012 = G_m[0, 1] + G_m[1, 2]
        w_021 = G_m[0, 1] + G_m[0, 2]
        w_120 = G_m[0, 2] + G_m[1, 2]

        if w_012 >= w_021 and w_012 >= w_120:
            return (nodes, w_012)

        if w_021 >= w_012 and w_021 >= w_120:
            return ([nodes[1], nodes[0], nodes[2]], w_021)

        return ([nodes[0], nodes[2], nodes[1]], w_120)

    if n_nodes == 2:
        if is_cyclic:
            return (nodes + [nodes[0]], 2 * G_m[0, 1])

        return (nodes, G_m[0, 1])

    return (nodes, 0)


@njit
def tsp_symmetric_driver(G_m, is_cyclic, is_top_level, start_node, end_node, k_neighbors, nodes, sol_a, sol_b, a, b, c, neighbor_lists, is_parallel):
    path_a = [a[x] for x in sol_a[0]]
    path_b = [b[x] for x in sol_b[0]]

    if start_node is None:
        best_path = stitch_symmetric(G_m, path_a, path_b)
    else:
        best_path = path_a + path_b
        best_weight = G_m[path_a[0], path_b[0]]
        weight = G_m[path_a[0], path_b[-1]]
        if weight < best_weight:
            path_b.reverse()
            best_path = path_a + path_b
            best_weight = weight

    if not c is None:
        best_path.append(c)

    best_path = restitch(G_m, best_path, True)

    if is_top_level:
        if is_cyclic:
            best_path += [best_path[0]]
            if is_parallel:
                best_path, _ = two_opt_parallel(best_path, G_m)
            else:
                best_path, _ = two_opt(best_path, G_m)
        elif not end_node is None:
            if is_parallel:
                best_path, _ = two_opt_parallel(best_path, G_m)
            else:
                best_path, _ = two_opt(best_path, G_m)
        elif not start_node is None:
            if is_parallel:
                best_path, _ = anchored_two_opt_parallel(best_path, G_m)
            else:
                best_path, _ = anchored_two_opt(best_path, G_m)
        else:
            if is_parallel:
                best_path, _ = one_way_two_opt_parallel(best_path, G_m)
            else:
                best_path, _ = one_way_two_opt(best_path, G_m)

        if k_neighbors > 0:
            if is_parallel:
                best_path, _ = targeted_three_opt_parallel(best_path, G_m, neighbor_lists, k_neighbors)
            else:
                best_path, _ = targeted_three_opt(best_path, G_m, neighbor_lists, k_neighbors)

    best_weight = path_length(best_path, G_m)

    return [nodes[x] for x in best_path], best_weight


def tsp_symmetric_header(G_m, nodes, quality, shots, anneal_t, anneal_h, repulsion_base, start_node, end_node, monte_carlo, is_cyclic, n_nodes, multi_start):
    if is_cyclic:
        start_node = None
        end_node = None

    if (start_node is None) and not (end_node is None):
        start_node = end_node
        end_node = None

    a, b, c = [], [], []
    if (start_node is None) and (end_node is None):
        if monte_carlo:
            a, b = monte_carlo_loop(n_nodes)
        else:
            best_cut = -float("inf")
            for _ in range(multi_start):
                bits = ([], [])
                while (len(bits[0]) == 0) or (len(bits[1]) == 0):
                    _, cut_value, bits = maxcut_tfim(G_m, quality=quality, shots=shots, anneal_t=anneal_t, anneal_h=anneal_h, repulsion_base=repulsion_base)
                if cut_value > best_cut:
                    best_cut = cut_value
                    a, b = bits
    else:
        is_cyclic = False
        a.append(nodes.index(start_node))
        b = list(range(n_nodes))
        b.remove(a[0])
        if end_node is not None:
            c.append(nodes.index(end_node))
            b.remove(c[0])

    G_a, G_b = init_G_a_b(G_m, a, b)

    return a, b, c, G_a, G_b, is_cyclic

def tsp_symmetric(
    G,
    start_node=None,
    end_node=None,
    quality=None,
    shots=None,
    anneal_t=None,
    anneal_h=None,
    repulsion_base=None,
    monte_carlo=True,
    k_neighbors=20,
    is_cyclic=True,
    multi_start=1,
    is_top_level=True,
    is_parallel=False,
    parallel_level=0
):
    dtype = opencl_context.dtype
    nodes = None
    n_nodes = 0
    G_m = None
    if isinstance(G, nx.Graph):
        nodes = list(G.nodes())
        n_nodes = len(nodes)
        G_m = nx.to_numpy_array(G, weight='weight', nonedge=0.0, dtype=dtype)
    else:
        n_nodes = len(G)
        nodes = list(range(n_nodes))
        G_m = G

    if n_nodes < 7:
        if n_nodes > 3:
            if is_cyclic:
                best_path, best_weight = tsp_bruteforce_cyclic(G_m, list(itertools.permutations(list(range(1, n_nodes)))))
            else:
                best_path, best_weight = tsp_bruteforce_acyclic(G_m, list(itertools.permutations(list(range(n_nodes)))))

            return [nodes[x] for x in best_path], best_weight

        return tsp_symmetric_brute_force_driver(G_m, n_nodes, nodes, is_cyclic)

    a, b, c, G_a, G_b, is_cylic = tsp_symmetric_header(G_m, nodes, quality, shots, anneal_t, anneal_h, repulsion_base, start_node, end_node, monte_carlo, is_cyclic, n_nodes, multi_start)

    if monte_carlo and is_parallel and (parallel_level < max_parallel_level) and (len(a) > 3) and (len(b) > 3):
        with ProcessPoolExecutor(max_workers=1) as executor:
            f = executor.submit(tsp_symmetric, G_a, None, None, quality, shots, anneal_t, anneal_h, repulsion_base, True, 0, False, multi_start, False, True, parallel_level + 1)
            sol_b = tsp_symmetric(G_b, None, None, quality, shots, anneal_t, anneal_h, repulsion_base, True, 0, False, multi_start, False, True, parallel_level + 1)
            sol_a = f.result()
    else:
        sol_a = tsp_symmetric(
            G_a,
            quality=quality,
            shots=shots,
            anneal_t=anneal_t,
            anneal_h=anneal_h,
            repulsion_base=repulsion_base,
            monte_carlo=monte_carlo,
            is_cyclic=False,
            is_top_level=False,
            k_neighbors=0,
            multi_start=multi_start,
            is_parallel=False
        )
        sol_b = tsp_symmetric(
            G_b,
            quality=quality,
            shots=shots,
            anneal_t=anneal_t,
            anneal_h=anneal_h,
            repulsion_base=repulsion_base,
            monte_carlo=monte_carlo,
            is_cyclic=False,
            is_top_level=False,
            k_neighbors=0,
            multi_start=multi_start,
            is_parallel=False
        )

    neighbor_lists = [[0]]
    if k_neighbors > 0:
        # Precompute nearest neighbors for each node
        neighbor_lists = [
            ([(float("inf"), 0)] * k_neighbors) for i in range(n_nodes)
        ]

        for i in range(n_nodes):
            for j in range(n_nodes):
                val = G_m[i, j]
                if val < neighbor_lists[i][-1][0]:
                    neighbor_lists[i][-1] = (val, j)
                    neighbor_lists[i].sort(key=lambda x: x[0])

        for i in range(len(neighbor_lists)):
            neighbor_lists[i] = [x[1] for x in neighbor_lists[i]]

    return tsp_symmetric_driver(G_m, is_cyclic, is_top_level, start_node, end_node, k_neighbors, nodes, sol_a, sol_b, a, b, c[0] if len(c) else None, np.array(neighbor_lists), is_parallel)


@njit
def tsp_asymmetric_brute_force_driver(G_m, n_nodes, nodes, is_cyclic):
    if n_nodes == 2:
        weight = G_m[0, 1]
        if G_m[1, 0] < weight:
            weight = G_m[1, 0]
            nodes.reverse()

        if is_cyclic:
            return (nodes + [nodes[0]], 2 * weight)

        return (nodes, weight)

    return (nodes, 0)


@njit
def tsp_asymmetric_driver(G_m, is_reversed, is_cyclic, is_top_level, start_node, end_node, k_neighbors, nodes, sol_a, sol_b, a, b, c, neighbor_lists, is_parallel):
    path_a = [a[x] for x in sol_a[0]]
    path_b = [b[x] for x in sol_b[0]]

    if start_node is None:
        best_path = stitch_asymmetric(G_m, path_a, path_b)
    else:
        best_path = path_a + path_b

    if not c is None:
        best_path.append(c)

    best_path = restitch(G_m, best_path, False)

    if is_top_level:
        if is_cyclic:
            best_path += [best_path[0]]
            final_path, best_weight = two_opt_parallel(best_path.copy(), G_m) if is_parallel else two_opt(best_path.copy(), G_m)
            best_path.reverse()
            path, weight = two_opt_parallel(best_path, G_m) if is_parallel else two_opt(best_path, G_m)
            if weight < best_weight:
                final_path, best_weight = path, weight
        elif not end_node is None:
            final_path, best_weight = two_opt_parallel(best_path.copy(), G_m) if is_parallel else two_opt(best_path.copy(), G_m)
            best_path.reverse()
            path, weight = two_opt_parallel(best_path, G_m) if is_parallel else two_opt(best_path, G_m)
            if weight < best_weight:
                final_path, best_weight = path, weight
        elif not start_node is None:
            final_path, best_weight = anchored_two_opt_parallel(best_path.copy(), G_m) if is_parallel else anchored_two_opt(best_path.copy(), G_m)
            best_path.reverse()
            path, weight = two_opt_parallel(best_path, G_m) if is_parallel else two_opt(best_path, G_m)
            if weight < best_weight:
                final_path, best_weight = path, weight
        else:
            final_path, best_weight = one_way_two_opt_parallel(best_path.copy(), G_m) if is_parallel else one_way_two_opt(best_path.copy(), G_m)
            best_path.reverse()
            path, weight = one_way_two_opt_parallel(best_path, G_m) if is_parallel else one_way_two_opt(best_path, G_m)
            if weight < best_weight:
                final_path, best_weight = path, weight

        if k_neighbors > 0:
            best_path = final_path
            final_path, best_weight = targeted_three_opt_parallel(best_path.copy(), G_m, neighbor_lists, k_neighbors) if is_parallel else targeted_three_opt(best_path.copy(), G_m, neighbor_lists, k_neighbors)
            best_path.reverse()
            path, weight = targeted_three_opt_parallel(best_path, G_m, neighbor_lists, k_neighbors) if is_parallel else targeted_three_opt(best_path, G_m, neighbor_lists, k_neighbors)
            if weight < best_weight:
                final_path, best_weight = path, weight
    else:
        final_path = best_path

    best_weight = path_length(final_path, G_m)

    if is_reversed:
        final_path.reverse()

    return [nodes[x] for x in final_path], best_weight


def tsp_asymmetric_header(G_m, nodes, quality, shots, anneal_t, anneal_h, repulsion_base, start_node, end_node, monte_carlo, is_cyclic, n_nodes, multi_start):
    if is_cyclic:
        start_node = None
        end_node = None

    is_reversed = False
    if (start_node is None) and not (end_node is None):
        is_reversed = True
        start_node = end_node
        end_node = None
        G_m = G_m.T

    a, b, c = [], [], []
    if (start_node is None) and (end_node is None):
        if monte_carlo:
            a, b = monte_carlo_loop(n_nodes)
        else:
            best_cut = -float("inf")
            for _ in range(multi_start):
                bits = ([], [])
                while (len(bits[0]) == 0) or (len(bits[1]) == 0):
                    _, cut_value, bits = maxcut_tfim((G_m + G_m.T) / 2, quality=quality, shots=shots, anneal_t=anneal_t, anneal_h=anneal_h, repulsion_base=repulsion_base)
                if cut_value > best_cut:
                    best_cut = cut_value
                    a, b = bits
    else:
        is_cyclic = False
        a.append(nodes.index(start_node))
        b = list(range(n_nodes))
        b.remove(a[0])
        if end_node is not None:
            c.append(nodes.index(end_node))
            b.remove(c[0])

    G_a, G_b = init_G_a_b(G_m, a, b)

    return a, b, c, G_a, G_b, is_cyclic, is_reversed


def tsp_asymmetric(
    G,
    start_node=None,
    end_node=None,
    quality=None,
    shots=None,
    anneal_t=None,
    anneal_h=None,
    repulsion_base=None,
    monte_carlo=True,
    k_neighbors=20,
    is_cyclic=True,
    multi_start=1,
    is_top_level=True,
    is_parallel=False,
    parallel_level=0
):
    dtype = opencl_context.dtype
    nodes = None
    n_nodes = 0
    G_m = None
    if isinstance(G, nx.Graph):
        nodes = list(G.nodes())
        n_nodes = len(nodes)
        G_m = nx.to_numpy_array(G, weight='weight', nonedge=0.0, dtype=dtype)
    else:
        n_nodes = len(G)
        nodes = list(range(n_nodes))
        G_m = G

    if n_nodes < 7:
        if n_nodes > 2:
            if is_cyclic:
                best_path, best_weight = tsp_bruteforce_cyclic(G_m, list(itertools.permutations(list(range(1, n_nodes)))))
            else:
                best_path, best_weight = tsp_bruteforce_acyclic(G_m, list(itertools.permutations(list(range(n_nodes)))))

            return [nodes[x] for x in best_path], best_weight

        return tsp_asymmetric_brute_force_driver(G_m, n_nodes, nodes, is_cyclic)

    a, b, c, G_a, G_b, is_cyclic, is_reversed = tsp_asymmetric_header(G_m, nodes, quality, shots, anneal_t, anneal_h, repulsion_base, start_node, end_node, monte_carlo, is_cyclic, n_nodes, multi_start)

    if monte_carlo and is_parallel and (parallel_level < max_parallel_level) and (len(a) > 2) and (len(b) > 2):
        with ProcessPoolExecutor(max_workers=1) as executor:
            f = executor.submit(tsp_asymmetric, G_a, None, None, quality, shots, anneal_t, anneal_h, repulsion_base, True, 0, False, multi_start, False, True, parallel_level + 1)
            sol_b = tsp_asymmetric(G_b, None, None, quality, shots, anneal_t, anneal_h, repulsion_base, True, 0, False, multi_start, False, True, parallel_level + 1)
            sol_a = f.result()
    else:
        sol_a = tsp_asymmetric(
            G_a,
            quality=quality,
            shots=shots,
            anneal_t=anneal_t,
            anneal_h=anneal_h,
            repulsion_base=repulsion_base,
            monte_carlo=monte_carlo,
            is_cyclic=False,
            is_top_level=False,
            k_neighbors=0,
            multi_start=multi_start,
            is_parallel=False
        )
        sol_b = tsp_asymmetric(
            G_b,
            quality=quality,
            shots=shots,
            anneal_t=anneal_t,
            anneal_h=anneal_h,
            repulsion_base=repulsion_base,
            monte_carlo=monte_carlo,
            is_cyclic=False,
            is_top_level=False,
            k_neighbors=0,
            multi_start=multi_start,
            is_parallel=False
        )

    neighbor_lists = [[0]]
    if k_neighbors > 0:
        # Precompute nearest neighbors for each node
        neighbor_lists = [
            ([(float("inf"), 0)] * k_neighbors) for i in range(n_nodes)
        ]

        for i in range(n_nodes):
            for j in range(n_nodes):
                val = G_m[i, j]
                if val < neighbor_lists[i][-1][0]:
                    neighbor_lists[i][-1] = (val, j)
                    neighbor_lists[i].sort(key=lambda x: x[0])

        for i in range(len(neighbor_lists)):
            neighbor_lists[i] = [x[1] for x in neighbor_lists[i]]

    return tsp_asymmetric_driver(G_m, is_reversed, is_cyclic, is_top_level, start_node, end_node, k_neighbors, nodes, sol_a, sol_b, a, b, c[0] if len(c) else None, np.array(neighbor_lists), is_parallel)


@njit
def get_G_m(G_data, G_rows, G_cols, low, high):
    if high < low:
        low, high = high, low

    start = G_rows[low]
    end = G_rows[low + 1]

    i = binary_search(G_cols[start:end], high) + start
    if i < end:
        return G_data[i]

    return 0.0


@njit
def path_length_sparse(path, G_data, G_rows, G_cols):
    tot_len = 0.0
    for i in range(len(path) - 1):
        low, high = path[i], path[i + 1]
        if high < low:
            low, high = high, low

        start = G_rows[low]
        end = G_rows[low + 1]

        i = binary_search(G_cols[start:end], high) + start
        if i < end:
            tot_len += G_data[i]

    return tot_len


@njit
def one_way_two_opt_sparse(best_path, G_data, G_rows, G_cols):
    improved = True
    best_dist = path_length_sparse(best_path, G_data, G_rows, G_cols)
    path_len = len(best_path)

    while improved:
        improved = False
        for i in range(1, path_len - 1):
            for j in range(i + 2, path_len):
                new_path = best_path[:]
                new_path[i:j] = best_path[j-1:i-1:-1]
                new_dist = path_length_sparse(new_path, G_data, G_rows, G_cols)
                if new_dist < best_dist:
                    best_path, best_dist = new_path, new_dist
                    improved = True

    return best_path, best_dist


@njit(parallel=True)
def one_way_two_opt_sparse_parallel(best_path, G_data, G_rows, G_cols):
    improved = True
    best_dist = path_length_sparse(best_path, G_data, G_rows, G_cols)
    paths = [best_path] * n_threads
    dists = [best_dist] * n_threads
    path_len = len(best_path)

    while improved:
        improved = False
        for i in range(1, path_len - 1):
            for j in range(i + 2, path_len, n_threads):
                max_k = min(n_threads, path_len - j)
                for k in prange(max_k):
                    l = j + k
                    new_path = best_path[:]
                    new_path[i:l] = best_path[l-1:i-1:-1]
                    dists[k] = path_length_sparse(new_path, G_data, G_rows, G_cols)
                    paths[k] = new_path
                for k in range(max_k):
                    if dists[k] < best_dist:
                        best_path, best_dist, improved = paths[k], dists[k], True

    return best_path, best_dist


@njit
def targeted_three_opt_sparse(path, G_data, G_rows, G_cols, neighbor_lists, k_neighbors=20):
    n = len(path)
    best_path = path[:]
    best_dist = path_length_sparse(best_path, G_data, G_rows, G_cols)
    improved = True

    while improved:
        improved = False
        for i in range(n - 5):
            for _j in range(k_neighbors):
                j = neighbor_lists[path[i], _j]
                if j <= i or j >= n-3:
                    continue
                for _k in range(k_neighbors):
                    k = neighbor_lists[path[j], _k]
                    if k <= j or k >= n-1:
                        continue

                    # 7 unique cases (same as brute force, but restricted)
                    new_path = best_path[:i+1] + best_path[i+1:j+1][::-1] + best_path[j+1:]
                    dist = path_length_sparse(new_path, G_data, G_rows, G_cols)
                    if dist < best_dist:
                        best_path, best_dist, improved = new_path, dist, True
                        break

                    new_path = best_path[:j+1] + best_path[j+1:k+1][::-1] + best_path[k+1:]
                    dist = path_length_sparse(new_path, G_data, G_rows, G_cols)
                    if dist < best_dist:
                        best_path, best_dist, improved = new_path, dist, True
                        break

                    new_path = best_path[:i+1] + best_path[i+1:j+1][::-1] + best_path[j+1:k+1][::-1] + best_path[k+1:]
                    dist = path_length_sparse(new_path, G_data, G_rows, G_cols)
                    if dist < best_dist:
                        best_path, best_dist, improved = new_path, dist, True
                        break

                    new_path = best_path[:i+1] + best_path[j+1:k+1] + best_path[i+1:j+1] + best_path[k+1:]
                    dist = path_length_sparse(new_path, G_data, G_rows, G_cols)
                    if dist < best_dist:
                        best_path, best_dist, improved = new_path, dist, True
                        break

                    new_path = best_path[:i+1] + best_path[j+1:k+1][::-1] + best_path[i+1:j+1][::-1] + best_path[k+1:]
                    dist = path_length_sparse(new_path, G_data, G_rows, G_cols)
                    if dist < best_dist:
                        best_path, best_dist, improved = new_path, dist, True
                        break

                    new_path = best_path[:i+1] + best_path[j+1:k+1] + best_path[i+1:j+1][::-1] + best_path[k+1:]
                    dist = path_length_sparse(new_path, G_data, G_rows, G_cols)
                    if dist < best_dist:
                        best_path, best_dist, improved = new_path, dist, True
                        break

                    new_path = best_path[:i+1] + best_path[j+1:k+1][::-1] + best_path[i+1:j+1] + best_path[k+1:]
                    dist = path_length_sparse(new_path, G_data, G_rows, G_cols)
                    if dist < best_dist:
                        best_path, best_dist, improved = new_path, dist, True
                        break

                if improved:
                    break

            if improved:
                break

        path = best_path[:]

    return best_path, best_dist


@njit(parallel=True)
def targeted_three_opt_sparse_parallel(path, G_data, G_rows, G_cols, neighbor_lists, k_neighbors=20):
    n = len(path)
    best_path = path[:]
    best_dist = path_length_sparse(best_path, G_data, G_rows, G_cols)
    paths = [best_path] * 7
    dists = [best_dist] * 7
    improved = True

    while improved:
        improved = False
        for i in range(n - 5):
            for _j in range(k_neighbors):
                j = neighbor_lists[path[i], _j]
                if j <= i or j >= n-3:
                    continue
                for _k in range(k_neighbors):
                    k = neighbor_lists[path[j], _k]
                    if k <= j or k >= n-1:
                        continue

                    # 7 unique cases (same as brute force, but restricted)
                    for l in prange(7):
                        match l:
                            case 0:
                                new_path = best_path[:i+1] + best_path[i+1:j+1][::-1] + best_path[j+1:]
                            case 1:
                                new_path = best_path[:j+1] + best_path[j+1:k+1][::-1] + best_path[k+1:]
                            case 2:
                                new_path = best_path[:i+1] + best_path[i+1:j+1][::-1] + best_path[j+1:k+1][::-1] + best_path[k+1:]
                            case 3:
                                new_path = best_path[:i+1] + best_path[j+1:k+1] + best_path[i+1:j+1] + best_path[k+1:]
                            case 4:
                                new_path = best_path[:i+1] + best_path[j+1:k+1][::-1] + best_path[i+1:j+1][::-1] + best_path[k+1:]
                            case 5:
                                new_path = best_path[:i+1] + best_path[j+1:k+1] + best_path[i+1:j+1][::-1] + best_path[k+1:]
                            case _:
                                new_path = best_path[:i+1] + best_path[j+1:k+1][::-1] + best_path[i+1:j+1] + best_path[k+1:]

                        dists[l] = path_length_sparse(new_path, G_data, G_rows, G_cols)
                        paths[l] = new_path

                    for l in range(7):
                        if dists[l] < best_dist:
                            best_path, best_dist, improved = paths[l], dists[l], True

                    if improved:
                        break

                if improved:
                    break

            if improved:
                break

        path = best_path[:]

    return best_path, best_dist


@njit
def stich_singlet_sparse(G_data, G_rows, G_cols, singlet, bulk):
    best_path = bulk.copy()
    best_weight = get_G_m(G_data, G_rows, G_cols, singlet, bulk[0])
    weight = get_G_m(G_data, G_rows, G_cols, singlet, bulk[-1])
    if weight < best_weight:
        best_weight = weight
        best_path += [singlet]
    else:
        best_path = [singlet] + best_path

    for i in range(1, len(bulk)):
        weight = (
            get_G_m(G_data, G_rows, G_cols, singlet, bulk[i - 1]) +
            get_G_m(G_data, G_rows, G_cols, singlet, bulk[i]) -
            get_G_m(G_data, G_rows, G_cols, bulk[i - 1], bulk[i])
        )
        if weight < best_weight:
            best_weight = weight
            best_path = bulk.copy()
            best_path.insert(i, singlet)

    return best_path


@njit
def stitch_sparse_symmetric(G_data, G_rows, G_cols, path_a, path_b):
    if len(path_a) == 1:
        return stich_singlet_sparse(G_data, G_rows, G_cols, path_a[0], path_b)

    if len(path_b) == 1:
        return stich_singlet_sparse(G_data, G_rows, G_cols, path_b[0], path_a)

    terminals_a = [path_a[0], path_a[-1]]
    terminals_b = [path_b[0], path_b[-1]]

    best_connect = get_G_m(G_data, G_rows, G_cols, terminals_a[1], terminals_b[0])
    best_path = path_b.copy()
    weight = get_G_m(G_data, G_rows, G_cols, terminals_a[0], terminals_b[1])
    if weight < best_connect:
        best_connect = weight
        best_path += path_a
    else:
        best_path = path_a + best_path

    for _ in range(2):
        for _ in range(2):
            for i in range(1, len(path_b)):
                weight = (
                    get_G_m(G_data, G_rows, G_cols, terminals_a[0], path_b[i - 1]) +
                    get_G_m(G_data, G_rows, G_cols, terminals_a[1], path_b[i]) -
                    get_G_m(G_data, G_rows, G_cols, path_b[i - 1], path_b[i])
                )
                if weight < best_connect:
                    best_connect = weight
                    best_path = path_b.copy()
                    best_path[i:i] = path_a
            path_a.reverse()
            terminals_a.reverse()
        path_a, path_b = path_b, path_a
        terminals_a, terminals_b = terminals_b, terminals_a

    return best_path


@njit
def restitch_sparse(G_data, G_rows, G_cols, path):
    l = len(path)
    mid = ((l + 1) if (l & 1) and (np.random.random() < 0.5) else l) >> 1

    if mid < 4:
        return path

    path_a = restitch_sparse(G_data, G_rows, G_cols, path[:mid])
    path_b = restitch_sparse(G_data, G_rows, G_cols, path[mid:])

    return stitch_sparse_symmetric(G_data, G_rows, G_cols, path_a, path_b)


@njit
def tsp_sparse_brute_force_driver(G_data, G_rows, G_cols, n_nodes, nodes):
    if n_nodes == 3:
        w_012 = get_G_m(G_data, G_rows, G_cols, 0, 1) + get_G_m(G_data, G_rows, G_cols, 1, 2)
        w_021 = get_G_m(G_data, G_rows, G_cols, 0, 1) + get_G_m(G_data, G_rows, G_cols, 0, 2)
        w_120 = get_G_m(G_data, G_rows, G_cols, 0, 2) + get_G_m(G_data, G_rows, G_cols, 1, 2)

        if w_012 >= w_021 and w_012 >= w_120:
            return (nodes, w_012)

        if w_021 >= w_012 and w_021 >= w_120:
            return ([nodes[1], nodes[0], nodes[2]], w_021)

        return ([nodes[0], nodes[2], nodes[1]], w_120)

    if n_nodes == 2:
        return (nodes, get_G_m(G_data, G_rows, G_cols, 0, 1))

    return (nodes, 0)


@njit
def tsp_sparse_bruteforce(G_data, G_rows, G_cols, perms):
    n = len(G_rows) - 1
    best_weight = float('inf')
    best_path = None

    # Must fix node 0 at start to remove rotational symmetry in cyclic case!

    max_i = len(perms[0]) - 1

    for path in perms:
        weight = 0.0
        for i in range(max_i):
            low, high = path[i], path[i + 1]
            if high < low:
                low, high = high, low

            start = G_rows[low]
            end = G_rows[low + 1]

            i = binary_search(G_cols[start:end], high) + start
            if i < end:
                weight += G_data[i]

        if weight < best_weight:
            best_weight = weight
            best_path = path

    best_path = list(best_path)

    return best_path, best_weight


def init_G_a_b_sparse(G_m, a, b, dtype):
    n_a_nodes = len(a)
    n_b_nodes = len(b)
    G_a = lil_matrix((n_a_nodes, n_a_nodes), dtype=opencl_context.dtype)
    G_b = lil_matrix((n_b_nodes, n_b_nodes), dtype=opencl_context.dtype)
    for i in range(n_a_nodes):
        for j in range(n_a_nodes):
            if i == j:
                continue
            weight = G_m[a[i], a[j]]
            if weight != 0.0:
                G_a[i, j] = weight
    for i in range(n_b_nodes):
        for j in range(n_b_nodes):
            if i == j:
                continue
            weight = G_m[b[i], b[j]]
            if weight != 0.0:
                G_b[i, j] = weight

    return G_a.tocsr(), G_b.tocsr()


@njit
def tsp_symmetric_sparse_driver(G_data, G_rows, G_cols, is_top_level, k_neighbors, nodes, sol_a, sol_b, a, b, neighbor_lists, is_parallel):
    path_a = [a[x] for x in sol_a[0]]
    path_b = [b[x] for x in sol_b[0]]

    best_path = stitch_sparse_symmetric(G_data, G_rows, G_cols, path_a, path_b)

    if is_top_level:
        if is_parallel:
            best_path, _ = one_way_two_opt_sparse_parallel(best_path, G_data, G_rows, G_cols)
        else:
            best_path, _ = one_way_two_opt_sparse(best_path, G_data, G_rows, G_cols)

        if k_neighbors > 0:
            if is_parallel:
                best_path, _ = targeted_three_opt_sparse_parallel(best_path, G_data, G_rows, G_cols, neighbor_lists, k_neighbors)
            else:
                best_path, _ = targeted_three_opt_sparse(best_path, G_data, G_rows, G_cols, neighbor_lists, k_neighbors)

    best_path = restitch_sparse(G_data, G_rows, G_cols, best_path)

    best_weight = path_length_sparse(best_path, G_data, G_rows, G_cols)

    return [nodes[x] for x in best_path], best_weight


def tsp_symmetric_sparse(
    G,
    k_neighbors=20,
    is_top_level=True,
    is_parallel=True,
    parallel_level=0
):
    dtype = opencl_context.dtype
    nodes = None
    n_nodes = 0
    G_m = None
    if isinstance(G, nx.Graph):
        nodes = list(G.nodes())
        n_nodes = len(nodes)
        G_m = to_scipy_sparse_upper_triangular(G, nodes, n_nodes, dtype)
    else:
        n_nodes = G.shape[0]
        nodes = list(range(n_nodes))
        G_m = G

    if n_nodes < 7:
        if n_nodes > 3:
            best_path, best_weight = tsp_sparse_bruteforce(G_m.data, G_m.indptr, G_m.indices, list(itertools.permutations(list(range(n_nodes)))))

            return [nodes[x] for x in best_path], best_weight

        return tsp_sparse_brute_force_driver(G_m.data, G_m.indptr, G_m.indices, n_nodes, nodes)

    a, b = monte_carlo_loop(n_nodes)

    G_a, G_b = init_G_a_b_sparse(G_m, a, b, dtype)

    if is_parallel and (parallel_level < max_parallel_level) and (len(a) > 3) and (len(b) > 3):
        with ProcessPoolExecutor(max_workers=1) as executor:
            f = executor.submit(tsp_symmetric_sparse, G_a, 0, False, True, parallel_level + 1)
            sol_b = tsp_symmetric_sparse(G_b, 0, False, True, parallel_level + 1)
            sol_a = f.result()
    else:
        sol_a = tsp_symmetric_sparse(
            G_a,
            is_top_level=False,
            k_neighbors=0,
            is_parallel=False
        )
        sol_b = tsp_symmetric_sparse(
            G_b,
            is_top_level=False,
            k_neighbors=0,
            is_parallel=False
        )

    neighbor_lists = [[0]]
    if k_neighbors > 0:
        # Precompute nearest neighbors for each node
        neighbor_lists = [
            ([(float("inf"), 0)] * k_neighbors) for i in range(n_nodes)
        ]

        for i in range(n_nodes):
            for j in range(n_nodes):
                val = G_m[i, j]
                if val < neighbor_lists[i][-1][0]:
                    neighbor_lists[i][-1] = (val, j)
                    neighbor_lists[i].sort(key=lambda x: x[0])

        for i in range(len(neighbor_lists)):
            neighbor_lists[i] = [x[1] for x in neighbor_lists[i]]

    return tsp_symmetric_sparse_driver(G_m.data, G_m.indptr, G_m.indices, is_top_level, k_neighbors, nodes, sol_a, sol_b, a, b, np.array(neighbor_lists), is_parallel)


@njit
def path_length_streaming(path, G_func):
    tot_len = 0.0
    for i in range(len(path) - 1):
        tot_len += G_func(path[i], path[i + 1])

    return tot_len


@njit
def one_way_two_opt_streaming(best_path, G_func):
    improved = True
    best_dist = path_length_streaming(best_path, G_func)
    path_len = len(best_path)

    while improved:
        improved = False
        for i in range(1, path_len - 1):
            for j in range(i + 2, path_len):
                new_path = best_path[:]
                new_path[i:j] = best_path[j-1:i-1:-1]
                new_dist = path_length_streaming(new_path, G_func)
                if new_dist < best_dist:
                    best_path, best_dist, improved = new_path, new_dist, True

    return best_path, best_dist


@njit(parallel=True)
def one_way_two_opt_streaming_parallel(best_path, G_func):
    improved = True
    best_dist = path_length_streaming(best_path, G_func)
    paths = [best_path] * n_threads
    dists = [best_dist] * n_threads
    path_len = len(best_path)

    while improved:
        improved = False
        for i in range(1, path_len - 1):
            for j in range(i + 2, path_len, n_threads):
                max_k = min(n_threads, path_len - j)
                for k in prange(max_k):
                    l = j + k
                    new_path = best_path[:]
                    new_path[i:l] = best_path[l-1:i-1:-1]
                    dists[k] = path_length_streaming(new_path, G_func)
                    paths[k] = new_path
                for k in range(max_k):
                    if dists[k] < best_dist:
                        best_path, best_dist, improved = paths[k], dists[k], True

    return best_path, best_dist


@njit
def targeted_three_opt_streaming(path, G_func, neighbor_lists, k_neighbors=20):
    n = len(path)
    best_path = path[:]
    best_dist = path_length_streaming(best_path, G_func)
    improved = True

    while improved:
        improved = False
        for i in range(n - 5):
            for _j in range(k_neighbors):
                j = neighbor_lists[path[i], _j]
                if j <= i or j >= n-3:
                    continue
                for _k in range(k_neighbors):
                    k = neighbor_lists[path[j], _k]
                    if k <= j or k >= n-1:
                        continue

                    # 7 unique cases (same as brute force, but restricted)
                    new_path = best_path[:i+1] + best_path[i+1:j+1][::-1] + best_path[j+1:]
                    dist = path_length_streaming(new_path, G_func)
                    if dist < best_dist:
                        best_path, best_dist, improved = new_path, dist, True
                        break

                    new_path = best_path[:j+1] + best_path[j+1:k+1][::-1] + best_path[k+1:]
                    dist = path_length_streaming(new_path, G_func)
                    if dist < best_dist:
                        best_path, best_dist, improved = new_path, dist, True
                        break

                    new_path = best_path[:i+1] + best_path[i+1:j+1][::-1] + best_path[j+1:k+1][::-1] + best_path[k+1:]
                    dist = path_length_streaming(new_path, G_func)
                    if dist < best_dist:
                        best_path, best_dist, improved = new_path, dist, True
                        break

                    new_path = best_path[:i+1] + best_path[j+1:k+1] + best_path[i+1:j+1] + best_path[k+1:]
                    dist = path_length_streaming(new_path, G_func)
                    if dist < best_dist:
                        best_path, best_dist, improved = new_path, dist, True
                        break

                    new_path = best_path[:i+1] + best_path[j+1:k+1][::-1] + best_path[i+1:j+1][::-1] + best_path[k+1:]
                    dist = path_length_streaming(new_path, G_func)
                    if dist < best_dist:
                        best_path, best_dist, improved = new_path, dist, True
                        break

                    new_path = best_path[:i+1] + best_path[j+1:k+1] + best_path[i+1:j+1][::-1] + best_path[k+1:]
                    dist = path_length_streaming(new_path, G_func)
                    if dist < best_dist:
                        best_path, best_dist, improved = new_path, dist, True
                        break

                    new_path = best_path[:i+1] + best_path[j+1:k+1][::-1] + best_path[i+1:j+1] + best_path[k+1:]
                    dist = path_length_streaming(new_path, G_func)
                    if dist < best_dist:
                        best_path, best_dist, improved = new_path, dist, True
                        break

                if improved:
                    break

            if improved:
                break

        path = best_path[:]

    return best_path, best_dist


@njit(parallel=True)
def targeted_three_opt_streaming_parallel(path, G_func, neighbor_lists, k_neighbors=20):
    n = len(path)
    best_path = path[:]
    best_dist = path_length_streaming(best_path, G_func)
    paths = [best_path] * 7
    dists = [best_dist] * 7
    improved = True

    while improved:
        improved = False
        for i in range(n - 5):
            for _j in range(k_neighbors):
                j = neighbor_lists[path[i], _j]
                if j <= i or j >= n-3:
                    continue
                for _k in range(k_neighbors):
                    k = neighbor_lists[path[j], _k]
                    if k <= j or k >= n-1:
                        continue

                    # 7 unique cases (same as brute force, but restricted)
                    for l in prange(7):
                        match l:
                            case 0:
                                new_path = best_path[:i+1] + best_path[i+1:j+1][::-1] + best_path[j+1:]
                            case 1:
                                new_path = best_path[:j+1] + best_path[j+1:k+1][::-1] + best_path[k+1:]
                            case 2:
                                new_path = best_path[:i+1] + best_path[i+1:j+1][::-1] + best_path[j+1:k+1][::-1] + best_path[k+1:]
                            case 3:
                                new_path = best_path[:i+1] + best_path[j+1:k+1] + best_path[i+1:j+1] + best_path[k+1:]
                            case 4:
                                new_path = best_path[:i+1] + best_path[j+1:k+1][::-1] + best_path[i+1:j+1][::-1] + best_path[k+1:]
                            case 5:
                                new_path = best_path[:i+1] + best_path[j+1:k+1] + best_path[i+1:j+1][::-1] + best_path[k+1:]
                            case _:
                                new_path = best_path[:i+1] + best_path[j+1:k+1][::-1] + best_path[i+1:j+1] + best_path[k+1:]

                        dists[l] = path_length_streaming(new_path, G_func)
                        paths[l] = new_path

                    for l in range(7):
                        if dists[l] < best_dist:
                            best_path, best_dist, improved = paths[l], dists[l], True

                    if improved:
                        break

                if improved:
                    break

            if improved:
                break

        path = best_path[:]

    return best_path, best_dist


@njit
def stich_singlet_streaming(G_func, singlet, bulk):
    best_path = bulk.copy()
    best_weight = G_func(singlet, bulk[0])
    weight = G_func(singlet, bulk[-1])
    if weight < best_weight:
        best_weight = weight
        best_path += [singlet]
    else:
        best_path = [singlet] + best_path

    for i in range(1, len(bulk)):
        weight = (
            G_func(singlet, bulk[i - 1]) +
            G_func(singlet, bulk[i]) -
            G_func(bulk[i - 1], bulk[i])
        )
        if weight < best_weight:
            best_weight = weight
            best_path = bulk.copy()
            best_path.insert(i, singlet)

    return best_path


@njit
def stitch_streaming_symmetric(G_func, path_a, path_b):
    if len(path_a) == 1:
        return stich_singlet_streaming(G_func, path_a[0], path_b)

    if len(path_b) == 1:
        return stich_singlet_streaming(G_func, path_b[0], path_a)

    terminals_a = [path_a[0], path_a[-1]]
    terminals_b = [path_b[0], path_b[-1]]

    best_connect = G_func(terminals_a[1], terminals_b[0])
    best_path = path_b.copy()
    weight = G_func(terminals_a[0], terminals_b[1])
    if weight < best_connect:
        best_connect = weight
        best_path += path_a
    else:
        best_path = path_a + best_path

    for _ in range(2):
        for _ in range(2):
            for i in range(1, len(path_b)):
                weight = (
                    G_func(terminals_a[0], path_b[i - 1]) +
                    G_func(terminals_a[1], path_b[i]) -
                    G_func(path_b[i - 1], path_b[i])
                )
                if weight < best_connect:
                    best_connect = weight
                    best_path = path_b.copy()
                    best_path[i:i] = path_a
            path_a.reverse()
            terminals_a.reverse()
        path_a, path_b = path_b, path_a
        terminals_a, terminals_b = terminals_b, terminals_a

    return best_path


@njit
def restitch_streaming(G_func, path):
    l = len(path)
    mid = ((l + 1) if (l & 1) and (np.random.random() < 0.5) else l) >> 1

    if mid < 4:
        return path

    path_a = restitch_streaming(G_func, path[:mid])
    path_b = restitch_streaming(G_func, path[mid:])

    return stitch_streaming_symmetric(G_func, path_a, path_b)


@njit
def tsp_streaming_brute_force_driver(G_func, n_nodes, nodes):
    if n_nodes == 3:
        w_012 = G_func(0, 1) + G_func(1, 2)
        w_021 = G_func(0, 1) + G_func(0, 2)
        w_120 = G_func(0, 2) + G_func(1, 2)

        if w_012 >= w_021 and w_012 >= w_120:
            return (nodes, w_012)

        if w_021 >= w_012 and w_021 >= w_120:
            return ([nodes[1], nodes[0], nodes[2]], w_021)

        return ([nodes[0], nodes[2], nodes[1]], w_120)

    if n_nodes == 2:
        return (nodes, G_func(0, 1))

    return (nodes, 0)


@njit
def tsp_streaming_bruteforce(G_func, perms, n):
    best_weight = float('inf')
    best_path = None

    # Must fix node 0 at start to remove rotational symmetry in cyclic case!

    max_i = len(perms[0]) - 1

    for path in perms:
        weight = 0.0
        for i in range(max_i):
            weight += G_func(path[i], path[i+1])

        if weight < best_weight:
            best_weight = weight
            best_path = path

    best_path = list(best_path)

    return best_path, best_weight


@njit
def tsp_symmetric_streaming_driver(G_func, is_top_level, k_neighbors, nodes, path_a, path_b, neighbor_lists, is_parallel):
    best_path = stitch_streaming_symmetric(G_func, path_a, path_b)

    if is_top_level:
        if is_parallel:
            best_path, _ = one_way_two_opt_streaming_parallel(best_path, G_func)
        else:
            best_path, _ = one_way_two_opt_streaming(best_path, G_func)

        if k_neighbors > 0:
            if is_parallel:
                best_path, _ = targeted_three_opt_streaming_parallel(best_path, G_func, neighbor_lists, k_neighbors)
            else:
                best_path, _ = targeted_three_opt_streaming(best_path, G_func, neighbor_lists, k_neighbors)

    best_path = restitch_streaming(G_func, best_path)

    best_weight = path_length_streaming(best_path, G_func)

    return [nodes[x] for x in best_path], best_weight


def tsp_symmetric_streaming(
    G_func,
    nodes,
    k_neighbors=20,
    is_top_level=True,
    is_parallel=True,
    parallel_level=0
):
    dtype = opencl_context.dtype
    n_nodes = len(nodes)

    if n_nodes < 7:
        if n_nodes > 3:
            best_path, best_weight = tsp_streaming_bruteforce(G_func, list(itertools.permutations(nodes)), n_nodes)

            return best_path, best_weight

        return tsp_streaming_brute_force_driver(G_func, n_nodes, nodes)

    a, b = monte_carlo_loop(n_nodes)

    if is_parallel and (parallel_level < max_parallel_level) and (len(a) > 3) and (len(b) > 3):
        with ProcessPoolExecutor(max_workers=1) as executor:
            f = executor.submit(tsp_symmetric_streaming, G_func, a, 0, False, True, parallel_level + 1)
            sol_b = tsp_symmetric_streaming(G_func, b, 0, False, True, parallel_level + 1)
            sol_a = f.result()
    else:
        sol_a = tsp_symmetric_streaming(
            G_func,
            a,
            is_top_level=False,
            k_neighbors=0,
            is_parallel=False
        )
        sol_b = tsp_symmetric_streaming(
            G_func,
            b,
            is_top_level=False,
            k_neighbors=0,
            is_parallel=False
        )

    neighbor_lists = [[0]]
    if k_neighbors > 0:
        # Precompute nearest neighbors for each node
        neighbor_lists = [
            ([(float("inf"), 0)] * k_neighbors) for i in range(n_nodes)
        ]

        for i in nodes:
            for j in nodes:
                val = G_func(i, j)
                if val < neighbor_lists[i][-1][0]:
                    neighbor_lists[i][-1] = (val, j)
                    neighbor_lists[i].sort(key=lambda x: x[0])

        for i in range(len(neighbor_lists)):
            neighbor_lists[i] = [x[1] for x in neighbor_lists[i]]

    return tsp_symmetric_streaming_driver(G_func, is_top_level, k_neighbors, nodes, sol_a[0], sol_b[0], np.array(neighbor_lists), is_parallel)
