from .spin_glass_solver import spin_glass_solver
import itertools
import networkx as nx
from numba import njit
import numpy as np


# two_opt() and targeted_three_opt() written by Elara (OpenAI ChatGPT instance)
@njit
def path_length(path, G_m):
    tot_len = 0.0
    for i in range(len(path) - 1):
        tot_len += G_m[path[i], path[i + 1]]

    return tot_len


@njit
def one_way_two_opt(path, G):
    improved = True
    best_path = path
    best_dist = path_length(best_path, G)
    path_len = len(path)

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
        path = best_path
    return best_path, best_dist


@njit
def anchored_two_opt(path, G):
    improved = True
    best_path = path
    best_dist = path_length(best_path, G)
    path_len = len(path)

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
        path = best_path
    return best_path, best_dist


@njit
def reversed_anchored_two_opt(path, G):
    improved = True
    best_path = path
    best_dist = path_length(best_path, G)
    path_len = len(path)

    while improved:
        improved = False
        for i in range(1, path_len - 2):
            for j in range(i + 2, path_len - 1):
                new_path = best_path[:]
                new_path[i:j] = best_path[j-1:i-1:-1]
                new_dist = path_length(new_path, G)
                if new_dist < best_dist:
                    best_path, best_dist = new_path, new_dist
                    improved = True
        path = best_path
    return best_path, best_dist


@njit
def two_opt(path, G):
    improved = True
    best_path = path
    best_dist = path_length(best_path, G)
    path_len = len(path)

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
        path = best_path
    return best_path, best_dist


@njit
def targeted_three_opt(path, W, k_neighbors=20):
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

    # Precompute nearest neighbors for each node
    neighbor_lists = [
        np.argsort(W[i])[:k_neighbors] for i in range(n)
    ]

    while improved:
        improved = False
        for i in range(n - 5):
            for j in neighbor_lists[path[i]]:
                if j <= i or j >= n-3:
                    continue
                for k in neighbor_lists[path[j]]:
                    if k <= j or k >= n-1:
                        continue

                    # Extract indices
                    A, B, C, D, E, F = path[i], path[i+1], path[j], path[j+1], path[k], path[k+1]

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
def tsp_symmetric_driver(G_m, is_cyclic, is_top_level, start_node, end_node, k_neighbors, nodes, sol_a, sol_b, a, b, c):
    path_a = [a[x] for x in sol_a[0]]
    path_b = [b[x] for x in sol_b[0]]

    restitch(G_m, path_a, True)
    restitch(G_m, path_b, True)

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

    if is_top_level:
        if is_cyclic:
            best_path += [best_path[0]]
            best_path, _ = two_opt(best_path, G_m)
        elif not end_node is None:
            best_path, _ = two_opt(best_path, G_m)
        elif not start_node is None:
            best_path, _ = anchored_two_opt(best_path, G_m)
        else:
            best_path, _ = one_way_two_opt(best_path, G_m)

        if k_neighbors > 0:
            best_path, _ = targeted_three_opt(best_path, G_m, k_neighbors)

        # We just corrected segments of 2 and 3,
        # and this is top level,
        # so correct segments of 4 to 7.
        restitch(G_m, best_path, True)

    best_weight = path_length(best_path, G_m)

    return [nodes[x] for x in best_path], best_weight


def tsp_symmetric(G, start_node=None, end_node=None, quality=None, shots=None, monte_carlo=True, k_neighbors=20, is_cyclic=True, multi_start=1, is_top_level=True):
    nodes = None
    n_nodes = 0
    G_m = None
    if isinstance(G, nx.Graph):
        nodes = list(G.nodes())
        n_nodes = len(nodes)
        G_m = nx.to_numpy_array(G, weight='weight', nonedge=0.0)
    else:
        n_nodes = len(G)
        nodes = list(range(n_nodes))
        G_m = G

    if is_cyclic:
        start_node = None
        end_node = None

    if n_nodes < 7:
        if n_nodes > 3:
            if is_cyclic:
                best_path, best_weight = tsp_bruteforce_cyclic(G_m, list(itertools.permutations(list(range(1, n_nodes)))))
            else:
                best_path, best_weight = tsp_bruteforce_acyclic(G_m, list(itertools.permutations(list(range(n_nodes)))))

            return [nodes[x] for x in best_path], best_weight

        return tsp_symmetric_brute_force_driver(G_m, n_nodes, nodes, is_cyclic)

    if (start_node is None) and not (end_node is None):
        start_node = end_node
        end_node = None

    a = []
    b = []
    c = []
    if (start_node is None) and (end_node is None):
        if monte_carlo:
            a, b = monte_carlo_loop(n_nodes)
        else:
            best_energy = float("inf")
            for _ in range(multi_start):
                bits = ([], [])
                while (len(bits[0]) == 0) or (len(bits[1]) == 0):
                    _, _, bits, energy = spin_glass_solver(G_m, quality=quality, shots=shots)
                if energy < best_energy:
                    best_energy = energy
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

    sol_a = tsp_symmetric(G_a, quality=quality, monte_carlo=monte_carlo, is_cyclic=False, is_top_level=False, k_neighbors=0, multi_start=multi_start)
    sol_b = tsp_symmetric(G_b, quality=quality, monte_carlo=monte_carlo, is_cyclic=False, is_top_level=False, k_neighbors=0, multi_start=multi_start)

    return tsp_symmetric_driver(G_m, is_cyclic, is_top_level, start_node, end_node, k_neighbors, nodes, sol_a, sol_b, a, b, c[0] if len(c) else None)


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
def tsp_asymmetric_driver(G_m, is_reversed, is_cyclic, is_top_level, start_node, end_node, k_neighbors, nodes, sol_a, sol_b, a, b, c):
    path_a = [a[x] for x in sol_a[0]]
    path_b = [b[x] for x in sol_b[0]]

    restitch(G_m, path_a, False)
    restitch(G_m, path_b, False)

    if start_node is None:
        best_path = stitch_asymmetric(G_m, path_a, path_b)
    else:
        best_path = path_a + path_b

    if not c is None:
        best_path.append(c)

    if is_top_level:
        if is_cyclic:
            best_path += [best_path[0]]
            final_path, best_weight = two_opt(best_path.copy(), G_m)
            best_path.reverse()
            path, weight = two_opt(best_path, G_m)
            if weight < best_weight:
                final_path, best_weight = path, weight
        elif not end_node is None:
            final_path, best_weight = two_opt(best_path.copy(), G_m)
            best_path.reverse()
            path, weight = two_opt(best_path, G_m)
            if weight < best_weight:
                final_path, best_weight = path, weight
        elif not start_node is None:
            final_path, best_weight = anchored_two_opt(best_path.copy(), G_m)
            best_path.reverse()
            path, weight = reversed_anchored_two_opt(best_path, G_m)
            if weight < best_weight:
                final_path, best_weight = path, weight
        else:
            final_path, best_weight = one_way_two_opt(best_path.copy(), G_m)
            best_path.reverse()
            path, weight = one_way_two_opt(best_path, G_m)
            if weight < best_weight:
                final_path, best_weight = path, weight

        if k_neighbors > 0:
            best_path = final_path
            final_path, best_weight = targeted_three_opt(best_path.copy(), G_m, k_neighbors)
            best_path.reverse()
            path, weight = targeted_three_opt(best_path.copy(), G_m, k_neighbors)
            if weight < best_weight:
                final_path, best_weight = path, weight

        # We just corrected segments of 2 and 3,
        # and this is top level,
        # so correct segments of 4 to 7.
        restitch(G_m, final_path, True)
    else:
        final_path = best_path

    best_weight = path_length(final_path, G_m)

    if is_reversed:
        final_path.reverse()

    return [nodes[x] for x in final_path], best_weight


def tsp_asymmetric(G, start_node=None, end_node=None, quality=None, shots=None, monte_carlo=True, k_neighbors=20, is_cyclic=True, multi_start=1, is_top_level=True):
    nodes = None
    n_nodes = 0
    G_m = None
    if isinstance(G, nx.Graph):
        nodes = list(G.nodes())
        n_nodes = len(nodes)
        G_m = nx.to_numpy_array(G, weight='weight', nonedge=0.0)
    else:
        n_nodes = len(G)
        nodes = list(range(n_nodes))
        G_m = G

    if is_cyclic:
        start_node = None
        end_node = None

    if n_nodes < 7:
        if n_nodes > 2:
            if is_cyclic:
                best_path, best_weight = tsp_bruteforce_cyclic(G_m, list(itertools.permutations(list(range(1, n_nodes)))))
            else:
                best_path, best_weight = tsp_bruteforce_acyclic(G_m, list(itertools.permutations(list(range(n_nodes)))))

            return [nodes[x] for x in best_path], best_weight

        return tsp_asymmetric_brute_force_driver(G_m, n_nodes, nodes, is_cyclic)


    is_reversed = False
    if (start_node is None) and not (end_node is None):
        is_reversed = True
        start_node = end_node
        end_node = None
        G_m = G_m.T

    a = []
    b = []
    c = []
    if (start_node is None) and (end_node is None):
        if monte_carlo:
            a, b = monte_carlo_loop(n_nodes)
        else:
            best_energy = float("inf")
            for _ in range(multi_start):
                bits = ([], [])
                while (len(bits[0]) == 0) or (len(bits[1]) == 0):
                    _, _, bits, energy = spin_glass_solver((G_m + G_m.T) / 2, quality=quality, shots=shots)
                if energy < best_energy:
                    best_energy = energy
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

    sol_a = tsp_asymmetric(G_a, quality=quality, monte_carlo=monte_carlo, is_cyclic=False, is_top_level=False, k_neighbors=0, multi_start=multi_start)
    sol_b = tsp_asymmetric(G_b, quality=quality, monte_carlo=monte_carlo, is_cyclic=False, is_top_level=False, k_neighbors=0, multi_start=multi_start)

    return tsp_asymmetric_driver(G_m, is_reversed, is_cyclic, is_top_level, start_node, end_node, k_neighbors, nodes, sol_a, sol_b, a, b, c[0] if len(c) else None)
