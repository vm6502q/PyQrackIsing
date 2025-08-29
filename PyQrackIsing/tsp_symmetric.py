from .spin_glass_solver import spin_glass_solver
import networkx as nx
from numba import njit
import numpy as np



# By Gemini (Google Search AI)
def int_to_bitstring(integer, length):
    return (bin(integer)[2:].zfill(length))[::-1]


# two_opt() and targeted_three_opt() written by Elara (OpenAI ChatGPT instance)
@njit
def path_length(path, G_m):
    tot_len = 0.0
    for i in range(len(path)-1):
        tot_len += G_m[path[i], path[i+1]]

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
            for j in range(i+1, path_len + 1):
                if j - i == 1:  # adjacent edges, skip
                    continue
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
        for i in range(1, path_len - 1):
            for j in range(i+1, path_len):
                if j - i == 1:  # adjacent edges, skip
                    continue
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

    while improved:
        improved = False
        for i in range(1, len(path) - 2):
            for j in range(i+1, len(path) - 1):
                if j - i == 1:  # adjacent edges, skip
                    continue
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
def stitch(G_m, path_a, path_b, sol_weight):
    is_single_a = len(path_a) == 1
    is_single_b = len(path_b) == 1
    best_path = [0]
    best_weight = 0.0
    if is_single_a or is_single_b:
        singlet = 0
        bulk = [0]
        if is_single_a:
            singlet = path_a[0]
            bulk = path_b
        elif is_single_b:
            singlet = path_b[0]
            bulk = path_a

        best_weight = G_m[singlet, bulk[0]]
        best_path = [singlet] + bulk
        weight = G_m[singlet, bulk[-1]]
        if weight < best_weight:
            best_weight = weight
            best_path = bulk + [singlet]
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
    else:
        terminals_a = [path_a[0], path_a[-1]]
        terminals_b = [path_b[0], path_b[-1]]

        best_weight = G_m[terminals_a[1], terminals_b[0]]
        best_path = path_a + path_b
        weight = G_m[terminals_a[0], terminals_b[1]]
        if weight < best_weight:
            best_weight = weight
            best_path = path_b + path_a
        for _ in range(2):
            for _ in range(2):
                for i in range(1, len(path_b)):
                    weight = (
                        G_m[terminals_a[0], path_b[i - 1]] +
                        G_m[terminals_a[1], path_b[i]] -
                        G_m[path_b[i - 1], path_b[i]]
                    )
                    if weight < best_weight:
                        best_weight = weight
                        best_path = path_b.copy()
                        best_path[i:i] = path_a
                path_a.reverse()
                terminals_a.reverse()
            path_a, path_b = path_b, path_a
            terminals_a, terminals_b = terminals_b, terminals_a

    return best_path, best_weight


def tsp_symmetric(G, start_node=None, end_node=None, quality=1, shots=None, correction_quality=2, monte_carlo=False, is_3_opt=True, k_neighbors=20, is_cyclic=True, multi_start=1, is_top_level=True):
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

    if n_nodes == 0:
        return ([], 0)
    if n_nodes == 1:
        return ([nodes[0]], 0)
    if n_nodes == 2:
        if is_cyclic:
            return ([nodes[0], nodes[1], nodes[0]], 2 * G_m[0, 1])
        else:
            return ([nodes[0], nodes[1]], G_m[0, 1])

    if (start_node is None) and not (end_node is None):
        start_node = end_node
        end_node = None

    a = []
    b = []
    c = []
    if (start_node is None) and (end_node is None):
        best_energy = float("inf")
        for _ in range(multi_start):
            energy = 0.0
            _a = []
            _b = []
            while (len(_a) == 0) or (len(_b) == 0):
                bits = ([], [])
                if monte_carlo:
                    for i in range(n_nodes):
                        if np.random.random() < 0.5:
                            bits[0].append(i)
                        else:
                            bits[1].append(i)
                else:
                    _, _, bits, energy = spin_glass_solver(G_m, quality=quality, shots=shots, correction_quality=correction_quality)
                _a = list(bits[0])
                _b = list(bits[1])
            if energy < best_energy:
                best_energy = energy
                a, b = _a, _b
    else:
        is_cyclic = False
        a.append(nodes.index(start_node))
        b = list(range(n_nodes))
        b.remove(a[0])
        if end_node is not None:
            c.append(nodes.index(end_node))
            b.remove(c[0])

    G_a, G_b = init_G_a_b(G_m, a, b)

    sol_a = tsp_symmetric(G_a, quality=quality, correction_quality=correction_quality, monte_carlo=monte_carlo, is_cyclic=False, is_top_level=False, is_3_opt=False, multi_start=multi_start)
    sol_b = tsp_symmetric(G_b, quality=quality, correction_quality=correction_quality, monte_carlo=monte_carlo, is_cyclic=False, is_top_level=False, is_3_opt=False, multi_start=multi_start)

    path_a = [a[x] for x in sol_a[0]]
    path_b = [b[x] for x in sol_b[0]]

    sol_weight = sol_a[1] + sol_b[1]

    if len(c):
        sol_weight += G_m[b[-1], c[0]]

    if (len(path_a) == 1) and (len(path_b) == 1):
        if len(c):
            return (path_a + path_b + c, (sol_weight + G_m[path_b[0], c[0]] + G_m[path_a[0], c[0]]) if is_cyclic else (sol_weight + G_m[path_b[0], c[0]]))
        return (path_a + path_b, (sol_weight + G_m[path_a[0], path_b[0]]) if is_cyclic else sol_weight)

    if start_node is None:
        best_path, best_weight = stitch(G_m, path_a, path_b, sol_weight)
    else:
        best_path = path_a + path_b
        best_weight = G_m[path_a[0], path_b[0]]
        weight = G_m[path_a[0], path_b[-1]]
        if weight < best_weight:
            path_b.reverse()
            best_path = path_a + path_b
            best_weight = weight
        best_weight += sol_weight

    if len(c):
        best_path += c
        best_weight += G_m[best_path[-1], c[0]]

    if is_top_level:
        if is_cyclic:
            best_path += [best_path[0]]
            best_path, best_weight = two_opt(best_path, G_m)
        elif not end_node is None:
            best_path, best_weight = two_opt(best_path, G_m)
        elif not start_node is None:
            best_path.reverse()
            best_path, best_weight = anchored_two_opt(best_path, G_m)
            best_path.reverse()
        else:
            best_path, best_weight = one_way_two_opt(best_path, G_m)

        if is_3_opt:
            best_path, best_weight = targeted_three_opt(best_path, G_m, k_neighbors)
    else:
        best_path, best_weight = one_way_two_opt(best_path, G_m)

        if is_cyclic:
            cycle_node = best_path[0]
            best_weight += G_m[cycle_node, best_path[-1]]
            best_path += [cycle_node]

    return [nodes[x] for x in best_path], best_weight
