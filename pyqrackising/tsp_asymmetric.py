# Mapping from ATSP to TSP, wholly by Elara (the OpenAI GPT)

from .tsp_symmetric import tsp_symmetric
import networkx as nx
import numpy as np


def atsp_to_symmetric(G_asym):
    """
    Transform an asymmetric TSP (ATSP) instance into a symmetric TSP instance
    using the standard 2n-node construction.

    Args:
        G_asym (np.ndarray): n x n adjacency matrix of ATSP distances.
                             Assumes G_asym[i, i] = 0.

    Returns:
        G_sym (np.ndarray): 2n x 2n symmetric adjacency matrix
                            corresponding to the transformed TSP.
    """
    n = len(G_asym)
    N = 2 * n
    G_sym = np.full((N, N), np.inf, dtype=np.float64)

    for i in range(n):
        in_i = 2 * i
        out_i = 2 * i + 1

        # Zero edge between in-node and out-node
        G_sym[in_i, out_i] = 0.0
        G_sym[out_i, in_i] = 0.0  # symmetric

        # Map asymmetric costs into symmetric structure
        for j in range(n):
            if i == j:
                continue
            in_j = 2 * j
            out_j = 2 * j + 1
            cost = G_asym[i, j]
            G_sym[out_i, in_j] = cost
            G_sym[in_j, out_i] = cost  # symmetric

    # Replace infinities with a large number (or leave as inf if solver can handle it)
    G_sym[G_sym == np.inf] = 1e256

    return G_sym


def symmetric_to_atsp_path(sym_path, n):
    """
    Convert a tour from the symmetric 2n-node TSP back to an n-node ATSP tour.

    Args:
        sym_path (list[int]): Tour from tsp_symmetric() on 2n-node graph.
        n (int): Number of cities in the original ATSP.

    Returns:
        atsp_path (list[int]): Tour over the original n nodes.
    """
    atsp_path = []
    for node in sym_path:
        if node % 2 == 0:  # only keep "in" nodes (even indices)
            atsp_path.append(node // 2)
    return atsp_path


def digraph_to_asym_matrix(G):
    """
    Convert a networkx.DiGraph to a numpy adjacency matrix.
    """
    n = G.number_of_nodes()
    mapping = dict(zip(G.nodes(), range(n)))
    G = nx.relabel_nodes(G, mapping)  # ensure 0..n-1
    G_asym = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(G_asym, 0.0)

    for u, v, d in G.edges(data=True):
        G_asym[u, v] = d.get("weight", 1.0)

    return G_asym


def symmetric_to_atsp_path_and_weight(sym_path, G_asym):
    """
    Convert a symmetric 2n-node TSP tour back into an ATSP tour
    and compute its true weight.

    Args:
        sym_path (list[int]): Tour from tsp_symmetric() on 2n-node graph.
        G_asym (np.ndarray): Original n x n asymmetric adjacency matrix.

    Returns:
        atsp_path (list[int]): Tour over original n nodes.
        atsp_weight (float): Total weight of the ATSP tour.
    """
    n = len(G_asym)
    atsp_path = []
    for node in sym_path:
        if node % 2 == 0:  # only keep "in" nodes
            atsp_path.append(node // 2)

    # Compute true ATSP weight
    atsp_weight = 0.0
    for i in range(len(atsp_path) - 1):
        atsp_weight += G_asym[atsp_path[i], atsp_path[i+1]]
    atsp_weight += G_asym[atsp_path[-1], atsp_path[0]]  # close cycle

    return atsp_path, atsp_weight


def tsp_asymmetric(G, start_node=None, quality=1, shots=None, correction_quality=2, monte_carlo=False, k_neighbors=20, is_cyclic=True, multi_start=1, is_top_level=True):
    G_asym = digraph_to_asym_matrix(G) if isinstance(G, nx.DiGraph) else G

    # Transform to symmetric TSP
    G_sym = atsp_to_symmetric(G_asym)

    # Solve
    return symmetric_to_atsp_path_and_weight(tsp_symmetric(
        G_sym,
        start_node=start_node,
        quality=quality,
        shots=shots,
        correction_quality=correction_quality,
        monte_carlo=monte_carlo,
        k_neighbors=k_neighbors,
        is_cyclic=is_cyclic,
        multi_start=multi_start,
        is_top_level=is_top_level
    )[0], G_asym)
