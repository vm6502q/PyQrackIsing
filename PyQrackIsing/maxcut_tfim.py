import networkx as nx
import numpy as np
import tfim_sampler
from numba import njit


# Written by Elara (OpenAI custom GPT)
def local_repulsion_choice(adjacency, degrees, weights, n, m):
    """
    Pick m nodes (bit positions) out of n with repulsion bias:
    - High-degree nodes are already less likely
    - After choosing a node, its neighbors' probabilities are further reduced
    """

    # Base weights: inverse degree
    # degrees = np.array([len(adjacency.get(i, [])) for i in range(n)], dtype=np.float64)
    # weights = 1.0 / (degrees + 1.0)
    weights = weights.copy()

    chosen = []
    available = set(range(n))

    for _ in range(m):
        if not available:
            break

        # Normalize weights over remaining nodes
        sub_weights = np.array([weights[i] for i in available], dtype=np.float64)
        sub_weights /= sub_weights.sum()
        sub_nodes = list(available)

        # Sample one node
        idx = np.random.choice(len(sub_nodes), p=sub_weights)
        node = sub_nodes[idx]
        chosen.append(node)

        # Remove node from available
        available.remove(node)

        # Repulsion: penalize neighbors
        for nbr in adjacency.get(node, []):
            if nbr in available:
                weights[nbr] *= 0.5  # halve neighbor's weight (tunable!)

    # Build integer mask
    mask = 0
    for pos in chosen:
        mask |= (1 << pos)

    return mask


@njit
def evaluate_cut_edges(samples, flat_edges):
    best_value = -1
    best_solution = None
    best_cut_edges = None
    for state in samples:
        cut_edges = []
        for i in range(len(flat_edges) // 2):
            i2 = i << 1
            u, v = flat_edges[i2], flat_edges[i2 + 1]
            if ((state >> u) & 1) != ((state >> v) & 1):
                cut_edges.append((u, v))
        cut_size = len(cut_edges)
        if cut_size > best_value:
            best_value = cut_size
            best_solution = state
            best_cut_edges = cut_edges

    return best_value, best_solution, best_cut_edges


# By Gemini (Google Search AI)
def int_to_bitstring(integer, length):
    return (bin(integer)[2:].zfill(length))[::-1]


def maxcut_tfim(
    G,
    quality = 12,
    shots = None,
):
    # Number of qubits/nodes
    n_qubits = G.number_of_nodes()
    if shots is None:
        # Number of measurement shots
        shots = n_qubits << quality

    degrees = np.array([sum(abs(edge_attributes.get('weight', 1.0)) for neighbor, edge_attributes in G.adj[i].items()) for i in range(n_qubits)], dtype=np.float64)
    thresholds = tfim_sampler._maxcut_hamming_cdf(degrees, quality)
    G_dict = nx.to_dict_of_lists(G)
    weights = 1.0 / (degrees + 1.0)
    samples = []
    for s in range(shots):
        # First dimension: Hamming weight
        mag_prob = np.random.random()
        m = 0
        while thresholds[m] < mag_prob:
            m += 1
        m += 1
        # Second dimension: permutation within Hamming weight
        samples.append(local_repulsion_choice(G_dict, degrees, weights, n_qubits, m))

    best_value, best_solution, best_cut_edges = evaluate_cut_edges(samples, [int(item) for tup in G.edges() for item in tup])

    return best_value, int_to_bitstring(best_solution, n_qubits), best_cut_edges
