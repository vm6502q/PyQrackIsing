import networkx as nx
import numpy as np
import tfim_sampler
from numba import njit


# Written by Elara (OpenAI custom GPT)
def local_repulsion_choice(nodes, adjacency, degrees, weights, n, m):
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
    available = set(range(len(nodes)))

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
        for nbr in adjacency.get(nodes[node], []):
            idx = nodes.index(nbr)
            if idx in available:
                weights[idx] *= 0.5  # halve neighbor's weight (tunable!)

    # Build integer mask
    mask = 0
    for pos in chosen:
        mask |= (1 << pos)

    return mask


def evaluate_cut_edges(samples, edge_keys, edge_values):
    best_value = float("-inf")
    best_solution = None
    best_cut_edges = None

    for state in samples:
        cut_edges = []
        cut_value = 0
        for i in range(len(edge_values)):
            k = i << 1
            u, v = edge_keys[k], edge_keys[k + 1]
            if ((state >> u) & 1) != ((state >> v) & 1):
                cut_value += edge_values[i]

        if cut_value > best_value:
            best_value = cut_value
            best_solution = state

    return best_solution, float(best_value)


# By Gemini (Google Search AI)
def int_to_bitstring(integer, length):
    return (bin(integer)[2:].zfill(length))[::-1]


def maxcut_tfim(
    G,
    quality = 12,
    shots = None,
):
    # Number of qubits/nodes
    nodes = list(G.nodes())
    n_qubits = len(nodes)

    if n_qubits == 0:
        return '', 0, ([], [])

    if shots is None:
        # Number of measurement shots
        shots = n_qubits << quality

    J_eff = np.array([-sum(edge_attributes.get('weight', 1.0) for _, edge_attributes in G.adj[n].items()) for n in nodes], dtype=np.float64)
    degrees = np.array([sum(abs(edge_attributes.get('weight', 1.0)) for _, edge_attributes in G.adj[n].items()) for n in nodes], dtype=np.float64)
    thresholds = tfim_sampler._maxcut_hamming_cdf(n_qubits, J_eff, degrees, quality)
    G_dict = nx.to_dict_of_lists(G)
    J_max = max(J_eff)
    weights = 1.0 / (1.0 + (J_max - J_eff))
    samples = []
    for s in range(shots):
        # First dimension: Hamming weight
        mag_prob = np.random.random()
        m = 0
        while thresholds[m] < mag_prob:
            m += 1
        m += 1
        # Second dimension: permutation within Hamming weight
        samples.append(local_repulsion_choice(nodes, G_dict, degrees, weights, n_qubits, m))

    # We only need unique instances
    samples = list(set(samples))

    edge_keys = []
    edge_values = []
    for u, v, data in G.edges(data=True):
        edge_keys.append(nodes.index(u))
        edge_keys.append(nodes.index(v))
        edge_values.append(data.get("weight", 1.0))

    best_solution, best_value = evaluate_cut_edges(samples, edge_keys, edge_values)

    bit_string = int_to_bitstring(best_solution, n_qubits)
    bit_list = list(bit_string)
    l, r = [], []
    for i in range(len(bit_list)):
        b = (bit_list[i] == '1')
        if b:
            r.append(nodes[i])
        else:
            l.append(nodes[i])

    return bit_string, best_value, (l, r)
