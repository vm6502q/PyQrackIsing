import networkx as nx
import numpy as np
from numba import njit, prange


@njit
def probability_by_hamming_weight(J, h, z, theta, t, n_qubits):
    # critical angle
    theta_c = np.arcsin(
        max(
            -1.0,
            min(
                1.0,
                (1.0 if J > 0.0 else -1.0) if np.isclose(abs(z * J), 0.0) else (abs(h) / (z * J)),
            ),
        )
    )
    delta_theta = theta - theta_c
    bias = np.zeros(n_qubits + 1)

    if np.isclose(abs(h), 0.0):
        bias[0] = 1.0
    elif np.isclose(abs(J), 0.0):
        bias.fill(1.0 / (n_qubits + 1.0))
    else:
        sin_delta = np.sin(delta_theta)
        omega = 1.5 * np.pi
        t2 = 1.0
        p = (
            pow(2.0, abs(J / h) - 1.0)
            * (1.0 + sin_delta * np.cos(J * omega * t + theta) / (1.0 + np.sqrt(t / t2)))
            - 0.5
        )

        if p >= 1024:
            bias[0] = 1.0
        else:
            tot_n = 0.0
            for q in range(n_qubits + 1):
                n = 1.0 / ((n_qubits + 1) * pow(2.0, p * q))
                bias[q] = n
                tot_n += n
            for q in range(n_qubits + 1):
                bias[q] /= tot_n

    if J > 0.0:
        bias = bias[::-1]

    return bias


@njit(parallel=True)
def maxcut_hamming_cdf(n_qubits, J_func, degrees, mult_log2):
    if n_qubits == 0:
        return np.empty(0, dtype=np.float64)

    n_steps = n_qubits << mult_log2
    shots = n_qubits << mult_log2
    delta_t = 1.0 / (n_steps << (mult_log2 >> 1))
    h_mult = (1 << (mult_log2 >> 1)) / (n_steps * delta_t)
    hamming_prob = np.zeros(n_qubits - 1)

    for step in range(n_steps):
        t = step * delta_t
        tm1 = (step - 1) * delta_t
        for q in prange(n_qubits):
            z = degrees[q]
            J_eff = J_func[q]
            h_t = h_mult * t
            bias = probability_by_hamming_weight(J_eff, h_t, z, 0.0, t, n_qubits)
            if step == 0:
                for i in range(len(hamming_prob)):
                    hamming_prob[i] += bias[i + 1]
                continue

            last_bias = probability_by_hamming_weight(J_eff, h_t, z, 0.0, tm1, n_qubits)
            for i in range(len(hamming_prob)):
                hamming_prob[i] += bias[i + 1] - last_bias[i + 1]

    tot_prob = sum(hamming_prob)
    for i in prange(len(hamming_prob)):
        hamming_prob[i] /= tot_prob

    tot_prob = 0.0
    for i in range(len(hamming_prob)):
        tot_prob += hamming_prob[i]
        hamming_prob[i] = tot_prob
    hamming_prob[-1] = 1.0

    return hamming_prob


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
        mask |= 1 << pos

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
    quality=12,
    shots=None,
):
    # Number of qubits/nodes
    nodes = list(G.nodes())
    n_qubits = len(nodes)

    if n_qubits == 0:
        return "", 0, ([], [])

    if shots is None:
        # Number of measurement shots
        shots = n_qubits << quality

    J_eff = np.array(
        [
            -sum(edge_attributes.get("weight", 1.0) for _, edge_attributes in G.adj[n].items())
            for n in nodes
        ],
        dtype=np.float64,
    )
    degrees = np.array(
        [
            sum(abs(edge_attributes.get("weight", 1.0)) for _, edge_attributes in G.adj[n].items())
            for n in nodes
        ],
        dtype=np.float64,
    )
    # thresholds = tfim_sampler._maxcut_hamming_cdf(n_qubits, J_eff, degrees, quality)
    thresholds = maxcut_hamming_cdf(n_qubits, J_eff, degrees, quality)
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
        b = bit_list[i] == "1"
        if b:
            r.append(nodes[i])
        else:
            l.append(nodes[i])

    return bit_string, best_value, (l, r)
