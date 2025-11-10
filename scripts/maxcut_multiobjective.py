# Random MAXCUT (for execution time tests)
# Produced by Dan Strano, Elara (the OpenAI custom GPT)

from pyqrackising import maxcut_tfim_sparse, spin_glass_solver_sparse
import networkx as nx
import numpy as np
import sys
import time


# Random heavy hex spin glass adjacency matrix
def generate_heavy_hex_graph(
    num_nodes=42,
    num_edges=46,
    mean_weight=0.0,
    std_weight=1.0,
    seed=None,
):
    """
    Generate a random heavy-hex-like graph with Gaussian edge weights.
    
    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    num_edges : int
        Number of edges in the graph.
    mean_weight : float
        Mean of Gaussian-distributed edge weights.
    std_weight : float
        Standard deviation of Gaussian-distributed edge weights.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    G : networkx.Graph
        Weighted heavy-hex-like graph.
    """

    rng = np.random.default_rng(seed)
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    # build a pseudo-lattice backbone
    for i in range(num_nodes - 1):
        if rng.random() < 0.8:  # 80% chance to connect to next node
            G.add_edge(i, (i + 1) % num_nodes)

    # randomly add or remove edges to reach target edge count
    while len(G.edges) < num_edges:
        u, v = rng.integers(0, num_nodes, 2)
        if u != v and not G.has_edge(u, v):
            # Avoid too-high degree nodes (>=3) to mimic heavy hex
            if G.degree(u) < 3 and G.degree(v) < 3:
                G.add_edge(u, v)
    while len(G.edges) > num_edges:
        edge = random.choice(list(G.edges))
        G.remove_edge(*edge)

    # assign Gaussian weights
    for u, v in G.edges:
        G[u][v]["weight"] = rng.normal(mean_weight, std_weight)

    return G


if __name__ == "__main__":
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42

    start = time.perf_counter()
    G = generate_heavy_hex_graph(num_nodes=42, num_edges=46, seed=42)
    seconds = time.perf_counter() - start
    print(f"{seconds} seconds to initialize the graph (statement of the problem itself)")

    print(f"Random seed: {seed}")
    print(f"Node count: 42")
    start = time.perf_counter()
    bitstring, cut_value, cut, energy = spin_glass_solver_sparse(G, is_spin_glass=False)
    seconds = time.perf_counter() - start

    print(f"Seconds to solution: {seconds}")
    print(f"Bipartite cut bit string: {bitstring}")
    print(f"Cut weight: {cut_value}")
