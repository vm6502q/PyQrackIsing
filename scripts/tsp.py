# Spin Glass Ground State (considered NP-complete)
# Produced by Dan Strano, Elara (the OpenAI custom GPT)

from PyQrackIsing import spin_glass_solver
import networkx as nx
import numpy as np

# Traveling Salesman Problem (normalized to longest segment)
def generate_tsp_graph(n_nodes=64, seed=None):
    if not (seed is None):
        np.random.seed(seed)
    G = nx.Graph()
    for u in range(n_nodes):
        for v in range(u, n_nodes):
            if u == v:
                continue
            G.add_edge(u, v, weight=-np.random.random())
    return G


if __name__ == "__main__":
    # NP-complete spin glass
    G = generate_tsp_graph(n_nodes=64, seed=42)
    cut_value, bitstring, cut_edges, energy = spin_glass_solver(G)

    print((cut_value, bitstring, cut_edges))
    print(f"Ground State Energy: {energy}")
