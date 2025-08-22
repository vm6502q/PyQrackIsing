# Spin Glass Ground State (considered NP-complete)
# Produced by Dan Strano, Elara (the OpenAI custom GPT)

from PyQrackIsing import tsp_symmetric
import networkx as nx
import numpy as np

# Traveling Salesman Problem (normalized to longest segment)
def generate_tsp_graph(n_nodes=64, seed=None):
    if not (seed is None):
        np.random.seed(seed)
    G = nx.Graph()
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            G.add_edge(u, v, weight=np.random.random())
    return G


if __name__ == "__main__":
    # NP-complete spin glass
    n_nodes = 64
    G = generate_tsp_graph(n_nodes=n_nodes, seed=42)
    circuit, path_length = tsp_symmetric(G, quality=4)

    print(f"Node count: {n_nodes}")
    print(f"Path: {circuit}")
    print(f"Path length: {path_length}")
    print("(The average randomized and normalized separation between each and every node is about 0.5.)")
