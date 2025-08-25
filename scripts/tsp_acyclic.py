# Traveling Salesman Problem (considered NP-complete)
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
    # NP-complete TSP
    n_nodes = 64
    G = generate_tsp_graph(n_nodes=n_nodes, seed=42)
    best_circuit, best_path_length = tsp_symmetric(
        G, start_node=0, is_cyclic=False
    )
    for i in range(15):
        circuit, path_length = tsp_symmetric(G, start_node=0, is_cyclic=False)
        if path_length < best_path_length:
            best_circuit = circuit
            best_path_length = path_length

    reconstructed_node_count = len(set(best_circuit))
    reconstructed_path_length = 0
    for i in range(len(best_circuit) - 1):
        reconstructed_path_length += G[best_circuit[i]][best_circuit[i + 1]]["weight"]

    print(f"Path: {best_circuit}")
    print(f"Actual node count: {n_nodes}")
    print(f"Solution distinct node count: {reconstructed_node_count}")
    print(f"Claimed path length: {best_path_length}")
    print(f"Verified path length: {reconstructed_path_length}")
    print(
        "(The average randomized and normalized separation between each and every node is about 0.5.)"
    )
