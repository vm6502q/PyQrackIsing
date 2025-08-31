# Spin Glass Ground State (considered NP-complete)
# Produced by Dan Strano, Elara (the OpenAI custom GPT)

from PyQrackIsing import spin_glass_solver
import networkx as nx
import numpy as np


# NP-complete spin glass
def generate_spin_glass_graph(n_nodes=64, degree=3, seed=None):
    if not (seed is None):
        np.random.seed(seed)
    G = nx.random_regular_graph(d=degree, n=n_nodes, seed=seed)
    for u, v in G.edges():
        G[u][v]["weight"] = np.random.choice([-1, 1])  # spin glass couplings
    return G


if __name__ == "__main__":
    # NP-complete spin glass
    G = generate_spin_glass_graph(n_nodes=64, seed=42)
    best_bitstring, best_cut_value, best_cut, best_energy = spin_glass_solver(G)
    for i in range(15):
        bitstring, cut_value, cut, energy = spin_glass_solver(G)
        if energy < best_energy:
            best_bitstring = bitstring
            best_cut_value = cut_value
            best_cut = cut
            best_energy = energy

    print((best_bitstring, best_cut_value, best_cut))
    print(f"Ground State Energy: {best_energy}")
