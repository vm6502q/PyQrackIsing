# MAXCUT
# Produced by Dan Strano, Elara (the OpenAI custom GPT), and Gemini (Google Search AI)

# We reduce transverse field Ising model for globally uniform J and h parameters from a 2^n-dimensional problem to an (n+1)-dimensional approximation that suffers from no Trotter error. Upon noticing most time steps for Quantinuum's parameters had roughly a quarter to a third (or thereabouts) of their marginal probability in |0> state, it became obvious that transition to and from |0> state should dominate the mechanics. Further, the first transition tends to be to or from any state with Hamming weight of 1 (in other words, 1 bit set to 1 and the rest reset 0, or n bits set for Hamming weight of n). Further, on a torus, probability of all states with Hamming weight of 1 tends to be exactly symmetric. Assuming approximate symmetry in every respective Hamming weight, the requirement for the overall probability to converge to 1.0 or 100% in the limit of an infinite-dimensional Hilbert space suggests that Hamming weight marginal probability could be distributed like a geometric series. A small correction to exact symmetry should be made to favor closeness of "like" bits to "like" bits (that is, geometric closeness on the torus of "1" bits to "1" bits and "0" bits to "0" bits), but this does not affect average global magnetization. Adding an oscillation component with angular frequency proportional to J, we find excellent agreement with Trotterization approaching the limit of infinitesimal time step, for R^2 (coefficient of determination) of normalized marginal probability distribution of ideal Trotterized simulation as described by the (n+1)-dimensional approximate model, as well as for R^2 and RMSE (root-mean-square error) of global magnetization curve values.

# After tackling the case where parameters are uniform and independent of time, we generalize the model by averaging per-qubit behavior as if the static case and per-time-step behavior as finite difference. This provides the basis of a novel physics-inspired (adiabatic TFIM) MAXCUT approximate solver that often gives optimal or exact answers on a wide selection of graph types.

import networkx as nx
import numpy as np
from pyqrackising import maxcut_tfim_sparse, spin_glass_solver_sparse

# from cvxgraphalgs.algorithms import goemans_williamson_weighted


# By Gemini (Google Search AI)
def int_to_bitstring(integer, length):
    return (bin(integer)[2:].zfill(length))[::-1]


def compute_energy(bitstring, G):
    theta_bits = [b == "1" for b in list(bitstring)]
    energy = 0
    for u, v, data in G.edges(data=True):
        value = data.get("weight", 1.0)
        spin_u = 1 if theta_bits[u] else -1
        spin_v = 1 if theta_bits[v] else -1
        energy += value * spin_u * spin_v

    return energy


# NP-complete spin glass
def generate_spin_glass_graph(n_nodes=16, degree=3, seed=None):
    if not (seed is None):
        np.random.seed(seed)
    G = nx.random_regular_graph(d=degree, n=n_nodes, seed=seed)
    for u, v in G.edges():
        G[u][v]["weight"] = np.random.choice([-1, 1])  # spin glass couplings
    return G


if __name__ == "__main__":
    # We usually achieve the exact value
    # (or optimal, for Erdős–Rényi, with unknown exact value)
    # for each of the following examples.

    # Example: Peterson graph
    # G = nx.petersen_graph()
    # Known MAXCUT size: 12

    # Example: Icosahedral graph
    # G = nx.icosahedral_graph()
    # Known MAXCUT size: 20

    # Example: Complete bipartite K_{m, n}
    # m, n = 8, 8
    # G = nx.complete_bipartite_graph(m, n)
    # Known MAXCUT size: m * n

    # Generate a "harder" test case: Erdős–Rényi random graph with 20 nodes, edge probability 0.5
    n_nodes = 20
    edge_prob = 0.5
    G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=42)
    # Cut value is approximately 63 for this example.

    # Create a Barabási–Albert (BA) graph with 20 nodes and 2 edges to attach from a new node to existing nodes
    # G = nx.barabasi_albert_graph(n=20, m=2, seed=42)

    # Non-uniform edge weights
    # G = nx.Graph()
    # G.add_edge(0, 1, weight=3.69)
    # G.add_edge(0, 2, weight=2.2)
    # G.add_edge(0, 3, weight=2.26)
    # G.add_edge(0, 4, weight=4.01)
    # G.add_edge(0, 5, weight=1.29)

    # NP-complete spin glass
    # G = generate_spin_glass_graph(seed=42)

    bitstring, cut_value, cut = maxcut_tfim_sparse(G, quality=5)
    # bitstring, cut_value, cut, energy = spin_glass_solver_sparse(G)
    # bitstring, cut_value, cut, energy = spin_glass_solver_sparse(G, best_guess=maxcut_tfim_sparse(G, quality=8)[0])
    print((bitstring, cut_value, cut))

    # cut = goemans_williamson_weighted(G)
    # left = cut.left
    # right = cut.right
    # gw_int = 0
    # for b in right:
    #     gw_int |= 1 << b
    # gw_bit_string = int_to_bitstring(gw_int, G.number_of_nodes())
    # gw_energy = compute_energy(gw_bit_string, G)
    # print(f"GW: {(gw_bit_string, gw_energy)}")

    # bitstring, cut_value, energy = spin_glass_solver_sparse(G, best_guess=gw_bit_string)
    # print(f"vs: {(bitstring, energy)}")
