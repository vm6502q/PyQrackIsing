# Spin Glass Ground State (considered NP-complete)
# Produced by Dan Strano, Elara (the OpenAI custom GPT)

from PyQrackIsing import spin_glass_solver, maxcut_tfim
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
            G.add_edge(u, v, weight=np.random.random())
    return G


def get_best_stitch(adjacency, a_term, terminals_b):
    best_weight = float("inf")
    best_edge = None
    for nbr in adjacency.get(a_term, []):
        if nbr not in terminals_b:
            continue
        u, v = (a_term, nbr) if a_term < nbr else (nbr, a_term)
        weight = G[a_term][nbr]["weight"]
        if weight < best_weight:
            best_weight = weight
            best_edge = (u, v)

    return best_weight, best_edge


def recurse_tsp(G):
    if G.number_of_nodes() == 1:
        return ([0], 0)
    if G.number_of_nodes() == 2:
        return ([0, 1], G[0][1])

    nodes = list(G.nodes())

    a = []
    b = []
    while (len(a) == 0) or (len(b) == 0):
        cut_value, bitstring, cut_edges, energy = spin_glass_solver(G)
        for idx, bit in enumerate(bitstring):
            if bit == '1':
                b.append(nodes[idx])
            else:
                a.append(nodes[idx])

    G_a = nx.Graph()
    G_b = nx.Graph()
    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 1.0)
        if (u in a) and (v in a):
            G_a.add_edge(a.index(u), a.index(v), weight=weight)
            continue
        elif (u in b) and (v in b):
            G_b.add_edge(b.index(u), b.index(v), weight=weight)
            continue

    sol_a = recurse_tsp(G_a) if len(a) > 2 else (([0, 1], G_a[0][1]['weight']) if len(a) == 2 else ([0], 0))
    sol_b = recurse_tsp(G_b) if len(b) > 2 else (([0, 1], G_b[0][1]['weight']) if len(b) == 2 else ([0], 0))
    sol_weight  = sol_a[1] + sol_b[1]

    path_a = [a[idx] for idx in sol_a[0]]
    path_b = [b[idx] for idx in sol_b[0]]

    terminals_a = (path_a[0], path_a[-1])
    terminals_b = (path_b[0], path_b[-1])

    # Find the best edge to stitch "a" to "b"
    
    adjacency = nx.to_dict_of_lists(G)
    best_weight, best_edge = get_best_stitch(adjacency, terminals_a[0], terminals_b)
    weight, edge = get_best_stitch(adjacency, terminals_a[1], terminals_b)
    if weight < best_weight:
        best_weight = weight
        best_edge = edge

    if best_edge[0] == terminals_a[0]:
        path_a.reverse()
    if best_edge[1] == terminals_b[1]:
        path_b.reverse()

    return (path_a + path_b, sol_weight + best_weight)

if __name__ == "__main__":
    # NP-complete spin glass
    n_nodes = 128
    G = generate_tsp_graph(n_nodes=n_nodes, seed=42)
    circuit, path_length = recurse_tsp(G)

    print(f"Node count: {n_nodes}")
    print(f"Path: {circuit}")
    print(f"Path length: {path_length}")
    print("(The average randomized and normalized separation between each and every node is about 0.5.)")
