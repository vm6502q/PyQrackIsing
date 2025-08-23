from .spin_glass_solver import spin_glass_solver
import networkx as nx


def get_best_stitch(adjacency, terminals_a, terminals_b, is_cyclic):
    best_weight = float("inf")
    best_edge = None
    for a in range(2):
        a_term = terminals_a[a]
        for b in range(2):
            b_term = terminals_b[b]
            weight = adjacency[a_term][b_term]["weight"]
            if is_cyclic:
                n_a_term = terminals_a[0 if a else 1]
                n_b_term = terminals_b[0 if b else 1]
                weight += adjacency[n_a_term][n_b_term]["weight"]
            if weight < best_weight:
                best_weight = weight
                best_edge = (a, b)

    return best_weight, best_edge


def tsp_symmetric(G, quality=2, is_cyclic=True, start_node=None):
    if G.number_of_nodes() == 1:
        return ([0], 0)
    if G.number_of_nodes() == 2:
        return ([0, 1], G[0][1]["weight"])

    nodes = list(G.nodes())

    a = []
    b = []
    if not (start_node is None):
        a = [start_node]
        b = nodes
        b.remove(start_node)
    else:
        while (len(a) == 0) or (len(b) == 0):
            bitstring, _, _, _ = spin_glass_solver(G, quality=quality)
            for idx, bit in enumerate(bitstring):
                if bit == "1":
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

    sol_a = (
        tsp_symmetric(G_a, quality=quality, is_cyclic=False)
        if len(a) > 2
        else (([0, 1], G_a[0][1]["weight"]) if len(a) == 2 else ([0], 0))
    )
    sol_b = (
        tsp_symmetric(G_b, quality=quality, is_cyclic=False)
        if len(b) > 2
        else (([0, 1], G_b[0][1]["weight"]) if len(b) == 2 else ([0], 0))
    )
    sol_weight = sol_a[1] + sol_b[1]

    path_a = [a[idx] for idx in sol_a[0]]
    path_b = [b[idx] for idx in sol_b[0]]

    terminals_a = (path_a[0], path_a[-1])
    terminals_b = (path_b[0], path_b[-1])

    # Find the best edge to stitch "a" to "b"
    best_weight, best_end = get_best_stitch(G, terminals_a, terminals_b, is_cyclic)

    if best_end[0] == 0:
        path_a.reverse()
    if best_end[1] == 1:
        path_b.reverse()

    return (path_a + path_b, sol_weight + best_weight)
