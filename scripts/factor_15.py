import itertools
import networkx as nx

from pyqrackising import tsp_symmetric

def toy_factoring_as_tsp(N, factors=None, cyclic=False, fixed_endpoints=None):
    """
    Toy example: encode factoring as a TSP instance.

    Each "city" corresponds to a state (bit, carry) in the multiplication path.
    The tour cost encodes whether the assignment multiplies correctly to N.

    Parameters:
        N (int): modulus to factor
        factors (tuple or None): if provided, the true factors (for toy check)
        cyclic (bool): if True, require cycle; else path.
        fixed_endpoints (tuple or None): if path and endpoints fixed, (start, end).

    Returns:
        G (networkx.Graph): weighted TSP-style graph.
    """
    # Represent N in binary
    n_bits = N.bit_length()

    # For toy demo, define "cities" = bit positions of candidate factors
    cities = [f"bit{i}" for i in range(n_bits)]

    G = nx.Graph()
    G.add_nodes_from(cities)

    # Define edge weights as penalties for inconsistent bit assignments
    # Here: just a heuristic placeholder encoding
    for (u, v) in itertools.combinations(cities, 2):
        iu, iv = int(u[3:]), int(v[3:])
        weight = abs(iu - iv)
        G.add_edge(u, v, weight=weight)

    # Optionally enforce path endpoints by adding dummy nodes
    if not cyclic and fixed_endpoints:
        start, end = fixed_endpoints
        G.add_node(start)
        G.add_node(end)
        for city in cities:
            G.add_edge(start, city, weight=0.5)
            G.add_edge(end, city, weight=0.5)

    return G

# Example: encode factoring 15 (3*5) as TSP-like problem
G_toy = toy_factoring_as_tsp(15, factors=(3,5), cyclic=False, fixed_endpoints=("start","end"))
print(tsp_symmetric(G_toy, start_node="start", end_node="end", is_cyclic=False))
