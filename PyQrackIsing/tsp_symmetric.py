from .spin_glass_solver import spin_glass_solver
import networkx as nx


# two_opt() written by Elara (OpenAI ChatGPT instance)
def two_opt(path, G):
    improved = True
    best_path = path
    best_dist = path_length(best_path, G)
    
    while improved:
        improved = False
        for i in range(1, len(path) - 2):
            for j in range(i+1, len(path) - 1):
                if j - i == 1:  # adjacent edges, skip
                    continue
                new_path = best_path[:]
                new_path[i:j] = best_path[j-1:i-1:-1]  # reverse segment
                new_dist = path_length(new_path, G)
                if new_dist < best_dist:
                    best_path, best_dist = new_path, new_dist
                    improved = True
        path = best_path
    return best_path, best_dist

def path_length(path, G):
    return sum(G[path[i]][path[i+1]].get("weight", 1.0) for i in range(len(path)-1))

def tsp_symmetric(G, quality=0, shots=None, correction_quality=2, is_cyclic=True, start_node=None):
    nodes = list(G.nodes())
    n_nodes = len(nodes)

    if n_nodes == 0:
        return ([], 0)
    if n_nodes == 1:
        return ([nodes[0]], 0)
    if n_nodes == 2:
        return ([nodes[0], nodes[1]], G[nodes[0]][nodes[1]].get("weight", 1.0))

    a = []
    b = []
    if not (start_node is None):
        a = [start_node]
        b = nodes
        b.remove(start_node)
    else:
        while (len(a) == 0) or (len(b) == 0):
            bits = ''
            _, _, bits, _ = spin_glass_solver(G, quality=quality, shots=shots, correction_quality=correction_quality)
            a = list(bits[0])
            b = list(bits[1])

    G_a = nx.Graph()
    G_b = nx.Graph()
    G_a.add_nodes_from(a)
    G_b.add_nodes_from(b)
    for u, v, data in G.edges(data=True):
        if (u in a) and (v in a):
            G_a.add_edge(u, v, weight=data.get("weight", 1.0))
            continue

        if (u in b) and (v in b):
            G_b.add_edge(u, v, weight=data.get("weight", 1.0))

    sol_a = tsp_symmetric(G_a, quality=quality, is_cyclic=False)
    sol_b = tsp_symmetric(G_b, quality=quality, is_cyclic=False)

    path_a = sol_a[0]
    path_b = sol_b[0]

    sol_weight = sol_a[1] + sol_b[1]

    single = None
    is_single_a = len(path_a) == 1
    is_single_b = len(path_b) == 1

    if is_single_a and is_single_b:
        return (path_a + path_b, sol_weight + G[path_a[0]][path_b[0]].get("weight", 1.0))

    singlet = None
    bulk = None
    if is_single_a:
        singlet = path_a[0]
        bulk = path_b
    elif is_single_b:
        singlet = path_b[0]
        bulk = path_a

    if not singlet is None:
        best_weight = G[singlet][bulk[0]].get("weight", 1.0)
        best_path = [singlet] + bulk
        weight = G[singlet][bulk[-1]].get("weight", 1.0)
        if weight < best_weight:
            best_weight = weight
            best_path = bulk + [singlet]
        for i in range(len(bulk) - 1):
            weight = (
                G[singlet][bulk[i]].get("weight", 1.0) +
                G[singlet][bulk[i + 1]].get("weight", 1.0) -
                G[bulk[i]][bulk[i + 1]].get("weight", 1.0)
            )
            if weight < best_weight:
                best_weight = weight
                best_path = bulk.copy().insert(singlet, i + 1)

        best_path, best_weight = two_opt(best_path, G)
 
        if is_cyclic:
            best_weight += G[best_path[-1]][best_path[0]].get("weight", 1.0)
            best_path += [best_path[0]]

        return best_path, best_weight

    terminals_a = [path_a[0], path_a[-1]]
    terminals_b = [path_b[0], path_b[-1]]

    for _ in range(2):
        for _ in range(2):
            best_weight = G[terminals_a[1]][terminals_b[0]].get("weight", 1.0)
            best_path = path_a + path_b
            weight = G[terminals_b[1]][terminals_a[0]].get("weight", 1.0)
            if weight < best_weight:
                best_weight = weight
                best_path = path_b + path_a
            for i in range(len(path_b) - 1):
                weight = (
                    G[terminals_a[0]][path_b[i]].get("weight", 1.0) +
                    G[terminals_a[1]][path_b[i + 1]].get("weight", 1.0) -
                    G[path_b[i]][path_b[i + 1]].get("weight", 1.0)
                )
                if weight < best_weight:
                    best_weight = weight
                    best_path = path_b.copy()
                    best_path[i + 1:i + 1] = path_a
            path_a.reverse()
            terminals_a.reverse()
        path_a, path_b = path_b, path_a
        terminals_a, terminals_b = terminals_b, terminals_a

    best_path, best_weight = two_opt(best_path, G)
 
    if is_cyclic:
        best_weight += G[best_path[-1]][best_path[0]].get("weight", 1.0)
        best_path += [best_path[0]]

    return best_path, best_weight
