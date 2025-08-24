import networkx as nx
import random
import time
import math
import pandas as pd
import matplotlib.pyplot as plt

from PyQrackIsing import tsp_symmetric


# Generate a Euclidean TSP instance with n nodes in the unit square
def generate_euclidean_tsp(n, seed=42):
    random.seed(seed)
    points = {i: (random.random(), random.random()) for i in range(n)}
    G = nx.complete_graph(n)
    for u, v in G.edges():
        x1, y1 = points[u]
        x2, y2 = points[v]
        G[u][v]["weight"] = math.hypot(x1 - x2, y1 - y2)
    return G, points


def generate_clustered_tsp(n, clusters=4, spread=0.05, seed=42):
    random.seed(seed)
    points = {}
    centers = [(random.random(), random.random()) for _ in range(clusters)]
    for i in range(n):
        cx, cy = random.choice(centers)
        px = min(max(cx + random.uniform(-spread, spread), 0), 1)
        py = min(max(cy + random.uniform(-spread, spread), 0), 1)
        points[i] = (px, py)

    G = nx.complete_graph(n)
    for u, v in G.edges():
        x1, y1 = points[u]
        x2, y2 = points[v]
        G[u][v]["weight"] = math.hypot(x1 - x2, y1 - y2)
    return G, points


# Heuristic TSP solver (simple nearest neighbor for baseline)
def tsp_nearest_neighbor(G, start=0):
    visited = [start]
    total_weight = 0
    while len(visited) < len(G.nodes):
        last = visited[-1]
        next_node = min(
            (n for n in G.nodes if n not in visited), key=lambda x: G[last][x]["weight"]
        )
        total_weight += G[last][next_node]["weight"]
        visited.append(next_node)
    total_weight += G[visited[-1]][visited[0]]["weight"]  # close the loop
    return visited, total_weight


# Christofides approximation (via networkx)
def tsp_christofides(G):
    return nx.approximation.traveling_salesman_problem(
        G, cycle=True, method=nx.approximation.christofides
    )


# Simulated Annealing for TSP (basic implementation)
def tsp_simulated_annealing(G, temp=1000, cooling=0.995, max_iter=5000):
    nodes = list(G.nodes)
    current = nodes[:]
    random.shuffle(current)
    best = current[:]

    def path_length(path):
        return sum(G[path[i]][path[(i + 1) % len(path)]]["weight"] for i in range(len(path)))

    best_length = path_length(best)
    current_length = best_length

    for _ in range(max_iter):
        i, j = random.sample(range(len(nodes)), 2)
        current[i], current[j] = current[j], current[i]
        new_length = path_length(current)
        delta = new_length - current_length
        if delta < 0 or math.exp(-delta / temp) > random.random():
            current_length = new_length
            if new_length < best_length:
                best = current[:]
                best_length = new_length
        else:
            current[i], current[j] = current[j], current[i]
        temp *= cooling

    return best, best_length


# Validation: check if path is a Hamiltonian cycle
def validate_tsp_solution(G, path):
    return len(path) == len(G.nodes) + 1 and set(path[:-1]) == set(G.nodes)


def get_path_length(G, path):
    return sum(G[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1))


# Benchmark framework with realistic (Euclidean) TSP graphs
def benchmark_tsp_realistic(n_nodes=64, trials=3):
    results = {
        "Nearest Neighbor": [],
        "Christofides": [],
        "Simulated Annealing": [],
        "PyQrackIsing": [],
    }
    G, _ = generate_clustered_tsp(n_nodes)

    # Exclude numba JIT overhead with warmup:
    tsp_symmetric(G)

    for trial in range(trials):
        # Nearest neighbor
        start = time.time()
        path, length = tsp_nearest_neighbor(G)
        results["Nearest Neighbor"].append((time.time() - start, length))
        assert validate_tsp_solution(
            G, path + [path[0]]
        ), f"Invalid nearest neighbor solution in trial {trial}"

        # Christofides
        start = time.time()
        path_c = tsp_christofides(G)
        results["Christofides"].append((time.time() - start, get_path_length(G, path_c)))
        assert validate_tsp_solution(G, path_c), f"Invalid Christofides solution in trial {trial}"

        # Simulated annealing
        start = time.time()
        path_s, length_s = tsp_simulated_annealing(G)
        results["Simulated Annealing"].append((time.time() - start, length_s))
        assert validate_tsp_solution(
            G, path_s + [path_s[0]]
        ), f"Invalid SA solution in trial {trial}"

        start = time.time()
        path_q, length_q = tsp_symmetric(G)
        results["PyQrackIsing"].append((time.time() - start, length_q))
        assert validate_tsp_solution(
            G, path_q + [path_s[0]]
        ), f"Invalid PyQrackIsing solution in trial {trial}"

    return results


# Run benchmark for 32 and 64 nodes
results = [
    benchmark_tsp_realistic(32),
    benchmark_tsp_realistic(64),
    # benchmark_tsp_realistic(128),
    # benchmark_tsp_realistic(256),
]

for results_dict in results:
    for key, value in results_dict.items():
        transposed = list(zip(*value))
        time = sum(transposed[0]) / len(transposed[0])
        length = min(transposed[1])
        results_dict[key] = {"seconds": time, "length": length}

# Combine into dataframe
df32 = pd.DataFrame(
    {
        "Nearest Neighbor  (32)": results[0]["Nearest Neighbor"],
        "Christofides  (32)": results[0]["Christofides"],
        "Simulated Annealing  (32)": results[0]["Simulated Annealing"],
        "PyQrackIsing  (32)": results[0]["PyQrackIsing"],
    }
)
print(df32)
df64 = pd.DataFrame(
    {
        "Nearest Neighbor  (64)": results[1]["Nearest Neighbor"],
        "Christofides  (64)": results[1]["Christofides"],
        "Simulated Annealing  (64)": results[1]["Simulated Annealing"],
        "PyQrackIsing  (64)": results[1]["PyQrackIsing"],
    }
)
print(df64)
# df128 = pd.DataFrame(
#     {
#         "Nearest Neighbor (128)": results[2]["Nearest Neighbor"],
#         "Christofides (128)": results[2]["Christofides"],
#         "Simulated Annealing (128)": results[2]["Simulated Annealing"],
#         "PyQrackIsing (128)": results[2]["PyQrackIsing"],
#     }
# )
print(df128)
# df256 = pd.DataFrame(
#     {
#         "Nearest Neighbor (256)": results[3]["Nearest Neighbor"],
#         "Christofides (256)": results[3]["Christofides"],
#         "Simulated Annealing (256)": results[3]["Simulated Annealing"],
#         "PyQrackIsing (256)": results[3]["PyQrackIsing"],
#     }
# )
# print(df256)
