import networkx as nx
import numpy as np
import os
import random
import time
import math
import multiprocessing
import warnings
import pandas as pd

from multiprocessing import shared_memory

from pyqrackising import tsp_symmetric


def generate_noneuclidean_tsp(n_nodes, seed=42):
    if not (seed is None):
        np.random.seed(seed)
    G = nx.Graph()
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            G.add_edge(u, v, weight=np.random.random())
    return G, None


# Generate a Euclidean TSP instance with n nodes in the unit square
def generate_euclidean_tsp(n, seed=42):
    if not (seed is None):
        random.seed(seed)
    points = {i: (random.random(), random.random()) for i in range(n)}
    G = nx.complete_graph(n)
    for u, v in G.edges():
        x1, y1 = points[u]
        x2, y2 = points[v]
        G[u][v]["weight"] = math.hypot(x1 - x2, y1 - y2)
    return G, points


def generate_clustered_tsp(n, clusters=4, spread=0.05, seed=42):
    if not (seed is None):
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


def tsp_qrack(args):
    shm_name, shape, dtype = args
    # Reattach to existing shared memory by name
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    G = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    path, length = tsp_symmetric(G, monte_carlo=True)
    existing_shm.close()

    return path, length
    


# Validation: check if path is a Hamiltonian cycle
def validate_tsp_solution(G, path):
    return len(path) == len(G.nodes) + 1 and set(path[:-1]) == set(G.nodes)


def get_path_length(G, path):
    return sum(G[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1))


# Benchmark framework with realistic (Euclidean) TSP graphs
def benchmark_tsp_realistic(n_nodes=64):
    results = {
        "Nearest Neighbor": [],
        "Christofides": [],
        "Simulated Annealing": [],
        "PyQrackIsing": [],
    }
    G, _ = generate_euclidean_tsp(n_nodes)
    multi_start = os.cpu_count()

    # Exclude numba JIT overhead with warmup:
    tsp_symmetric(nx.Graph())

    # Nearest neighbor
    start = time.time()
    path, length = tsp_nearest_neighbor(G)
    results["Nearest Neighbor"].append((time.time() - start, length))
    if not validate_tsp_solution(G, path + [path[0]]):
        warnings.warn("Invalid nearest neighbor solution!")

    # Christofides
    start = time.time()
    path_c = tsp_christofides(G)
    results["Christofides"].append((time.time() - start, get_path_length(G, path_c)))
    if not validate_tsp_solution(G, path_c):
        warnings.warn("Invalid Christofides solution!")

    # Simulated annealing
    args = [G] * multi_start
    start = time.time()
    with multiprocessing.Pool(processes=multi_start) as pool:
        mp_results = pool.map(tsp_simulated_annealing, args)
    mp_results.sort(key=lambda r: r[1])
    path_s, length_s = mp_results[0]
    results["Simulated Annealing"].append((time.time() - start, length_s))
    if not validate_tsp_solution(G, path_s + [path_s[0]]):
        warnings.warn("Invalid simulated annealing solution!")

    # allocate shared memory
    _G_m = nx.to_numpy_array(G, weight='weight', nonedge=0.0)
    shm = shared_memory.SharedMemory(create=True, size=_G_m.nbytes)
    G_m = np.ndarray(_G_m.shape, dtype=_G_m.dtype, buffer=shm.buf)
    G_m[:] = _G_m[:]  # copy initial data
    args = [(shm.name, G_m.shape, G_m.dtype)] * multi_start

    start = time.time()
    with multiprocessing.Pool(processes=multi_start) as pool:
        mp_results = pool.map(tsp_qrack, args)
    mp_results.sort(key=lambda r: r[1])
    path_q, length_q = mp_results[0]
    results["PyQrackIsing"].append((time.time() - start, length_q))
    if not validate_tsp_solution(G, path_q):
        warnings.warn("Invalid PyQrackIsing solution!")

    G_m = None
    shm.close()
    shm.unlink()

    return results


# Run benchmark for 32 through 4096 nodes
for i in range(5, 12):
    n_nodes = 1 << i
    results_dict = benchmark_tsp_realistic(n_nodes)
    for key, value in results_dict.items():
        transposed = list(zip(*value))
        seconds = sum(transposed[0]) / len(transposed[0])
        length = min(transposed[1])
        results_dict[key] = {"seconds": seconds, "length": length}
    # Combine into dataframe
    df = pd.DataFrame(
        {
            f"Nearest Neighbor ({n_nodes})": results_dict["Nearest Neighbor"],
            F"Christofides ({n_nodes})": results_dict["Christofides"],
            F"Simulated Annealing ({n_nodes})": results_dict["Simulated Annealing"],
            F"PyQrackIsing ({n_nodes})": results_dict["PyQrackIsing"],
        }
    )
    print(df)
