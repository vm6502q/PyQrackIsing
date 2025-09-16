# Traveling Salesman Problem (considered NP-complete)
# Produced by Dan Strano, Elara (the OpenAI custom GPT)

from pyqrackising import tsp_symmetric
from multiprocessing import shared_memory
import multiprocessing
import networkx as nx
import numpy as np
import os
import sys


# Traveling Salesman Problem (normalized to longest segment)
def generate_tsp_graph(n_nodes=64, seed=None):
    if not (seed is None):
        np.random.seed(seed)
    G = nx.Graph()
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            G.add_edge(u, v, weight=np.random.random())
    return G


def bootstrap_worker(args):
    path, length = tsp_symmetric(G=args[0], monte_carlo=True, k_neighbors=args[1])
    args[2].close()

    return path, length


if __name__ == "__main__":
    # NP-complete TSP
    n_nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    multi_start = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    k_neighbors = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else None

    G = generate_tsp_graph(n_nodes=n_nodes, seed=seed)
    _G_m = nx.to_numpy_array(G, weight='weight', nonedge=0.0)
    shm = shared_memory.SharedMemory(create=True, size=_G_m.nbytes)
    G_m = np.ndarray(_G_m.shape, dtype=_G_m.dtype, buffer=shm.buf)
    G_m[:] = _G_m[:]
    _G_m = None
    G = None

    args = []
    for i in range(multi_start):
        args.append((G_m, k_neighbors, shm))
    with multiprocessing.Pool(processes=multi_start) as pool:
        results = pool.map(bootstrap_worker, args)

    results.sort(key=lambda r: r[1])
    best_circuit, best_path_length = results[0]

    reconstructed_node_count = len(set(best_circuit))
    reconstructed_path_length = 0
    best_nodes = None
    for i in range(len(best_circuit) - 1):
        reconstructed_path_length += G_m[best_circuit[i]][best_circuit[i + 1]]

    G_m = None
    shm.unlink()

    print(f"Random seed: {seed}")
    print(f"Path: {best_circuit}")
    print(f"Actual node count: {n_nodes}")
    print(f"Solution distinct node count: {reconstructed_node_count}")
    print(f"Claimed path length: {best_path_length}")
    print(f"Verified path length: {reconstructed_path_length}")
    print(
        "(The average randomized and normalized separation between each and every node is about 0.5.)"
    )
