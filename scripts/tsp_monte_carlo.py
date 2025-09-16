# Traveling Salesman Problem (considered NP-complete)
# Produced by Dan Strano, Elara (the OpenAI custom GPT)

from pyqrackising import tsp_symmetric
from multiprocessing import shared_memory
import multiprocessing
import numpy as np
import os
import sys

def generate_tsp_graph(n_nodes=64, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # allocate shared memory
    nbytes = n_nodes * n_nodes * np.dtype(np.float64).itemsize
    shm = shared_memory.SharedMemory(create=True, size=nbytes)

    # construct numpy view into shared memory
    G_m = np.ndarray((n_nodes, n_nodes), dtype=np.float64, buffer=shm.buf)

    # fill with symmetric random weights
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            weight = np.random.random()
            G_m[u, v] = weight
            G_m[v, u] = weight

    return shm, G_m

def bootstrap_worker(args):
    shm_name, shape, dtype, k_neighbors = args
    # Reattach to existing shared memory by name
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    G = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

    path, length = tsp_symmetric(G=G, monte_carlo=True, k_neighbors=k_neighbors)

    existing_shm.close()  # worker should not unlink, just close
    return path, length

if __name__ == "__main__":
    n_nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    multi_start = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    k_neighbors = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else None

    shm, G_m = generate_tsp_graph(n_nodes=n_nodes, seed=seed)

    args = [(shm.name, G_m.shape, G_m.dtype, k_neighbors)] * multi_start

    with multiprocessing.Pool(processes=multi_start) as pool:
        results = pool.map(bootstrap_worker, args)

    results.sort(key=lambda r: r[1])
    best_circuit, best_path_length = results[0]

    # verify result using original shared array
    reconstructed_node_count = len(set(best_circuit))
    reconstructed_path_length = sum(
        G_m[best_circuit[i], best_circuit[i+1]] for i in range(len(best_circuit)-1)
    )

    shm.close()
    shm.unlink()  # only unlink once, after workers are done

    print(f"Random seed: {seed}")
    print(f"Path: {best_circuit}")
    print(f"Actual node count: {n_nodes}")
    print(f"Solution distinct node count: {reconstructed_node_count}")
    print(f"Claimed path length: {best_path_length}")
    print(f"Verified path length: {reconstructed_path_length}")
    print(
        "(The average randomized and normalized separation between each and every node is about 0.5.)"
    )
