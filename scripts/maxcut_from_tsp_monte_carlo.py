# Traveling Salesman Problem (considered NP-complete)
# Produced by Dan Strano, Elara (the OpenAI custom GPT)

from pyqrackising import tsp_symmetric
from multiprocessing import shared_memory
import multiprocessing
import numpy as np
import os
import sys
import time


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

    path, length = tsp_symmetric(G=G, is_cyclic=False, monte_carlo=True, k_neighbors=k_neighbors)

    existing_shm.close()  # worker should not unlink, just close
    return path, length


def tsp_to_maxcut_bipartition(tsp_path, weights):
    n = len(tsp_path)
    best_cut_value = -float("inf")
    best_partition = None
    direction = 0

    for offset in [-1, 0, 1]:
        mid = n // 2 + offset
        A = set(tsp_path[:mid])
        B = set(tsp_path[mid:])
        cut_value = sum(weights[u, v] for u in A for v in B)
        if cut_value > best_cut_value:
            best_cut_value = cut_value
            best_partition = (A, B)
            direction = offset

    if direction == 0:
        return best_partition, best_cut_value

    improved = True
    best_offset = direction
    while improved:
        improved = False
        offset = best_offset + direction
        mid = n // 2 + offset
        A = set(tsp_path[:mid])
        B = set(tsp_path[mid:])
        cut_value = sum(weights[u, v] for u in A for v in B)
        if cut_value > best_cut_value:
            best_cut_value = cut_value
            best_partition = (A, B)
            best_offset = offset
            improved = True

    return best_partition, best_cut_value


if __name__ == "__main__":
    n_nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    multi_start = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    k_neighbors = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else None

    print(f"Random seed: {seed}")
    print(f"Node count: {n_nodes}")

    start = time.perf_counter()
    shm, G_m = generate_tsp_graph(n_nodes=n_nodes, seed=seed)
    seconds = time.perf_counter() - start
    print(f"{seconds} seconds to initialize the adjacency matrix (statement of the problem itself)")

    start = time.perf_counter()

    args = [(shm.name, G_m.shape, G_m.dtype, k_neighbors)] * multi_start

    with multiprocessing.Pool(processes=multi_start) as pool:
        results = pool.map(bootstrap_worker, args)

    results.sort(key=lambda r: r[1])
    best_circuit, best_path_length = results[0]

    seconds = time.perf_counter() - start

    print(f"Seconds to TSP solution: {seconds}")

    start = time.perf_counter()
    partition, cut_value = tsp_to_maxcut_bipartition(best_circuit, G_m)
    seconds = time.perf_counter() - start

    shm.close()
    shm.unlink()  # only unlink once, after workers are done

    print(f"Seconds to MAXCUT solution: {seconds}")
    print(f"Partition: {partition}")
    print(f"Cut value: {cut_value}")
    print("(The average randomized and normalized separation between each and every node is about 0.5.)")
