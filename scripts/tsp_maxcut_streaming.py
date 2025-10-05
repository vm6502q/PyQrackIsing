# MAXCUT from TSP (Streaming)
# Produced by Dan Strano, Elara (the OpenAI custom GPT)

from pyqrackising import tsp_maxcut_streaming
from numba import njit
import os
import sys
import time


if __name__ == "__main__":
    n_nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    multi_start = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    k_neighbors = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else None

    print(f"Random seed: {seed}")
    print(f"Node count: {n_nodes}")

    @njit
    def G_func(i, j):
        if i > j:
            i, j = j, i
        return ((j + 1) % (i + 1)) / n_nodes

    start = time.perf_counter()
    bit_string, cut_value, partition, energy = tsp_maxcut_streaming(G_func, list(range(n_nodes)))
    seconds = time.perf_counter() - start

    print(f"Seconds to MAXCUT solution: {seconds}")
    print(f"Partition: {bit_string}")
    print(f"Cut value: {cut_value}")
