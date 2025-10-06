# Random MAXCUT (for execution time tests)
# Produced by Dan Strano, Elara (the OpenAI custom GPT)

from pyqrackising import maxcut_tfim, spin_glass_solver
from numba import njit, prange
import numpy as np
import sys
import time


# Random MAXCUT adjacency matrix
@njit
def generate_adjacency(n_nodes=64, seed=None):
    if not (seed is None):
        np.random.seed(seed)

    G_m = np.empty((n_nodes, n_nodes), dtype=np.float32)

    for u in prange(n_nodes):
        for v in range(u + 1, n_nodes):
            weight = np.random.random()
            G_m[u, v] = weight
            G_m[v, u] = weight

    return G_m


if __name__ == "__main__":
    n_nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    quality = int(sys.argv[2]) if len(sys.argv) > 2 else None
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else None
    is_spin_glass = (sys.argv[3] not in ['0', 'False']) if len(sys.argv) > 3 else False

    start = time.perf_counter()
    G_m = generate_adjacency(n_nodes=n_nodes, seed=seed)
    seconds = time.perf_counter() - start
    print(f"{seconds} seconds to initialize the adjacency matrix (statement of the problem itself)")

    print(f"Random seed: {seed}")
    print(f"Node count: {n_nodes}")
    start = time.perf_counter()
    if is_spin_glass:
        bitstring, cut_value, cut, energy = spin_glass_solver(G_m, quality=quality)
    else:
        bitstring, cut_value, cut = maxcut_tfim(G_m, quality=quality)
    seconds = time.perf_counter() - start

    print(f"Seconds to solution: {seconds}")
    print(f"Bipartite cut bit string: {bitstring}")
    print(f"Cut weight: {cut_value}")
    print(
        "(The average randomized and symmetric weight between each and every node is about 0.5, from the range 0.0 to 1.0.)"
    )
