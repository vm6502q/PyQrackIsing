# Random MAXCUT (for execution time tests)
# Produced by Dan Strano, Elara (the OpenAI custom GPT)

from pyqrackising import maxcut_tfim_sparse, spin_glass_solver_sparse
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
import sys
import time


# Random MAXCUT adjacency matrix
def generate_adjacency(n_nodes=64, seed=None):
    if not (seed is None):
        np.random.seed(seed)

    lil = lil_matrix((n_nodes, n_nodes), dtype=np.float32)

    for u in range(n_nodes >> 1):
        for v in range(u + 1, n_nodes >> 1):
            lil[u, v] = np.random.random()

    for u in range(n_nodes >> 1, n_nodes):
        for v in range(u + 1, n_nodes):
            lil[u, v] = np.random.random()

    return lil.tocsr()


if __name__ == "__main__":
    n_nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    quality = int(sys.argv[2]) if len(sys.argv) > 2 else None
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else None
    is_spin_glass = (sys.argv[4] not in ['0', 'False']) if len(sys.argv) > 4 else False

    start = time.perf_counter()
    G_m = generate_adjacency(n_nodes=n_nodes, seed=seed)
    seconds = time.perf_counter() - start
    print(f"{seconds} seconds to initialize the adjacency matrix (statement of the problem itself)")

    print(f"Random seed: {seed}")
    print(f"Node count: {n_nodes}")
    start = time.perf_counter()
    if is_spin_glass:
        bitstring, cut_value, cut, energy = spin_glass_solver_sparse(G_m, quality=quality, is_spin_glass=False)
    else:
        bitstring, cut_value, cut = maxcut_tfim_sparse(G_m, quality=quality)
    seconds = time.perf_counter() - start

    print(f"Seconds to solution: {seconds}")
    print(f"Bipartite cut bit string: {bitstring}")
    print(f"Cut weight: {cut_value}")
    print(
        "(The sparsity is 50%, and the randomized and symmetric average nonzero weight is about 0.5, from the range 0.0 to 1.0.)"
    )
