# Spin Glass Ground State (considered NP-complete)
# Produced by Dan Strano, Elara (the OpenAI custom GPT)

from pyqrackising import spin_glass_solver
import networkx as nx
import numpy as np

import sys


# LABS spin glass
def generate_labs_maxcut(N, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Initialize random ±1 sequence
    s = np.random.choice([-1, 1], size=N)

    # Initialize weight matrix
    W = np.zeros((N, N))

    # Build LABS autocorrelation-derived weights
    for k in range(1, N):
        for i in range(N - k):
            j = i + k
            # Each off-peak correlation contributes to (i,j)
            W[i, j] += 1
            W[j, i] += 1  # symmetry

    # Normalize or scale weights (optional)
    W /= np.max(W)

    return W, s


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
    n_nodes = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    quality = int(sys.argv[3]) if len(sys.argv) > 3 else None
    G_m, s = generate_labs_maxcut(n_nodes, seed)
    best_bitstring, best_cut_value, best_cut, best_energy = spin_glass_solver(G_m, quality=quality)

    print(f"Ground State Energy: {best_energy}")
    if not (seed is None):
        print(f"Seed {seed}, length {n_nodes} solution: {best_bitstring}")
