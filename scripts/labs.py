# Spin Glass Ground State (considered NP-complete)
# Produced by Dan Strano, Elara (the OpenAI custom GPT)

from pyqrackising import spin_glass_solver
import networkx as nx
import numpy as np

import sys
from itertools import combinations


# LABS spin glass
def generate_labs_qubo(N, lam=10.0):
    """
    Generate a LABS instance as a quadratic QUBO-like form
    by expanding quartic terms with auxiliary spins.
    
    Parameters
    ----------
    N : int
        Length of the binary sequence.
    lam : float
        Penalty coefficient for enforcing auxiliary constraints.
    
    Returns
    -------
    W : np.ndarray
        Full symmetric weight matrix (size ~ N^2).
    labels : list[str]
        Names of variables (original spins + auxiliaries).
    """
    # Label base spins
    spins = [f"s{i}" for i in range(N)]
    aux = []
    index_map = {s: idx for idx, s in enumerate(spins)}
    
    # Placeholder dictionary for coupling terms
    couplings = {}

    # Each k defines a set of autocorrelation interactions
    for k in range(1, N):
        for i, j in combinations(range(N - k), 2):
            # quartic term: s_i * s_{i+k} * s_j * s_{j+k}
            # introduce two auxiliaries a_i_k and a_j_k
            ai = f"a_{i}_{k}"
            aj = f"a_{j}_{k}"
            for a in (ai, aj):
                if a not in index_map:
                    index_map[a] = len(index_map)
                    aux.append(a)

            # term: a_i_k * a_j_k
            couplings[(ai, aj)] = couplings.get((ai, aj), 0.0) + 1.0

            # penalty to enforce a_i_k = s_i * s_{i+k}
            for (x, y) in [(ai, f"s{i}"), (ai, f"s{i+k}")]:
                couplings[(x, y)] = couplings.get((x, y), 0.0) - lam

            # same for a_j_k
            for (x, y) in [(aj, f"s{j}"), (aj, f"s{j+k}")]:
                couplings[(x, y)] = couplings.get((x, y), 0.0) - lam

            # diagonal penalties to ensure consistency
            for a in (ai, aj):
                couplings[(a, a)] = couplings.get((a, a), 0.0) + 2 * lam

    # Build full symmetric matrix
    M = len(index_map)
    W = np.zeros((M, M))
    for (x, y), w in couplings.items():
        i, j = index_map[x], index_map[y]
        W[i, j] += w
        if i != j:
            W[j, i] += w  # symmetric

    labels = list(index_map.keys())
    return W, labels


if __name__ == "__main__":
    n_nodes = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    quality = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    G_m, labels = generate_labs_qubo(n_nodes)
    best_bitstring, best_cut_value, best_cut, best_energy = spin_glass_solver(G_m, quality=quality)

    print(f"Ground State Energy: {best_energy}")
    print(f"Length {n_nodes} solution: {best_bitstring}")
    print(f"Labels: {labels}")
