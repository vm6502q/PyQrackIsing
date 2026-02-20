# Random MAXCUT (for execution time tests)
# Produced by Dan Strano, Elara (the OpenAI custom GPT)

from pyqrackising import maxcut_tfim_streaming, spin_glass_solver_streaming
from numba import njit
import numpy as np
import sys
import time

if __name__ == "__main__":
    n_nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    quality = int(sys.argv[2]) if len(sys.argv) > 2 else None
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else None
    is_spin_glass = (sys.argv[4] not in ["0", "False"]) if len(sys.argv) > 4 else False

    # This is a contrived example.
    # The function must use numba NJIT.
    # (In practice, even if you use other Python functionality like itertools,
    # you can pre-calculate and load the data as a list through the arguments tuple.)

    @njit
    def G_func(i, j):
        if i > j:
            i, j = j, i
        return ((j + 1) % (i + 1)) / n_nodes

    print(f"Random seed: {seed}")
    print(f"Node count: {n_nodes}")
    start = time.perf_counter()
    if is_spin_glass:
        bitstring, cut_value, cut, energy = spin_glass_solver_streaming(G_func, list(range(n_nodes)), quality=quality, is_spin_glass=False)
    else:
        bitstring, cut_value, cut = maxcut_tfim_streaming(G_func, list(range(n_nodes)), quality=quality)
    seconds = time.perf_counter() - start

    print(f"Seconds to solution: {seconds}")
    print(f"Bipartite cut bit string: {bitstring}")
    print(f"Cut weight: {cut_value}")
