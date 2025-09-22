# Random MAXCUT (for execution time tests)
# Produced by Dan Strano, Elara (the OpenAI custom GPT)

from pyqrackising import spin_glass_solver_streaming
from numba import njit
import numpy as np
import sys
import time


# This is a contrived example.
# The function must use numba NJIT.
# (In practice, even if you use other Python functionality like itertools,
# you can pre-calculate and load the data as a list through the arguments tuple.)
@njit
def G_func(node_pair, args_tuple):
    i, j = min(node_pair), max(node_pair)
    return ((j + 1) % (i + 1)) / args_tuple[0]


if __name__ == "__main__":
    n_nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    quality = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else None

    print(f"Random seed: {seed}")
    print(f"Node count: {n_nodes}")
    start = time.perf_counter()
    bitstring, cut_value, cut, energy = spin_glass_solver_streaming(G_func, list(range(n_nodes)), G_func_args_tuple=(n_nodes,), quality=quality)
    seconds = time.perf_counter() - start

    print(f"Seconds to solution: {seconds}")
    print(f"Bipartite cut bit string: {bitstring}")
    print(f"Cut weight: {cut_value}")
    print(
        "(The average randomized and symmetric weight between each and every node is about 0.5, from the range 0.0 to 1.0.)"
    )
