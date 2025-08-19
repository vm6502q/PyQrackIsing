# PyQrack Ising
Efficiently generate near-ideal samples from transverse field Ising model (TFIM), and TFIM-inspired MAXCUT solutions

(It's "the **Ising** on top.")

[![PyPI Downloads](https://static.pepy.tech/badge/pyqrackising)](https://pepy.tech/projects/pyqrackising)

## Copyright and license
(c) Daniel Strano and the Qrack contributors 2025. All rights reserved.

## Installation
From PyPi:
```
pip3 install PyQrackIsing
```

From Source: install `pybind11`, then
```
pip3 install .
```
in the root source directory (with `setup.py`).

Windows users might find Windows Subsystem Linux (WSL) to be the easier and preferred choice for installation.

## Usage

```py
from PyQrackIsing import generate_tfim_samples

samples = generate_tfim_samples(
    J=-1.0,
    h=2.0,
    z=4,
    theta=0.174532925199432957,
    t=5,
    n_qubits=56,
    shots=100
)
```

There are two other functions, `tfim_magnetization()` and `tfim_square_magnetization()`, that follow the same function signature except without the `shots` argument.

The library also provides a TFIM-inspired (approximate) MAXCUT solver:
```py
from PyQrackIsing import maxcut_tfim
import networkx as nx

G = nx.petersen_graph()
best_cut_value, best_solution_bit_string, best_cut_edges = maxcut_tfim(G, quality=12)
```

The (integer) `quality` setting is optional, with a default value of `12`, but you can turn it up for higher-quality results, or turn it down to save time. (You can also optionally specify the number of measurement `shots` as an argument, if you want specific fine-grained control over resource usage.) If you want to run MAXCUT on a graph with non-uniform edge weights, specify them as the `weight` attribute of each edge, with `networkx`. (If any `weight` attribute is not defined, the solver assumes it's `1.0` for that edge.)

Based on a combination of the TFIM-inspired MAXCUT solver and another technique for finding ground-state energy in quantum chemistry that we call the _"binary Clifford eigensolver,"_ we also provide an (approximate) spin glass ground-state solver:
```py
from PyQrackIsing import spin_glass_solver
import networkx as nx
import numpy as np


# NP-complete spin glass
def generate_spin_glass_graph(n_nodes=16, degree=3, seed=None):
    if not (seed is None):
        np.random.seed(seed)
    G = nx.random_regular_graph(d=degree, n=n_nodes, seed=seed)
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.choice([-1, 1])  # spin glass couplings
    return G


G = generate_spin_glass_graph(n_nodes=64, seed=42)
cut_value, bitstring, cut_edges, energy = spin_glass_solver(G, quality=2, best_guess=None)
# cut_value, bitstring, cut_edges, energy = spin_glass_solver(G, best_guess=maxcut_tfim(G, quality=12)[1])
```
The `quality` setting default is `2`, and overhead grows roughly like the qubit count to the power of `quality` (so `O(n ** 2)` for default). `best_guess` gives the option to seed the algorithm with a best guess as to the maximal cut (as an integer, binary string, or list of booleans). By default, `spin_glass_solver()` uses `maxcut_tfim(G, quality=0)` as `best_guess`, which typically works well, but it could be seeded with higher `maxcut_tfim()` `quality` or Goemans-Williamson, for example. This function is designed with a sign convention for weights such that it can immediately be used as a MAXCUT solver itself: you might need to reverse the sign convention on your weights for spin glass graphs, but this is only convention.

## About
Transverse field Ising model (TFIM) is the basis of most claimed algorithmic "quantum advantage," circa 2025, with the notable exception of Shor's integer factoring algorithm.

Sometimes a solution (or at least near-solution) to a monster of a differential equation hits us out of the blue. Then, it's easy to _validate_ the guess, if it's right. (We don't question it and just move on with our lives, from there.)

**Special thanks to OpenAI GPT "Elara," for help on the model and converting the original Python scripts to PyBind11!**
