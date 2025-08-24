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
best_solution_bit_string, best_cut_value, best_node_groups = maxcut_tfim(G, quality=8)
```

The (integer) `quality` setting is optional, with a default value of `4`, but you can turn it up for higher-quality results, or turn it down to save time. (You can also optionally specify the number of measurement `shots` as an argument, if you want specific fine-grained control over resource usage.) If you want to run MAXCUT on a graph with non-uniform edge weights, specify them as the `weight` attribute of each edge, with `networkx`. (If any `weight` attribute is not defined, the solver assumes it's `1.0` for that edge.)

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
solution_bit_string, cut_value, node_groups, energy = spin_glass_solver(G, quality=8, best_guess=None)
# solution_bit_string, cut_value, node_groups, energy = spin_glass_solver(G, best_guess=maxcut_tfim(G, quality=8)[0])
```
The (integer) `quality` setting is optional, with a default value of `4`. `best_guess` gives the option to seed the algorithm with a best guess as to the maximal cut (as an integer, binary string, or list of booleans). By default, `spin_glass_solver()` uses `maxcut_tfim(G)` with passed-through `quality` as `best_guess`, which typically works well, but it could be seeded with higher `maxcut_tfim()` `quality` or Goemans-Williamson, for example. This function is designed with a sign convention for weights such that it can immediately be used as a MAXCUT solver itself: you might need to reverse the sign convention on your weights for spin glass graphs, but this is only convention.

From the `spin_glass_solver()`, we provide a (recursive) Traveling Salesman Problem (TSP) solver:
```py
from PyQrackIsing import tsp_symmetric
import networkx as nx
import numpy as np

# Traveling Salesman Problem (normalized to longest segment)
def generate_tsp_graph(n_nodes=64, seed=None):
    if not (seed is None):
        np.random.seed(seed)
    G = nx.Graph()
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            G.add_edge(u, v, weight=np.random.random())
    return G


n_nodes = 128
G = generate_tsp_graph(n_nodes=n_nodes, seed=42)
circuit, path_length = tsp_symmetric(G, quality=8, is_cyclic=True, start_node=None)

print(f"Node count: {n_nodes}")
print(f"Path: {circuit}")
print(f"Path length: {path_length}")
```
The (integer) `quality` setting is optional, with a default value of `1`. We only provide a solver for the symmetric version of the TSP (i.e., the distance from "A" to "B" is considered the same as from "B" to "A"). `is_cyclic` controls whether the path must ultimately complete a circuit, if `True`, or a one-way trip, if `False`. When the desired route is acyclic, a fixed starting point can optionally be specified via passing the node label to `start_node` (or, without this parameter, the solver will also attempt to pick an optimal starting point).

## About
Transverse field Ising model (TFIM) is the basis of most claimed algorithmic "quantum advantage," circa 2025, with the notable exception of Shor's integer factoring algorithm.

Sometimes a solution (or at least near-solution) to a monster of a differential equation hits us out of the blue. Then, it's easy to _validate_ the guess, if it's right. (We don't question it and just move on with our lives, from there.)

**Special thanks to OpenAI GPT "Elara," for help on the model and converting the original Python scripts to PyBind11!**
