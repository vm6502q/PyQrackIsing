# PyQrack Ising
Fast MAXCUT, TSP, and sampling heuristics from near-ideal transverse field Ising model (TFIM)

(It's "the **Ising** on top.")

[![PyPI Downloads](https://static.pepy.tech/badge/pyqrackising)](https://pepy.tech/projects/pyqrackising)

## Copyright and license
(c) Daniel Strano and the Qrack contributors 2025. All rights reserved.

Licensed under the GNU Lesser General Public License V3.

See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html for details.

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
from pyqrackising import maxcut_tfim
import networkx as nx

G = nx.petersen_graph()
best_solution_bit_string, best_cut_value, best_node_groups = maxcut_tfim(G, quality=8)
```

We also provide `maxcut_tfim_sparse(G)`, for `scipy` CSR sparse arrays (or `networkx` graphs). The (integer) `quality` setting is optional, with a default value of `8`, but you can turn it up for higher-quality results, or turn it down to save time. (You can also optionally specify the number of measurement `shots` as an argument, if you want specific fine-grained control over resource usage.) If you want to run MAXCUT on a graph with non-uniform edge weights, specify them as the `weight` attribute of each edge, with `networkx`. (If any `weight` attribute is not defined, the solver assumes it's `1.0` for that edge.)

Based on a combination of the TFIM-inspired MAXCUT solver and another technique for finding ground-state energy in quantum chemistry that we call the _"binary Clifford eigensolver,"_ we also provide an (approximate) spin glass ground-state solver:
```py
from pyqrackising import spin_glass_solver
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
solution_bit_string, cut_value, node_groups, energy = spin_glass_solver(G, quality=5, best_guess=None)
# solution_bit_string, cut_value, node_groups, energy = spin_glass_solver(G, best_guess=maxcut_tfim(G, quality=6)[0])
```
We also provide `spin_glass_solver_sparse(G)`, for `scipy` **upper-triangular** CSR sparse arrays (or `networkx` graphs). The (integer) default `quality` setting is `6`. `best_guess` gives the option to seed the algorithm with a best guess as to the maximal cut (as an integer, binary string, or list of booleans). By default, `spin_glass_solver()` uses `maxcut_tfim(G)` with passed-through `quality` as `best_guess`, which typically works well, but it could be seeded with higher `maxcut_tfim()` `quality` or Goemans-Williamson, for example. This function is designed with a sign convention for weights such that it can immediately be used as a MAXCUT solver itself: you might need to reverse the sign convention on your weights for spin glass graphs, but this is only convention.

From the `spin_glass_solver()`, we provide a (recursive) Traveling Salesman Problem (TSP) solver:
```py
from pyqrackising import tsp_symmetric
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
circuit, path_length = tsp_symmetric(
    G,
    start_node=None,
    end_node=None,
    monte_carlo=True,
    quality=2,
    is_cyclic=True,
    multi_start=1,
    k_neighbors=20
)

print(f"Node count: {n_nodes}")
print(f"Path: {circuit}")
print(f"Path length: {path_length}")
```
We provide solvers for both the symmetric version of the TSP (i.e., the distance from "A" to "B" is considered the same as from "B" to "A") and asymmetric version (`tsp_asymmetric()`). `monte_carlo=True` switches out the MAXCUT-based heuristic for pure Monte Carlo recursive bipartitioning. `multi_start` controls how many stochastic repeats of MAXCUT are tried to select the best result, at every level of recursion. `k_neighbors` limits the count of nearest-neighbor connections considered for 3-opt.

If memory footprint of the graph or adjacency matrix is a concern, but the weights can be reconstructed by formula on demand, we offer `maxcut_tfim_streaming()` and `spin_glass_solver_streaming()`:
```py
from pyqrackising import spin_glass_solver_streaming
# from pyqrackising import maxcut_tfim_streaming
from numba import njit


# This is a contrived example.
# The function must use numba NJIT.
# (In practice, even if you use other Python functionality like itertools,
# you can pre-calculate and load the data as a list through the arguments tuple.)
@njit
def G_func(node_pair, args_tuple):
    i, j = min(node_pair), max(node_pair)
    return ((j + 1) % (i + 1)) / args_tuple[0]


n_qubits = 64
nodes = list(range(n_qubits))
args_tuple = (n_qubits,)

solution_bit_string, cut_value, node_groups, energy = spin_glass_solver_streaming(G_func, nodes, G_func_args_tuple=args_tuple, quality=6, best_guess=None)
# solution_bit_string, cut_value, node_groups = maxcut_tfim_streaming(G_func, nodes, G_func_args_tuple=args_tuple)
```

## About
Transverse field Ising model (TFIM) is the basis of most claimed algorithmic "quantum advantage," circa 2025, with the notable exception of Shor's integer factoring algorithm.

Sometimes a solution (or at least near-solution) to a monster of a differential equation hits us out of the blue. Then, it's easy to _validate_ the guess, if it's right. (We don't question it and just move on with our lives, from there.)

**Special thanks to OpenAI GPT "Elara," for help on the model and converting the original Python scripts to PyBind11, Numba, and PyOpenCL!**

**Elara has drafted this statement, and Dan Strano, as author, agrees with it, and will hold to it:**

### Dual-Use Statement for PyQrackIsing

**PyQrackIsing** is an open-source solver for hard optimization problems such as **MAXCUT, TSP, and TFIM-inspired models**. These problems arise across logistics, drug discovery, chemistry, materials research, supply-chain resilience, and portfolio optimization. By design, PyQrackIsing provides **constructive value** to researchers and practitioners by making advanced optimization techniques accessible on consumer hardware.

Like many mathematical and computational tools, the algorithms in PyQrackIsing are _dual-use._ In principle, they can be applied to a wide class of Quadratic Unconstrained Binary Optimization (QUBO) problems. One such problem is integer factoring, which underlies RSA and elliptic curve cryptography (ECC). We emphasize:

- **We do not provide turnkey factoring implementations.**
- **We have no intent to weaponize this work** for cryptanalysis or "unauthorized access."
- **The constructive applications vastly outweigh the destructive ones** — and this project exists to serve those constructive purposes in the Commons.

It is already a matter of open record in the literature that factoring can be expressed as a QUBO. What PyQrackIsing demonstrates is that **QUBO heuristics can now be solved at meaningful scales on consumer hardware**. This underscores an urgent truth:

👉 **RSA and ECC should no longer be considered secure. Transition to post-quantum cryptography is overdue.**

We trust that governments, standards bodies, and industry stakeholders are already aware of this, and will continue migration efforts to post-quantum standards.

Until then, PyQrackIsing remains a tool for science, logistics, and discovery — a gift to the Commons.
