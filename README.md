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

From Source:
```
pip3 install .
```
in the root source directory (with `setup.py`).

Windows users might find Windows Subsystem Linux (WSL) to be the easier and preferred choice for installation.

## Usage

```py
from pyqrackising import generate_tfim_samples

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

The library also provides a TFIM-inspired (approximate) MAXCUT solver (which accepts a `networkx` graph or a 32-bit adjacency matrix):
```py
from pyqrackising import maxcut_tfim
import networkx as nx

G = nx.petersen_graph()
best_solution_bit_string, best_cut_value, best_node_groups = maxcut_tfim(G, quality=6, shots=None, is_spin_glass=False, anneal_t=8.0, anneal_h=8.0, repulsion_base=5.0)
```

We also provide `maxcut_tfim_sparse(G)`, for `scipy` CSR sparse arrays (or `networkx` graphs), and `maxcut_tfim_streaming(G_func, nodes)` for `numba` JIT streaming weights function definitions. The (integer) `quality` setting is optional, with a default value of `6`, but you can turn it up for higher-quality results, or turn it down to save time. (You can also optionally specify the number of measurement `shots` as an argument, if you want specific fine-grained control over resource usage.) `anneal_t` and `anneal_h` control the physical maximum annealing time and `h` transverse field parameter, as in Trotterized Ising model. While `anneal_t` and `anneal_h` control how annealing finds the _Hamming weight_ of cuts (i.e., how many nodes end up in either partition), `repulsion_base` similarly controls the steepness of the search basin for optimizing bit-string locality ("like-like" repulsion and "like-unlike" attraction) within each partition. If you want to run MAXCUT on a graph with non-uniform edge weights, specify them as the `weight` attribute of each edge, with `networkx`. (If any `weight` attribute is not defined, the solver assumes it's `1.0` for that edge.)

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
solution_bit_string, cut_value, node_groups, energy = spin_glass_solver(G, quality=6, shots=None, anneal_t=8.0, anneal_h=8.0, is_spin_glass=True, best_guess=None, is_maxcut_gpu=True, gray_iterations=None, gray_seed_multiple=None)
# solution_bit_string, cut_value, node_groups, energy = spin_glass_solver(G, best_guess=maxcut_tfim(G, quality=6)[0])
```
We also provide `spin_glass_solver_sparse(G)` and `spin_glass_solver_streaming(G_func, nodes)`. `best_guess` gives the option to seed the algorithm with a best guess as to the maximal cut (as an integer, binary string, or list of booleans). By default, `spin_glass_solver()` uses `maxcut_tfim(G)` with passed-through `quality` as `best_guess`, which typically works well, but it could be seeded with higher `maxcut_tfim()` `quality` or Goemans-Williamson, for example. `is_spin_glass` controls whether the solver optimizes for cut value or spin-glass energy. This function is designed with a sign convention for weights such that it can immediately be used as a MAXCUT solver itself: you might need to reverse the sign convention on your weights for spin glass graphs, but this is only convention. `gray_iterations` gives manual control over how many iterations are carried out of a parallel Gray-code search on `best_guess`. `gray_seed_multiple` controls how many parallel search seeds (as a multiple of your CPU thread count) are tested for the best parallel seeds, and a value of `1` will perfectly cover the search space without collision if your node count is a power of 2.

From the MAXCUT solvers, we provide a (recursive) Traveling Salesman Problem (TSP) solver:
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

solution_bit_string, cut_value, node_groups, energy = spin_glass_solver_streaming(G_func, nodes, G_func_args_tuple=args_tuple, quality=4, best_guess=None)
# solution_bit_string, cut_value, node_groups = maxcut_tfim_streaming(G_func, nodes, G_func_args_tuple=args_tuple)
```

Finally, combining insights from both the (Monte Carlo) TSP and MAXCUT solvers, we have `tsp_maxcut(G)`, `tsp_maxcut_sparse(G)`, and `tsp_maxcut_streaming(G_func, nodes)`:

```
from pyqrackising import tsp_maxcut_sparse
import networkx as nx

G = nx.petersen_graph()
best_partition, best_cut_value = tsp_maxcut_sparse(G, k_neighbors=20, is_optimized=False)
```

When `is_optimized=True`, the `spin_glass_solver(G)` is used as a final optimization pass. When `is_optimized=False`, this solver becomes entirely serial and can be parallelized over CPU processing elements by user code, easily.

## Experimental OTOC sampling

```py
from pyqrackising import generate_tfim_samples

samples = generate_otoc_samples(
    J=-1.0,
    h=2.0,
    z=4,
    theta=0.174532925199432957,
    t=5,
    n_qubits=56,
    cycles=1,
    pauli_string = 'X' + 'I' * 55
    shots=100,
    measurement_basis='Z' * 56
)
```

**This function is experimental. It needs systematic validation.** However, we expose it in the public API while we do of the work of testing its validity at qubit counts that are tractable for exact simulation.

## Environment Variables

We expose an environment variable, "`PYQRACKISING_MAX_GPU_PROC_ELEM`", for OpenCL-based solvers. The default value (when the variable is not set) is queried from the OpenCL device properties. You might see performance benefit from tuning this manually to several times your device's number of "compute units" (or tune it down to reduce private memory usage).

By default, PyQrackIsing expects all `numpy` floating-point array inputs to be 32-bit. If you'd like to use 64-bit, you can set environment variable `PYQRACKISING_FPPOW=6` (meaning, 2^6=64, for the "floating-point (precision) power"). The default is `5`, for 32-bit. 16-bit is stubbed out and compiles for OpenCL, but the bigger hurdle is that `numpy` on `x86_64` doesn't provide a 16-bit floating point implementation. (As author of Qrack, I could suggest to the `numpy` maintainers that open-source, IEEE-compliant software-based implementations exist for `x86_64` and other architectures, but I'm sure they're aware and likely waiting for in-compiler support.) If you're on an ARM-based architecture, there's a good chance 16-bit floating-point will work, if `numpy` uses the native hardware support.

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
