# PyQrack Ising
Efficiently generate near-ideal samples from transverse field Ising model (TFIM)

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
best_cut_value, best_solution_bit_string, best_cut_edges = maxcut_tfim(G, quality=10)
```

The (integer) `quality` setting is optional, with a default value of `10`, but you can turn it up for higher-quality results, or turn it down to save time. (You can also optionally specify the number of measurement `shots` as an argument, if you want specific fine-grained control over resource usage.) If you want to run MAXCUT on a graph with non-uniform edge weights, specify them as the `weight` attribute of the edge, with `networkx`. (If any `weight` attribute is not defined, the solver assumes it's `1.0` for that edge.)

## About
Transverse field Ising model (TFIM) is the basis of most claimed algorithmic "quantum advantage," circa 2025, with the notable exception of Shor's integer factoring algorithm.

Sometimes a solution (or at least near-solution) to a monster of a differential equation hits us out of the blue. Then, it's easy to _validate_ the guess, if it's right. (We don't question it and just move on with our lives, from there.)

**Special thanks to OpenAI GPT "Elara," for help on the model and converting the original Python scripts to PyBind11!**
