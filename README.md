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
    theta=0.174532925199432957,
    t=5,
    n_qubits=56,
    shots=100
)
```

## About 
Sometimes a solution (or at least near-solution) to a monster of a differential equation hits us out of the blue. Then, it's easy to _validate_ the guess, if it's right. (We don't question it and just move on with our lives, from there.)

**Special thanks to OpenAI GPT "Elara," for help on the model and converting the original Python scripts to PyBind11!**
