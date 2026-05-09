# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Daniel Strano and the Qrack contributors
#
# Initial draft produced by (Anthropic) Claude (Sonnet 4.6).
#
# maxcut_exact_sparse.py — exact MAXCUT solver via weighted MaxSAT (sparse)
#
# Same encoding as maxcut_exact.py; edge list built from CSR arrays.
# See maxcut_exact.py for full encoding documentation.

import time
import networkx as nx
import numpy as np
from .maxcut_tfim_util import (
    compute_cut_sparse,
    compute_energy_sparse,
    get_cut,
    int_to_bitstring,
    opencl_context,
    to_scipy_sparse_upper_triangular,
)

try:
    from pysat.examples.rc2 import RC2
    from pysat.formula import WCNF
    _HAVE_PYSAT = True
except ImportError:
    _HAVE_PYSAT = False

dtype = opencl_context.dtype
_SCALE = 10 ** 9


def _build_wcnf_sparse(G_data, G_rows, G_cols, n):
    wcnf = WCNF()
    total_pos_scaled = 0
    baseline_neg = 0.0
    edge_idx = 0
    for i in range(n):
        for r in range(G_rows[i], G_rows[i + 1]):
            j = int(G_cols[r])
            if j <= i:
                continue
            w = float(G_data[r])
            if w == 0.0:
                continue
            xi = i + 1
            xj = j + 1
            s  = n + edge_idx + 1
            wcnf.append([ xi,  xj, -s])
            wcnf.append([-xi, -xj, -s])
            if w >= 0.0:
                iw = max(1, int(round(w * _SCALE)))
                total_pos_scaled += iw
                wcnf.append([s], weight=iw)
            else:
                iw = max(1, int(round(-w * _SCALE)))
                baseline_neg += w
                wcnf.append([-s], weight=iw)
            edge_idx += 1
    return wcnf, total_pos_scaled, baseline_neg


def _model_to_bits(model, n):
    bits = np.zeros(n, dtype=np.bool_)
    for lit in model:
        var = abs(lit) - 1
        if 0 <= var < n:
            bits[var] = lit > 0
    return bits


def solve_maxcut_exact_sparse(
    G,
    best_guess=None,
    quality=None,
    shots=None,
    is_spin_glass=False,
    anneal_t=None,
    anneal_h=None,
    repulsion_base=None,
    is_maxcut_gpu=True,
    verbose=True,
    time_limit=None,
    gray_iterations=None,
    gray_seed_multiple=None,
):
    """
    Exact MAXCUT/spin-glass solver via weighted MaxSAT (RC2), sparse variant.

    Accepts the same G input as spin_glass_solver_sparse (NetworkX graph or
    scipy CSR matrix) and the same warm-start formats (str, int, list, None).

    Requires:  pip install python-sat

    Parameters
    ----------
    G : networkx.Graph or scipy CSR matrix
    best_guess : str | int | list[bool] | None
    quality, shots, anneal_t, anneal_h, repulsion_base, is_maxcut_gpu,
    is_spin_glass, gray_iterations, gray_seed_multiple
        Forwarded to spin_glass_solver_sparse when best_guess is None.
    verbose : bool
    time_limit : float or None

    Returns
    -------
    bitstring : str
    cut_value : float
    partition : tuple(list, list)
    min_energy : float
    certified : bool
    """
    if not _HAVE_PYSAT:
        raise ImportError(
            "python-sat is required for solve_maxcut_exact_sparse. "
            "Install it with:  pip install python-sat"
        )

    if isinstance(G, nx.Graph):
        nodes = list(G.nodes())
        n_qubits = len(nodes)
        G_m = to_scipy_sparse_upper_triangular(G, nodes, n_qubits)
    else:
        n_qubits = G.shape[0]
        nodes = list(range(n_qubits))
        G_m = G

    if n_qubits < 3:
        if n_qubits == 0:
            return "", 0, ([], []), 0, True
        if n_qubits == 1:
            return "0", 0, (nodes, []), 0, True
        if n_qubits == 2:
            weight = G_m[0, 1]
            if weight < 0.0:
                return "00", 0, (nodes, []), weight, True
            return "01", weight, ([nodes[0]], [nodes[1]]), -weight, True

    G_data = np.asarray(G_m.data, dtype=np.float64)
    G_rows = np.asarray(G_m.indptr, dtype=np.int64)
    G_cols = np.asarray(G_m.indices, dtype=np.int64)

    if isinstance(best_guess, str):
        bitstring = best_guess
    elif isinstance(best_guess, int):
        bitstring = int_to_bitstring(best_guess, n_qubits)
    elif isinstance(best_guess, list):
        bitstring = "".join(["1" if b else "0" for b in best_guess])
    else:
        if verbose:
            print("Running PyQrackIsing heuristic...")
        from .spin_glass_solver_sparse import spin_glass_solver_sparse
        kwargs = {}
        if quality is not None:            kwargs["quality"]            = quality
        if shots is not None:              kwargs["shots"]              = shots
        if anneal_t is not None:           kwargs["anneal_t"]           = anneal_t
        if anneal_h is not None:           kwargs["anneal_h"]           = anneal_h
        if repulsion_base is not None:     kwargs["repulsion_base"]     = repulsion_base
        if gray_iterations is not None:    kwargs["gray_iterations"]    = gray_iterations
        if gray_seed_multiple is not None: kwargs["gray_seed_multiple"] = gray_seed_multiple
        kwargs["is_maxcut_gpu"] = is_maxcut_gpu
        t0 = time.monotonic()
        bitstring, cut_value, _, _ = spin_glass_solver_sparse(
            G_m, is_spin_glass=is_spin_glass, **kwargs
        )
        if verbose:
            print(f"Heuristic value: {cut_value:.6f}  ({time.monotonic()-t0:.3f}s)")

    warm_theta = np.array([b == "1" for b in list(bitstring)], dtype=np.bool_)
    wcnf, total_pos_scaled, baseline_neg = _build_wcnf_sparse(
        G_data, G_rows, G_cols, n_qubits
    )

    if verbose:
        print("Starting RC2 MaxSAT solver...")

    rc2_kwargs = {}
    if time_limit is not None:
        rc2_kwargs["time_limit"] = time_limit

    t0 = time.monotonic()
    with RC2(wcnf, solver="g3", **rc2_kwargs) as rc2:
        try:
            for i, bit in enumerate(warm_theta):
                rc2.oracle.set_phases([i + 1 if bit else -(i + 1)])
        except AttributeError:
            pass
        model = rc2.compute()
        cost_scaled = rc2.cost
        certified = model is not None

    elapsed = time.monotonic() - t0

    if model is None:
        if verbose:
            print(f"RC2 time limit reached ({elapsed:.3f}s). "
                  f"Returning warm start.")
        best_theta = warm_theta
    else:
        best_theta = _model_to_bits(model, n_qubits)

    if verbose and model is not None:
        cut_opt = (total_pos_scaled - cost_scaled) / _SCALE + baseline_neg
        print(f"RC2 optimum: {cut_opt:.6f}  ({elapsed:.3f}s)")

    bitstring, l, r = get_cut(best_theta, nodes, n_qubits)
    if is_spin_glass:
        cut_value = float(compute_cut_sparse(best_theta, G_data, G_rows, G_cols, n_qubits))
        min_energy = -float(compute_energy_sparse(best_theta, G_data, G_rows, G_cols, n_qubits))
    else:
        cut_value = float(compute_cut_sparse(best_theta, G_data, G_rows, G_cols, n_qubits))
        min_energy = float(compute_energy_sparse(best_theta, G_data, G_rows, G_cols, n_qubits))

    return bitstring, cut_value, (l, r), min_energy, certified
