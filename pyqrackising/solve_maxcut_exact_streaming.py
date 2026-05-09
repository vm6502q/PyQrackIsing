# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Daniel Strano and the Qrack contributors
#
# Initial draft produced by (Anthropic) Claude (Sonnet 4.6).
#
# maxcut_exact_streaming.py — exact MAXCUT solver via weighted MaxSAT (streaming)
#
# Same encoding as maxcut_exact.py; edge weights fetched via G_func(nodes[i], nodes[j]).
# See maxcut_exact.py for full encoding documentation.

import time
import numpy as np
from .maxcut_tfim_util import (
    compute_cut_streaming,
    compute_energy_streaming,
    get_cut,
    int_to_bitstring,
    opencl_context,
)

try:
    from pysat.examples.rc2 import RC2
    from pysat.formula import WCNF
    _HAVE_PYSAT = True
except ImportError:
    _HAVE_PYSAT = False

dtype = opencl_context.dtype

# Integer scaling factor: edge weights are multiplied by this and rounded
# to integers for the WCNF encoding.  2**32 gives sub-ppb precision for
# weights in the typical PyQrackIsing range of [-1, 1] and is half 64-bit
# integer precision.
_SCALE = 2 ** 32


def _build_wcnf_streaming(G_func, nodes, n, has_negative):
    wcnf = WCNF()
    total_pos_scaled = 0
    neg_scaled_total = 0
    baseline_neg = 0.0
    edge_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            w = float(G_func(nodes[i], nodes[j]))
            if w == 0.0:
                continue
            xi = i + 1
            xj = j + 1
            s  = n + edge_idx + 1
            wcnf.append([ xi,  xj, -s])   # xi=0,xj=0 -> s=0
            wcnf.append([-xi, -xj, -s])   # xi=1,xj=1 -> s=0
            if has_negative:
                wcnf.append([-xi,  xj,  s])   # xi=1,xj=0 -> s=1
                wcnf.append([ xi, -xj,  s])   # xi=0,xj=1 -> s=1
            if w >= 0.0:
                iw = max(1, int(round(w * _SCALE)))
                total_pos_scaled += iw
                wcnf.append([s], weight=iw)
            else:
                iw = max(1, int(round(-w * _SCALE)))
                baseline_neg += w
                neg_scaled_total += iw
                wcnf.append([-s], weight=iw)
            edge_idx += 1
    return wcnf, total_pos_scaled, neg_scaled_total, baseline_neg


def _model_to_bits(model, n):
    bits = np.zeros(n, dtype=np.bool_)
    for lit in model:
        var = abs(lit) - 1
        if 0 <= var < n:
            bits[var] = lit > 0
    return bits


def solve_maxcut_exact_streaming(
    G_func,
    nodes,
    best_guess=None,
    quality=None,
    shots=None,
    is_spin_glass=False,
    anneal_t=None,
    anneal_h=None,
    repulsion_base=None,
    verbose=True,
    time_limit=None,
    gray_iterations=None,
    gray_seed_multiple=None,
):
    """
    Exact MAXCUT/spin-glass solver via weighted MaxSAT (RC2), streaming variant.

    Accepts the same G_func + nodes input as spin_glass_solver_streaming and
    the same warm-start formats (str, int, list, or None).

    Requires:  pip install python-sat

    Parameters
    ----------
    G_func : callable(u, v) -> float
        Edge weight function (Numba-callable). Returns 0.0 for absent edges.
    nodes : array-like
        Ordered sequence of node identifiers.
    best_guess : str | int | list[bool] | None
    quality, shots, anneal_t, anneal_h, repulsion_base, is_spin_glass,
    gray_iterations, gray_seed_multiple
        Forwarded to spin_glass_solver_streaming when best_guess is None.
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
            "python-sat is required for solve_maxcut_exact_streaming. "
            "Install it with:  pip install python-sat"
        )

    nodes = np.asarray(nodes)
    n_qubits = len(nodes)

    if n_qubits < 3:
        if n_qubits == 0:
            return "", 0, ([], []), 0, True
        if n_qubits == 1:
            return "0", 0, (list(nodes), []), 0, True
        if n_qubits == 2:
            weight = G_func(nodes[0], nodes[1])
            if weight < 0.0:
                return "00", 0, (list(nodes), []), weight, True
            return "01", weight, ([nodes[0]], [nodes[1]]), -weight, True

    if isinstance(best_guess, str):
        bitstring = best_guess
    elif isinstance(best_guess, int):
        bitstring = int_to_bitstring(best_guess, n_qubits)
    elif isinstance(best_guess, list):
        bitstring = "".join(["1" if b else "0" for b in best_guess])
    else:
        if verbose:
            print("Running PyQrackIsing heuristic...")
        from .spin_glass_solver_streaming import spin_glass_solver_streaming
        kwargs = {}
        if quality is not None:            kwargs["quality"]            = quality
        if shots is not None:              kwargs["shots"]              = shots
        if anneal_t is not None:           kwargs["anneal_t"]           = anneal_t
        if anneal_h is not None:           kwargs["anneal_h"]           = anneal_h
        if repulsion_base is not None:     kwargs["repulsion_base"]     = repulsion_base
        if gray_iterations is not None:    kwargs["gray_iterations"]    = gray_iterations
        if gray_seed_multiple is not None: kwargs["gray_seed_multiple"] = gray_seed_multiple
        t0 = time.monotonic()
        bitstring, cut_value, _, _ = spin_glass_solver_streaming(
            G_func, nodes, is_spin_glass=is_spin_glass, **kwargs
        )
        if verbose:
            print(f"Heuristic value: {cut_value:.6f}  ({time.monotonic()-t0:.3f}s)")

    warm_theta = np.array([b == "1" for b in list(bitstring)], dtype=np.bool_)
    has_negative = any(
        G_func(nodes[i], nodes[j]) < 0.0
        for i in range(n_qubits) for j in range(i + 1, n_qubits)
    )
    wcnf, total_pos_scaled, neg_scaled_total, baseline_neg = _build_wcnf_streaming(
        G_func, nodes, n_qubits, has_negative
    )

    if verbose:
        print("Starting RC2 MaxSAT solver...")

    rc2_kwargs = {}
    if time_limit is not None:
        rc2_kwargs["time_limit"] = time_limit

    t0 = time.monotonic()
    with RC2(wcnf, solver="g3", **rc2_kwargs) as rc2:
        try:
            phases = [i + 1 if warm_theta[i] else -(i + 1)
                      for i in range(len(warm_theta))]
            rc2.oracle.set_phases(phases)
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
        cut_opt = baseline_neg + (total_pos_scaled + neg_scaled_total - cost_scaled) / _SCALE
        print(f"RC2 optimum: {cut_opt:.6f}  ({elapsed:.3f}s)")

    bitstring, l, r = get_cut(best_theta, list(nodes), n_qubits)
    if is_spin_glass:
        cut_value = float(compute_cut_streaming(best_theta, G_func, nodes, n_qubits))
        min_energy = -float(compute_energy_streaming(best_theta, G_func, nodes, n_qubits))
    else:
        cut_value = float(compute_cut_streaming(best_theta, G_func, nodes, n_qubits))
        min_energy = float(compute_energy_streaming(best_theta, G_func, nodes, n_qubits))

    return bitstring, cut_value, (l, r), min_energy, certified
