# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Daniel Strano and the Qrack contributors
#
# Initial draft produced by (Anthropic) Claude (Sonnet 4.6).
#
# bnb_exact_streaming.py — warm-start branch-and-bound exact MAXCUT solver
#                           (streaming / G_func + nodes variant)
#
# Objective convention (matches PyQrackIsing throughout):
#   Maximise  sum_{(i,j) in cut} w_{ij}
#   i.e. edge (i,j) contributes w_{ij} when bits[i] != bits[j].
#   Edge weights may be positive or negative; no diagonal / self-loops.
#
# Upper bound per B&B node: LP relaxation of the MAXCUT objective.
#   Linearisation: introduce y_ij = x_i * x_j for each free edge pair;
#   edge contribution becomes w_{ij} * (x_i + x_j - 2*y_ij).
#   Constraints: y_ij <= x_i, y_ij <= x_j, x_i + x_j - y_ij <= 1.
#   Users may swap in a tighter SDP bound by replacing _lp_upper_bound.

import time
import numpy as np
from numba import njit, prange
from .maxcut_tfim_util import (
    compute_cut_streaming,
    compute_energy_streaming,
    get_cut,
    heuristic_threshold,
    int_to_bitstring,
    opencl_context,
)

try:
    from scipy.optimize import linprog
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


dtype = opencl_context.dtype


# ---------------------------------------------------------------------------
# Numba-compiled inner loops (streaming / G_func)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _fixed_fixed_streaming(G_func, nodes, fixed_idx, fixed_val):
    total = 0.0
    nf = len(fixed_idx)
    for a in range(nf):
        i = fixed_idx[a]
        fi = fixed_val[a]
        for b in range(a + 1, nf):
            j = fixed_idx[b]
            fj = fixed_val[b]
            if fi != fj:
                ii, jj = (i, j) if i < j else (j, i)
                w = G_func(nodes[ii], nodes[jj])
                if w != 0.0:
                    total += w
    return total


@njit(cache=True)
def _fixed_free_linear_streaming(G_func, nodes, free_arr, fixed_idx, fixed_val):
    n_free = len(free_arr)
    nf = len(fixed_idx)
    coeff = np.zeros(n_free)
    const = 0.0
    for k in range(n_free):
        i = free_arr[k]
        for a in range(nf):
            j = fixed_idx[a]
            fval = fixed_val[a]
            ii, jj = (i, j) if i < j else (j, i)
            w = G_func(nodes[ii], nodes[jj])
            if w == 0.0:
                continue
            const += w * fval
            coeff[k] += w * (1.0 - 2.0 * fval)
    return const, coeff


@njit(cache=True)
def _influence_scores_streaming(G_func, nodes, free_arr):
    n_free = len(free_arr)
    scores = np.zeros(n_free)
    for k in range(n_free):
        i = free_arr[k]
        for m in range(n_free):
            if k == m:
                continue
            j = free_arr[m]
            ii, jj = (i, j) if i < j else (j, i)
            scores[k] += abs(G_func(nodes[ii], nodes[jj]))
    return scores


# ---------------------------------------------------------------------------
# LP upper bound
# ---------------------------------------------------------------------------

def _lp_upper_bound(G_func, nodes, free_arr, fixed_idx, fixed_val, n):
    ff = _fixed_fixed_streaming(G_func, nodes, fixed_idx, fixed_val)
    n_free = len(free_arr)

    if n_free == 0:
        return ff

    ff_const, coeff = _fixed_free_linear_streaming(
        G_func, nodes, free_arr, fixed_idx, fixed_val
    )

    # indices into free_arr, not node indices, since LP vars are indexed by position
    pairs = []
    for a in range(n_free):
        i = int(free_arr[a])
        for b in range(a + 1, n_free):
            j = int(free_arr[b])
            ii, jj = (i, j) if i < j else (j, i)
            w = float(G_func(nodes[ii], nodes[jj]))
            if w != 0.0:
                pairs.append((a, b, w))

    if not _HAVE_SCIPY:
        bound = ff + ff_const
        for k in range(n_free):
            bound += max(0.0, coeff[k])
        for _, _, w in pairs:
            bound += max(0.0, w)
        return bound

    n_pairs = len(pairs)
    total_vars = n_free + n_pairs

    c = np.zeros(total_vars)
    for k in range(n_free):
        c[k] = -coeff[k]
    for p, (ka, kb, w) in enumerate(pairs):
        c[ka] -= w
        c[kb] -= w
        c[n_free + p] += 2.0 * w

    bounds = [(0.0, 1.0)] * total_vars

    if not pairs:
        lp_val = sum(-c[k] if c[k] < 0.0 else 0.0 for k in range(n_free))
        return ff + ff_const + lp_val

    rows = 3 * n_pairs
    A = np.zeros((rows, total_vars))
    b_ub = np.empty(rows)
    for p, (ka, kb, _) in enumerate(pairs):
        r = 3 * p
        A[r,   ka] = -1.0; A[r,   n_free + p] =  1.0; b_ub[r]   = 0.0
        A[r+1, kb] = -1.0; A[r+1, n_free + p] =  1.0; b_ub[r+1] = 0.0
        A[r+2, ka] =  1.0; A[r+2, kb]         =  1.0
        A[r+2, n_free + p] = -1.0;                      b_ub[r+2] = 1.0

    res = linprog(c, A_ub=A, b_ub=b_ub, bounds=bounds,
                  method="highs", options={"disp": False})

    return (ff + ff_const + (-res.fun)) if res.success else float("inf")


# ---------------------------------------------------------------------------
# Branch and bound (internal)
# ---------------------------------------------------------------------------

def _branch_and_bound_streaming(G_func, nodes, warm_theta, warm_energy, n,
                                 is_spin_glass, verbose=True, time_limit=None):
    best_bits = warm_theta.copy()
    best_value = warm_energy

    if verbose:
        print(f"Warm-start incumbent: {best_value:.6f}")

    t_start = time.monotonic()
    nodes_explored = 0
    nodes_pruned = 0

    stack = [(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))]

    while stack:
        if time_limit is not None and (time.monotonic() - t_start) > time_limit:
            if verbose:
                print(f"Time limit reached. "
                      f"Nodes: {nodes_explored}, Pruned: {nodes_pruned}")
            return best_bits, best_value, False

        fixed_idx, fixed_val = stack.pop()
        nodes_explored += 1

        fixed_set = set(int(fixed_idx[k]) for k in range(len(fixed_idx)))
        free_arr = np.array([i for i in range(n) if i not in fixed_set],
                            dtype=np.int64)

        ub = _lp_upper_bound(G_func, nodes, free_arr, fixed_idx, fixed_val, n)

        if ub <= best_value + 1e-9:
            nodes_pruned += 1
            continue

        if len(free_arr) == 0:
            bits = np.zeros(n, dtype=np.bool_)
            for k in range(len(fixed_idx)):
                bits[fixed_idx[k]] = bool(fixed_val[k])
            val = float(
                compute_energy_streaming(bits, G_func, nodes, n)
                if is_spin_glass
                else compute_cut_streaming(bits, G_func, nodes, n)
            )
            if val > best_value:
                best_value = val
                best_bits = bits.copy()
                if verbose:
                    print(f"  New incumbent: {best_value:.6f}"
                          f"  (nodes: {nodes_explored})")
            continue

        scores = _influence_scores_streaming(G_func, nodes, free_arr)
        branch_var = int(free_arr[int(np.argmax(scores))])

        warm_val = int(best_bits[branch_var])
        for val in [warm_val, 1 - warm_val]:
            stack.append((
                np.append(fixed_idx, branch_var).astype(np.int64),
                np.append(fixed_val, val).astype(np.int64),
            ))

    elapsed = time.monotonic() - t_start
    if verbose:
        print(f"\nExact optimum: {best_value:.6f}")
        print(f"Nodes explored: {nodes_explored}  |  "
              f"Pruned: {nodes_pruned}  |  Time: {elapsed:.3f}s")

    return best_bits, best_value, True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    Exact MAXCUT/spin-glass solver: warm-start from spin_glass_solver_streaming
    then certify via branch and bound.

    Accepts the same G_func + nodes input as spin_glass_solver_streaming and
    the same warm-start formats (str, int, list, or None).

    Parameters
    ----------
    G_func : callable(u, v) -> float
        Edge weight function (Numba-callable). Returns 0.0 for absent edges.
    nodes : array-like
        Ordered sequence of node identifiers passed to G_func.
    best_guess : str | int | list[bool] | None
        Warm-start hint in any PyQrackIsing-compatible format. If None,
        spin_glass_solver_streaming is called to produce one.
    quality, shots, anneal_t, anneal_h, repulsion_base, is_spin_glass,
    gray_iterations, gray_seed_multiple
        Forwarded to spin_glass_solver_streaming when best_guess is None.
    verbose : bool
    time_limit : float or None
        Wall-clock seconds for the B&B phase.

    Returns
    -------
    bitstring : str
    cut_value : float
    partition : tuple(list, list)
    min_energy : float
    certified : bool
    """
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

    # Warm-start parser — mirrors spin_glass_solver_streaming exactly
    bitstring = ""
    cut_value = 0.0
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
        if quality is not None:           kwargs["quality"]            = quality
        if shots is not None:             kwargs["shots"]              = shots
        if anneal_t is not None:          kwargs["anneal_t"]           = anneal_t
        if anneal_h is not None:          kwargs["anneal_h"]           = anneal_h
        if repulsion_base is not None:    kwargs["repulsion_base"]     = repulsion_base
        if gray_iterations is not None:   kwargs["gray_iterations"]    = gray_iterations
        if gray_seed_multiple is not None: kwargs["gray_seed_multiple"] = gray_seed_multiple
        t0 = time.monotonic()
        bitstring, cut_value, _, _ = spin_glass_solver_streaming(
            G_func, nodes, is_spin_glass=is_spin_glass, **kwargs
        )
        if verbose:
            print(f"Heuristic value: {cut_value:.6f}  ({time.monotonic()-t0:.3f}s)")

    best_theta = np.array([b == "1" for b in list(bitstring)], dtype=np.bool_)
    max_energy = (
        compute_energy_streaming(best_theta, G_func, nodes, n_qubits)
        if is_spin_glass
        else compute_cut_streaming(best_theta, G_func, nodes, n_qubits)
    )

    if verbose:
        print("Starting branch and bound...")

    best_theta, max_energy, certified = _branch_and_bound_streaming(
        G_func, nodes, best_theta, max_energy, n_qubits, is_spin_glass,
        verbose=verbose, time_limit=time_limit,
    )

    bitstring, l, r = get_cut(best_theta, list(nodes), n_qubits)
    if is_spin_glass:
        cut_value = compute_cut_streaming(best_theta, G_func, nodes, n_qubits)
        min_energy = -max_energy
    else:
        cut_value = max_energy
        min_energy = compute_energy_streaming(best_theta, G_func, nodes, n_qubits)

    return bitstring, float(cut_value), (l, r), float(min_energy), certified
