# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Daniel Strano and the Qrack contributors
#
# Initial draft produced by (Anthropic) Claude (Sonnet 4.6).
#
# bnb_exact.py — warm-start branch-and-bound exact MAXCUT solver (dense)
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
import networkx as nx
import numpy as np
from numba import njit, prange
from .maxcut_tfim_util import (
    compute_cut,
    compute_energy,
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
# Numba-compiled inner loops
# ---------------------------------------------------------------------------

@njit(cache=True)
def _fixed_fixed(G_m, fixed_idx, fixed_val):
    total = 0.0
    nf = len(fixed_idx)
    for a in range(nf):
        i = fixed_idx[a]
        fi = fixed_val[a]
        for b in range(a + 1, nf):
            j = fixed_idx[b]
            fj = fixed_val[b]
            ii, jj = (i, j) if i < j else (j, i)
            if G_m[ii, jj] != 0.0 and fi != fj:
                total += G_m[ii, jj]
    return total


@njit(cache=True)
def _fixed_free_linear(G_m, free_arr, fixed_idx, fixed_val):
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
            w = G_m[ii, jj]
            if w == 0.0:
                continue
            const += w * fval
            coeff[k] += w * (1.0 - 2.0 * fval)
    return const, coeff


@njit(cache=True)
def _influence_scores(G_m, free_arr):
    n_free = len(free_arr)
    scores = np.zeros(n_free)
    for k in range(n_free):
        i = free_arr[k]
        for m in range(n_free):
            if k == m:
                continue
            j = free_arr[m]
            ii, jj = (i, j) if i < j else (j, i)
            scores[k] += abs(G_m[ii, jj])
    return scores


# ---------------------------------------------------------------------------
# LP upper bound
# ---------------------------------------------------------------------------

def _lp_upper_bound(G_m, free_arr, fixed_idx, fixed_val, n):
    ff = _fixed_fixed(G_m, fixed_idx, fixed_val)
    n_free = len(free_arr)

    if n_free == 0:
        return ff

    ff_const, coeff = _fixed_free_linear(G_m, free_arr, fixed_idx, fixed_val)

    if not _HAVE_SCIPY:
        bound = ff + ff_const
        for k in range(n_free):
            bound += max(0.0, coeff[k])
        for a in range(n_free):
            i = free_arr[a]
            for b in range(a + 1, n_free):
                j = free_arr[b]
                bound += max(0.0, G_m[i, j])
        return bound

    pairs = [(free_arr[a], free_arr[b])
             for a in range(n_free) for b in range(a + 1, n_free)
             if G_m[free_arr[a], free_arr[b]] != 0.0]
    n_pairs = len(pairs)
    total_vars = n_free + n_pairs
    free_idx_map = {int(v): k for k, v in enumerate(free_arr)}

    c = np.zeros(total_vars)
    for k in range(n_free):
        c[k] = -coeff[k]
    for p, (i, j) in enumerate(pairs):
        w = G_m[i, j]
        c[free_idx_map[i]] -= w
        c[free_idx_map[j]] -= w
        c[n_free + p] += 2.0 * w

    bounds = [(0.0, 1.0)] * total_vars

    if not pairs:
        lp_val = sum(-c[k] if c[k] < 0.0 else 0.0 for k in range(n_free))
        return ff + ff_const + lp_val

    rows = 3 * n_pairs
    A = np.zeros((rows, total_vars))
    b_ub = np.empty(rows)
    for p, (i, j) in enumerate(pairs):
        ki, kj = free_idx_map[i], free_idx_map[j]
        r = 3 * p
        A[r,   ki] = -1.0; A[r,   n_free + p] =  1.0; b_ub[r]   = 0.0
        A[r+1, kj] = -1.0; A[r+1, n_free + p] =  1.0; b_ub[r+1] = 0.0
        A[r+2, ki] =  1.0; A[r+2, kj]         =  1.0
        A[r+2, n_free + p] = -1.0;                      b_ub[r+2] = 1.0

    res = linprog(c, A_ub=A, b_ub=b_ub, bounds=bounds,
                  method="highs", options={"disp": False})

    return (ff + ff_const + (-res.fun)) if res.success else float("inf")


# ---------------------------------------------------------------------------
# Branch and bound (internal)
# ---------------------------------------------------------------------------

def _branch_and_bound(G_m, warm_theta, warm_energy, n, is_spin_glass,
                      verbose=True, time_limit=None):
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

        ub = _lp_upper_bound(G_m, free_arr, fixed_idx, fixed_val, n)

        if ub <= best_value + 1e-9:
            nodes_pruned += 1
            continue

        if len(free_arr) == 0:
            bits = np.zeros(n, dtype=np.bool_)
            for k in range(len(fixed_idx)):
                bits[fixed_idx[k]] = bool(fixed_val[k])
            val = float(
                compute_energy(bits, G_m, n) if is_spin_glass
                else compute_cut(bits, G_m, n)
            )
            if val > best_value:
                best_value = val
                best_bits = bits.copy()
                if verbose:
                    print(f"  New incumbent: {best_value:.6f}"
                          f"  (nodes: {nodes_explored})")
            continue

        scores = _influence_scores(G_m, free_arr)
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

def solve_maxcut_exact(
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
    Exact MAXCUT/spin-glass solver: warm-start from spin_glass_solver then
    certify via branch and bound.

    Accepts the same G input as spin_glass_solver (NetworkX graph or dense
    matrix) and the same warm-start formats (str, int, list, or None).

    Parameters
    ----------
    G : networkx.Graph or ndarray
        Graph or pre-built dense adjacency matrix.
    best_guess : str | int | list[bool] | None
        Warm-start hint in any PyQrackIsing-compatible format. If None,
        spin_glass_solver is called to produce one.
    quality, shots, anneal_t, anneal_h, repulsion_base, is_maxcut_gpu,
    is_spin_glass, gray_iterations, gray_seed_multiple
        Forwarded to spin_glass_solver when best_guess is None.
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
    # Mirror spin_glass_solver's G -> G_m conversion exactly
    if isinstance(G, nx.Graph):
        nodes = list(G.nodes())
        n_qubits = len(nodes)
        G_m = nx.to_numpy_array(G, weight="weight", nonedge=0.0, dtype=dtype)
    else:
        n_qubits = len(G)
        nodes = list(range(n_qubits))
        G_m = np.asarray(G, dtype=dtype)

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

    # Warm-start parser — mirrors spin_glass_solver exactly
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
        from .spin_glass_solver import spin_glass_solver
        kwargs = {}
        if quality is not None:           kwargs["quality"]            = quality
        if shots is not None:             kwargs["shots"]              = shots
        if anneal_t is not None:          kwargs["anneal_t"]           = anneal_t
        if anneal_h is not None:          kwargs["anneal_h"]           = anneal_h
        if repulsion_base is not None:    kwargs["repulsion_base"]     = repulsion_base
        if gray_iterations is not None:   kwargs["gray_iterations"]    = gray_iterations
        if gray_seed_multiple is not None: kwargs["gray_seed_multiple"] = gray_seed_multiple
        kwargs["is_maxcut_gpu"] = is_maxcut_gpu
        t0 = time.monotonic()
        bitstring, cut_value, _, _ = spin_glass_solver(
            G_m, is_spin_glass=is_spin_glass, **kwargs
        )
        if verbose:
            print(f"Heuristic value: {cut_value:.6f}  ({time.monotonic()-t0:.3f}s)")

    best_theta = np.array([b == "1" for b in list(bitstring)], dtype=np.bool_)
    if is_spin_glass:
        max_energy = compute_energy(best_theta, G_m, n_qubits)
    elif cut_value is None:
        max_energy = compute_cut(best_theta, G_m, n_qubits)
    else:
        max_energy = cut_value

    if verbose:
        print("Starting branch and bound...")

    best_theta, max_energy, certified = _branch_and_bound(
        G_m, best_theta, max_energy, n_qubits, is_spin_glass,
        verbose=verbose, time_limit=time_limit,
    )

    bitstring, l, r = get_cut(best_theta, nodes, n_qubits)
    if is_spin_glass:
        cut_value = compute_cut(best_theta, G_m, n_qubits)
        min_energy = -max_energy
    else:
        cut_value = max_energy
        min_energy = compute_energy(best_theta, G_m, n_qubits)

    return bitstring, float(cut_value), (l, r), float(min_energy), certified
