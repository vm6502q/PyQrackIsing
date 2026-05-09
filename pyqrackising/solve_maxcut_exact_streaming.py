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
#
# Numba is used for all inner loops (cut evaluation, branching heuristic,
# fixed-contribution accounting). The LP solver call (scipy HiGHS) stays
# in Python since JIT-compiling an LP solver is not practical.

import time
import numpy as np
from numba import njit, prange
from .maxcut_tfim_util import compute_cut_streaming, compute_energy_streaming

try:
    from scipy.optimize import linprog
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# Numba-compiled inner loops (streaming / G_func)
# ---------------------------------------------------------------------------

# Scoring delegates to maxcut_tfim_util so all PyQrackIsing solvers
# use a single consistent implementation.
def _cut_value_streaming(G_func, nodes, bits):
    n_qubits = len(nodes)
    return compute_cut_streaming(bits, G_func, nodes, n_qubits)


def _energy_value_streaming(G_func, nodes, bits):
    n_qubits = len(nodes)
    return compute_energy_streaming(bits, G_func, nodes, n_qubits)


@njit(cache=True)
def _fixed_fixed_streaming(G_func, nodes, fixed_idx, fixed_val):
    """
    Exact cut contribution from already-branched variable pairs (streaming).
    """
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
    """
    Decompose fixed-to-free edge contributions for the LP objective (streaming).

      MAXCUT contribution of fixed-free edges
        = fixed_free_const + sum_k coeff[k] * x_k

    where:
      fixed_free_const = sum_{k,fj} w_{k,fj} * fval_fj
      coeff[k]         = sum_{fj} w_{k,fj} * (1 - 2 * fval_fj)
    """
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
    """
    Branch-variable selection: sum of absolute edge weights to other free
    variables for each free variable (streaming).
    """
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
# LP upper bound (Python, calls scipy HiGHS)
# ---------------------------------------------------------------------------

def _lp_upper_bound(G_func, nodes, free_arr, fixed_idx, fixed_val, n):
    """
    LP relaxation upper bound for the MAXCUT subproblem (streaming).
    Fixed-variable contributions computed via Numba helpers;
    free-free relaxation solved by scipy HiGHS.
    Falls back to a loose but valid bound when scipy is unavailable.
    """
    ff = _fixed_fixed_streaming(G_func, nodes, fixed_idx, fixed_val)
    n_free = len(free_arr)

    if n_free == 0:
        return ff

    ff_const, coeff = _fixed_free_linear_streaming(
        G_func, nodes, free_arr, fixed_idx, fixed_val
    )

    # Collect free-free edges
    pairs = []
    for a in range(n_free):
        i = int(free_arr[a])
        for b in range(a + 1, n_free):
            j = int(free_arr[b])
            ii, jj = (i, j) if i < j else (j, i)
            w = float(G_func(nodes[ii], nodes[jj]))
            if w != 0.0:
                pairs.append((a, b, w))   # store indices into free_arr

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
# Warm-start parser  (matches PyQrackIsing solver convention)
# ---------------------------------------------------------------------------

def _parse_warm_start(
    best_guess,
    n_qubits,
    G_func,
    nodes,
    quality=None,
    shots=None,
    is_spin_glass=False,
    anneal_t=None,
    anneal_h=None,
    repulsion_base=None,
):
    """
    Resolve a warm-start hint into (bits: ndarray[bool], cut_val: float).

    Accepted formats (matching PyQrackIsing solver convention):
      str        — bitstring of '0'/'1' characters, length n_qubits
      int        — integer encoding (LSB = node 0)
      list[bool] — list of per-node boolean / 0-1 values
      None       — call maxcut_tfim_streaming or use a random partition
    """
    bitstring = ""
    cut_val = None

    if isinstance(best_guess, str):
        bitstring = best_guess

    elif isinstance(best_guess, int):
        bitstring = format(best_guess, f"0{n_qubits}b")[::-1]

    elif isinstance(best_guess, list):
        bitstring = "".join("1" if b else "0" for b in best_guess)

    else:
        if G_func is not None:
            from .spin_glass_solver_streaming import spin_glass_solver_streaming
            kwargs = {}
            if quality is not None:       kwargs["quality"]        = quality
            if shots is not None:         kwargs["shots"]          = shots
            if anneal_t is not None:      kwargs["anneal_t"]       = anneal_t
            if anneal_h is not None:      kwargs["anneal_h"]       = anneal_h
            if repulsion_base is not None: kwargs["repulsion_base"] = repulsion_base
            bitstring, cut_val, _ = maxcut_tfim_streaming(
                G_func, nodes, is_spin_glass=is_spin_glass, **kwargs
            )
        else:
            bits = np.random.randint(0, 2, size=n_qubits).astype(np.bool_)
            score_fn = _energy_value_streaming if is_spin_glass else _cut_value_streaming
            return bits, float(score_fn(G_func, nodes, bits))

    bits = np.array([b == "1" for b in bitstring], dtype=np.bool_)
    cut_val = float(
        _energy_value_streaming(G_func, nodes, bits) if is_spin_glass
        else _cut_value_streaming(G_func, nodes, bits)
    )
    return bits, cut_val


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def maxcut_branch_and_bound_streaming(
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
):
    """
    Exact MAXCUT solver via branch and bound with LP relaxation upper bounds
    (streaming / G_func + nodes variant).

    All inner loops are Numba JIT-compiled. The LP bound uses scipy HiGHS.

    Parameters
    ----------
    G_func : callable(u, v) -> float
        Edge weight function. G_func(u, v) returns the weight of edge (u, v),
        or 0.0 if the edge does not exist. Must be Numba-callable (@njit).
    nodes : array-like
        Ordered sequence of node identifiers passed to G_func.
    best_guess : str | int | list[bool] | None
        Warm-start hint in any PyQrackIsing-compatible format:
          str        — '0101...' bitstring
          int        — integer encoding (LSB = node 0)
          list[bool] — list of per-node boolean values
          None       — call maxcut_tfim_streaming or use a random partition
    quality, shots, anneal_t, anneal_h, repulsion_base, is_spin_glass
        Forwarded to maxcut_tfim_streaming when best_guess is None.
    verbose : bool
    time_limit : float or None
        Wall-clock seconds before returning best-so-far (not certified).

    Returns
    -------
    best_bits : ndarray[bool]
    best_value : float
    certified : bool
        True iff the returned solution is provably optimal.
    """
    nodes = np.asarray(nodes)
    n = len(nodes)

    best_bits, best_value = _parse_warm_start(
        best_guess, n, G_func, nodes,
        quality=quality, shots=shots, is_spin_glass=is_spin_glass,
        anneal_t=anneal_t, anneal_h=anneal_h, repulsion_base=repulsion_base,
    )

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
                _energy_value_streaming(G_func, nodes, bits) if is_spin_glass
                else _cut_value_streaming(G_func, nodes, bits)
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
):
    """
    Run maxcut_tfim_streaming on (G_func, nodes) to produce a warm start
    (unless best_guess is supplied), then certify it via branch and bound.

    Parameters
    ----------
    G_func : callable(u, v) -> float
        Edge weight function (Numba-callable).
    nodes : array-like
        Ordered sequence of node identifiers.
    best_guess : str | int | list[bool] | None
        Optional warm-start hint in any PyQrackIsing-compatible format.
        If None, maxcut_tfim_streaming is called to produce one.
    quality, shots, anneal_t, anneal_h, repulsion_base, is_spin_glass
        Forwarded to maxcut_tfim_streaming when best_guess is None.
    verbose : bool
    time_limit : float or None

    Returns
    -------
    heuristic_bits : ndarray[bool]
    heuristic_value : float
    exact_bits : ndarray[bool]
    exact_value : float
    certified : bool
    """
    nodes = np.asarray(nodes)
    n = len(nodes)

    if verbose and best_guess is None:
        print("Running PyQrackIsing heuristic...")

    t0 = time.monotonic()
    heuristic_bits, heuristic_value = _parse_warm_start(
        best_guess, n, G_func, nodes,
        quality=quality, shots=shots, is_spin_glass=is_spin_glass,
        anneal_t=anneal_t, anneal_h=anneal_h, repulsion_base=repulsion_base,
    )
    t_heuristic = time.monotonic() - t0

    if verbose:
        print(f"Heuristic value: {heuristic_value:.6f}  ({t_heuristic:.3f}s)")
        print("Starting branch and bound...")

    exact_bits, exact_value, certified = maxcut_branch_and_bound_streaming(
        G_func,
        nodes,
        best_guess=heuristic_bits,
        verbose=verbose,
        time_limit=time_limit,
    )

    if verbose:
        gap = exact_value - heuristic_value
        print(f"\nHeuristic gap closed: {gap:.6f}  "
              f"({'certified exact' if certified else 'time-limited'})")

    return heuristic_bits, heuristic_value, exact_bits, exact_value, certified
