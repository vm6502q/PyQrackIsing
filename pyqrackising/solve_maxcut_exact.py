# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Daniel Strano and the Qrack contributors
#
# Initial draft produced by (Anthropic) Claude (Sonnet 4.6).
#
# bnb_exact.py — warm-start branch-and-bound exact MAXCUT solver
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
from .maxcut_tfim_util import compute_cut, compute_energy

try:
    from scipy.optimize import linprog
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# Numba-compiled inner loops
# ---------------------------------------------------------------------------

# Scoring delegates to maxcut_tfim_util so all PyQrackIsing solvers
# use a single consistent implementation.
def _cut_value(Q, bits, n_qubits=None):
    if n_qubits is None:
        n_qubits = len(bits)
    return compute_cut(bits, Q, n_qubits)


def _energy_value(Q, bits, n_qubits=None):
    if n_qubits is None:
        n_qubits = len(bits)
    return compute_energy(bits, Q, n_qubits)


@njit(cache=True)
def _fixed_fixed(Q, fixed_idx, fixed_val):
    """
    Exact cut contribution from already-branched variable pairs.
    fixed_idx / fixed_val are parallel int64 arrays of (variable, value) pairs.
    """
    total = 0.0
    nf = len(fixed_idx)
    for a in range(nf):
        i = fixed_idx[a]
        fi = fixed_val[a]
        for b in range(a + 1, nf):
            j = fixed_idx[b]
            fj = fixed_val[b]
            ii, jj = (i, j) if i < j else (j, i)
            if Q[ii, jj] != 0.0 and fi != fj:
                total += Q[ii, jj]
    return total


@njit(cache=True)
def _fixed_free_linear(Q, free_arr, fixed_idx, fixed_val):
    """
    Decompose fixed-to-free edge contributions into a scalar constant and a
    per-free-variable linear coefficient vector, for use in the LP objective.

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
            w = Q[ii, jj]
            if w == 0.0:
                continue
            const += w * fval
            coeff[k] += w * (1.0 - 2.0 * fval)
    return const, coeff


@njit(cache=True)
def _influence_scores(Q, free_arr):
    """
    Branch-variable selection heuristic: for each free variable, sum the
    absolute weights of its edges to all other free variables (plus diagonal).
    The variable with the highest score is chosen to branch on next.
    """
    n_free = len(free_arr)
    scores = np.zeros(n_free)
    for k in range(n_free):
        i = free_arr[k]
        scores[k] += abs(Q[i, i])
        for m in range(n_free):
            if k == m:
                continue
            j = free_arr[m]
            ii, jj = (i, j) if i < j else (j, i)
            scores[k] += abs(Q[ii, jj])
    return scores


# ---------------------------------------------------------------------------
# LP upper bound (Python, calls scipy HiGHS)
# ---------------------------------------------------------------------------

def _lp_upper_bound(Q, free_arr, fixed_idx, fixed_val, n):
    """
    LP relaxation upper bound for the MAXCUT subproblem defined by fixed_idx /
    fixed_val. Fixed-variable contributions are computed via Numba helpers;
    the free-free relaxation is solved by scipy HiGHS.

    Falls back to a loose but valid bound when scipy is unavailable.
    """
    ff = _fixed_fixed(Q, fixed_idx, fixed_val)
    n_free = len(free_arr)

    if n_free == 0:
        return ff

    ff_const, coeff = _fixed_free_linear(Q, free_arr, fixed_idx, fixed_val)

    if not _HAVE_SCIPY:
        # Loose valid bound: take the best-case value of every free edge.
        bound = ff + ff_const
        for k in range(n_free):
            bound += max(0.0, coeff[k])
        for a in range(n_free):
            i = free_arr[a]
            for b in range(a + 1, n_free):
                j = free_arr[b]
                bound += max(0.0, Q[i, j])
        return bound

    pairs = [(free_arr[a], free_arr[b])
             for a in range(n_free) for b in range(a + 1, n_free)
             if Q[free_arr[a], free_arr[b]] != 0.0]
    n_pairs = len(pairs)
    total_vars = n_free + n_pairs
    free_idx_map = {int(v): k for k, v in enumerate(free_arr)}

    # LP objective (negated for minimisation).
    # Free-free edge (i,j) contributes w*(x_i + x_j - 2*y_ij) to the MAXCUT
    # objective, so in the minimisation objective:
    #   c[x_i] -= w,  c[x_j] -= w,  c[y_ij] += 2*w
    c = np.zeros(total_vars)
    for k in range(n_free):
        c[k] = -coeff[k]
    for p, (i, j) in enumerate(pairs):
        w = Q[i, j]
        c[free_idx_map[i]] -= w
        c[free_idx_map[j]] -= w
        c[n_free + p] += 2.0 * w

    bounds = [(0.0, 1.0)] * total_vars

    if not pairs:
        lp_val = sum(-c[k] if c[k] < 0.0 else 0.0 for k in range(n_free))
        return ff + ff_const + lp_val

    # Linearisation constraints: y_ij <= x_i,  y_ij <= x_j,  x_i+x_j-y_ij <= 1
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
# Warm-start parser  (matches PyQrackIsing solver convention)
# ---------------------------------------------------------------------------

def _parse_warm_start(
    best_guess,
    n_qubits,
    Q,
    G=None,
    quality=None,
    shots=None,
    is_spin_glass=False,
    anneal_t=None,
    anneal_h=None,
    repulsion_base=None,
    is_maxcut_gpu=True,
):
    """
    Resolve a warm-start hint into (bits: ndarray[bool], cut_val: float).

    Accepted formats (matching PyQrackIsing solver convention):
      str        — bitstring of '0'/'1' characters, length n_qubits
      int        — integer whose binary representation is the bitstring
                   (LSB = node 0, matching int_to_bitstring convention)
      list[bool] — list of per-node boolean / 0-1 values
      None       — call spin_glass_solver_sparse if G is provided,
                   otherwise fall back to a random partition
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
        if G is not None:
            from .spin_glass_solver import spin_glass_solver
            kwargs = {}
            if quality is not None:        kwargs["quality"]        = quality
            if shots is not None:          kwargs["shots"]          = shots
            if anneal_t is not None:       kwargs["anneal_t"]       = anneal_t
            if anneal_h is not None:       kwargs["anneal_h"]       = anneal_h
            if repulsion_base is not None: kwargs["repulsion_base"] = repulsion_base
            kwargs["is_maxcut_gpu"] = is_maxcut_gpu
            bitstring, cut_val, _, _ = spin_glass_solver(
                G, is_spin_glass=is_spin_glass, **kwargs
            )
        else:
            bits = np.random.randint(0, 2, size=n_qubits).astype(np.bool_)
            score_fn = _energy_value if is_spin_glass else _cut_value
            return bits, float(score_fn(Q, bits, n_qubits))

    bits = np.array([b == "1" for b in bitstring], dtype=np.bool_)
    # Always recompute using the util scoring so the heuristic value shown
    # is guaranteed consistent with the warm-start incumbent.
    n_qubits = len(bits)
    cut_val = float(
        _energy_value(Q, bits, n_qubits) if is_spin_glass
        else _cut_value(Q, bits, n_qubits)
    )
    return bits, cut_val


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def maxcut_branch_and_bound(
    Q,
    best_guess=None,
    G=None,
    quality=None,
    shots=None,
    is_spin_glass=False,
    anneal_t=None,
    anneal_h=None,
    repulsion_base=None,
    is_maxcut_gpu=True,
    verbose=True,
    time_limit=None,
):
    """
    Exact MAXCUT solver via branch and bound with LP relaxation upper bounds.

    All inner loops (cut evaluation, fixed-contribution accounting, branching
    heuristic) are Numba JIT-compiled. The LP bound uses scipy HiGHS.

    Parameters
    ----------
    Q : (n, n) float64 ndarray
        Dense upper-triangular edge weight matrix, no diagonal.
        Objective: maximise sum_{i<j, bits[i]!=bits[j]} Q[i,j].
    best_guess : str | int | list[bool] | None
        Warm-start hint in any PyQrackIsing-compatible format:
          str        — '0101...' bitstring
          int        — integer encoding of the bitstring (LSB = node 0)
          list[bool] — list of per-node boolean values
          None       — call spin_glass_solver_sparse (requires G) or
                       use a random partition
    G : networkx.Graph or None
        Passed to spin_glass_solver_sparse when best_guess is None.
    quality, shots, anneal_t, anneal_h, repulsion_base, is_maxcut_gpu, is_spin_glass
        Forwarded to spin_glass_solver_sparse when best_guess is None.
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
    Q = np.asarray(Q, dtype=np.float64)
    n = Q.shape[0]
    assert Q.shape == (n, n), "Q must be square"

    best_bits, best_value = _parse_warm_start(
        best_guess, n, Q, G=G,
        quality=quality, shots=shots, is_spin_glass=is_spin_glass,
        anneal_t=anneal_t, anneal_h=anneal_h,
        repulsion_base=repulsion_base, is_maxcut_gpu=is_maxcut_gpu,
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

        ub = _lp_upper_bound(Q, free_arr, fixed_idx, fixed_val, n)

        if ub <= best_value + 1e-9:
            nodes_pruned += 1
            continue

        if len(free_arr) == 0:
            bits = np.zeros(n, dtype=np.bool_)
            for k in range(len(fixed_idx)):
                bits[fixed_idx[k]] = bool(fixed_val[k])
            val = float(
                _energy_value(Q, bits, n) if is_spin_glass
                else _cut_value(Q, bits, n)
            )
            if val > best_value:
                best_value = val
                best_bits = bits.copy()
                if verbose:
                    print(f"  New incumbent: {best_value:.6f}"
                          f"  (nodes: {nodes_explored})")
            continue

        scores = _influence_scores(Q, free_arr)
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
):
    """
    Run spin_glass_solver_sparse on G to produce a warm start (unless
    best_guess is supplied), then certify it via branch and bound.

    Parameters
    ----------
    G : networkx.Graph
        Graph with arbitrary edge weights, no self-loops.
        Node labels must be contiguous integers 0..n-1.
    best_guess : str | int | list[bool] | None
        Optional warm-start hint in any PyQrackIsing-compatible format.
        If None, spin_glass_solver_sparse is called to produce one.
    quality, shots, anneal_t, anneal_h, repulsion_base, is_maxcut_gpu, is_spin_glass
        Forwarded to spin_glass_solver_sparse when best_guess is None.
    verbose : bool
    time_limit : float or None
        Wall-clock seconds for the B&B phase.

    Returns
    -------
    heuristic_bits : ndarray[bool]
    heuristic_value : float
    exact_bits : ndarray[bool]
    exact_value : float
    certified : bool
    """
    Q = _graph_to_dense_upper(G)
    n = Q.shape[0]

    if verbose and best_guess is None:
        print("Running PyQrackIsing heuristic...")

    t0 = time.monotonic()
    heuristic_bits, heuristic_value = _parse_warm_start(
        best_guess, n, Q, G=G,
        quality=quality, shots=shots, is_spin_glass=is_spin_glass,
        anneal_t=anneal_t, anneal_h=anneal_h,
        repulsion_base=repulsion_base, is_maxcut_gpu=is_maxcut_gpu,
    )
    t_heuristic = time.monotonic() - t0

    if verbose:
        print(f"Heuristic value: {heuristic_value:.6f}  ({t_heuristic:.3f}s)")
        print("Starting branch and bound...")

    exact_bits, exact_value, certified = maxcut_branch_and_bound(
        Q,
        best_guess=heuristic_bits,
        verbose=verbose,
        time_limit=time_limit,
    )

    if verbose:
        gap = exact_value - heuristic_value
        print(f"\nHeuristic gap closed: {gap:.6f}  "
              f"({'certified exact' if certified else 'time-limited'})")

    return heuristic_bits, heuristic_value, exact_bits, exact_value, certified


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _graph_to_dense_upper(G):
    """
    Convert a NetworkX graph to a dense upper-triangular float64 numpy matrix.
    Node labels must be contiguous integers 0..n-1.
    """
    nodes = sorted(G.nodes())
    n = len(nodes)
    Q = np.zeros((n, n), dtype=np.float64)
    for u, v, data in G.edges(data=True):
        i, j = (u, v) if u < v else (v, u)
        Q[i, j] = float(data.get("weight", 1.0))
    return Q
