# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Daniel Strano and the Qrack contributors
#
# Initial draft produced by (Anthropic) Claude (Sonnet 4.6).
#
# bnb_exact_streaming.py — warm-start branch-and-bound exact MAXCUT solver
#                           (streaming / G_func + nodes variant)
#
# Same parallelism strategy as bnb_exact.py; inner loops call G_func.

import time
import os
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

dtype = opencl_context.dtype


# ---------------------------------------------------------------------------
# Parallel kernels (streaming)
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _upper_bound_batch_streaming(G_func, nodes, fixed_vars, n, batch_size):
    ub = np.empty(batch_size)
    for b in prange(batch_size):
        total = 0.0
        for i in range(n):
            fi = fixed_vars[b, i]
            for j in range(i + 1, n):
                w = G_func(nodes[i], nodes[j])
                if w == 0.0:
                    continue
                fj = fixed_vars[b, j]
                if fi >= 0 and fj >= 0:
                    if fi != fj:
                        total += w
                else:
                    if w > 0.0:
                        total += w
        ub[b] = total
    return ub


@njit(parallel=True, cache=True)
def _eval_leaves_cut_streaming(G_func, nodes, fixed_vars, n, batch_size):
    vals = np.empty(batch_size)
    for b in prange(batch_size):
        cut = 0.0
        for i in range(n):
            bi = fixed_vars[b, i]
            for j in range(i + 1, n):
                w = G_func(nodes[i], nodes[j])
                if w != 0.0 and bi != fixed_vars[b, j]:
                    cut += w
        vals[b] = cut
    return vals


@njit(parallel=True, cache=True)
def _eval_leaves_energy_streaming(G_func, nodes, fixed_vars, n, batch_size):
    vals = np.empty(batch_size)
    for b in prange(batch_size):
        energy = 0.0
        for i in range(n):
            bi = fixed_vars[b, i]
            for j in range(i + 1, n):
                val = G_func(nodes[i], nodes[j])
                energy += -val if bi == fixed_vars[b, j] else val
        vals[b] = energy
    return vals


@njit(cache=True)
def _influence_scores_streaming(G_func, nodes, fixed_row, n):
    scores = np.full(n, -1.0)
    for i in range(n):
        if fixed_row[i] >= 0:
            continue
        s = 0.0
        for j in range(n):
            if i == j or fixed_row[j] >= 0:
                continue
            ii = i if i < j else j
            jj = j if i < j else i
            s += abs(G_func(nodes[ii], nodes[jj]))
        scores[i] = s
    return scores


# ---------------------------------------------------------------------------
# B&B loop
# ---------------------------------------------------------------------------

def _branch_and_bound_streaming(G_func, nodes, warm_theta, warm_energy, n,
                                 is_spin_glass, verbose=True, time_limit=None):
    best_bits = warm_theta.copy()
    best_value = warm_energy

    if verbose:
        print(f"Warm-start incumbent: {best_value:.6f}")

    root = np.full(n, -1, dtype=np.int8)
    stack = [root]

    t_start = time.monotonic()
    nodes_explored = 0
    nodes_pruned = 0
    batch_cap = os.cpu_count() * 4

    while stack:
        if time_limit is not None and (time.monotonic() - t_start) > time_limit:
            if verbose:
                print(f"Time limit reached. "
                      f"Nodes: {nodes_explored}, Pruned: {nodes_pruned}")
            return best_bits, best_value, False

        batch_size = min(batch_cap, len(stack))
        batch_nodes = [stack.pop() for _ in range(batch_size)]
        batch_arr = np.array(batch_nodes, dtype=np.int8)

        nodes_explored += batch_size

        ubs = _upper_bound_batch_streaming(G_func, nodes, batch_arr, n, batch_size)

        leaves = []
        interior = []
        for k in range(batch_size):
            if ubs[k] <= best_value + 1e-9:
                nodes_pruned += 1
                continue
            if int(np.sum(batch_arr[k] < 0)) == 0:
                leaves.append(k)
            else:
                interior.append(k)

        if leaves:
            leaf_arr = batch_arr[np.array(leaves, dtype=np.int64)]
            leaf_vals = (
                _eval_leaves_energy_streaming(G_func, nodes, leaf_arr, n, len(leaves))
                if is_spin_glass
                else _eval_leaves_cut_streaming(G_func, nodes, leaf_arr, n, len(leaves))
            )
            best_leaf = int(np.argmax(leaf_vals))
            if leaf_vals[best_leaf] > best_value:
                best_value = float(leaf_vals[best_leaf])
                best_bits = (leaf_arr[best_leaf] >= 1).copy()
                if verbose:
                    print(f"  New incumbent: {best_value:.6f}"
                          f"  (nodes: {nodes_explored})")

        for k in interior:
            row = batch_arr[k]
            scores = _influence_scores_streaming(G_func, nodes, row, n)
            branch_var = int(np.argmax(scores))
            warm_val = int(best_bits[branch_var])
            for val in [warm_val, 1 - warm_val]:
                child = row.copy()
                child[branch_var] = np.int8(val)
                stack.append(child)

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
    then certify via branch and bound with parallel Numba kernels.

    Accepts the same G_func + nodes input as spin_glass_solver_streaming and
    the same warm-start formats (str, int, list, or None).

    Parameters
    ----------
    G_func : callable(u, v) -> float
        Edge weight function (Numba-callable). Returns 0.0 for absent edges.
    nodes : array-like
        Ordered sequence of node identifiers passed to G_func.
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

    bitstring = ""
    cut_value = None
    energy_value = None
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
        bitstring, cut_value, _, energy_value = spin_glass_solver_streaming(
            G_func, nodes, is_spin_glass=is_spin_glass, **kwargs
        )
        if verbose:
            print(f"Heuristic value: {cut_value:.6f}  ({time.monotonic()-t0:.3f}s)")

    best_theta = np.array([b == "1" for b in list(bitstring)], dtype=np.bool_)
    if is_spin_glass:
        max_energy = compute_energy_streaming(best_theta, G_func, nodes, n_qubits) if energy_value is None else energy_value
    elif cut_value is None:
        max_energy = compute_cut_streaming(best_theta, G_func, nodes, n_qubits)
    else:
        max_energy = cut_value

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
