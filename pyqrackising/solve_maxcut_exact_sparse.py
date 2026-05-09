# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Daniel Strano and the Qrack contributors
#
# Initial draft produced by (Anthropic) Claude (Sonnet 4.6).
#
# bnb_exact_sparse.py — warm-start branch-and-bound exact MAXCUT solver (sparse)
#
# Same parallelism strategy as bnb_exact.py; inner loops work from CSR arrays.

import time
import os
import networkx as nx
import numpy as np
from numba import njit, prange
from .maxcut_tfim_util import (
    compute_cut_sparse,
    compute_energy_sparse,
    get_cut,
    heuristic_threshold_sparse,
    int_to_bitstring,
    opencl_context,
    to_scipy_sparse_upper_triangular,
)

dtype = opencl_context.dtype


# ---------------------------------------------------------------------------
# Parallel kernels (sparse CSR)
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _upper_bound_batch_sparse(G_data, G_rows, G_cols, fixed_vars, n, batch_size):
    """
    Parallel upper bound for a batch of B&B nodes (sparse CSR).
    fixed_vars : (batch_size, n) int8, -1=free, 0/1=fixed.
    """
    ub = np.empty(batch_size)
    for b in prange(batch_size):
        total = 0.0
        for i in range(n):
            fi = fixed_vars[b, i]
            for r in range(G_rows[i], G_rows[i + 1]):
                j = G_cols[r]
                if j <= i:
                    continue
                w = G_data[r]
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
def _eval_leaves_cut_sparse(G_data, G_rows, G_cols, fixed_vars, n, batch_size):
    vals = np.empty(batch_size)
    for b in prange(batch_size):
        cut = 0.0
        for i in range(n):
            bi = fixed_vars[b, i]
            for r in range(G_rows[i], G_rows[i + 1]):
                j = G_cols[r]
                if j > i and G_data[r] != 0.0 and bi != fixed_vars[b, j]:
                    cut += G_data[r]
        vals[b] = cut
    return vals


@njit(parallel=True, cache=True)
def _eval_leaves_energy_sparse(G_data, G_rows, G_cols, fixed_vars, n, batch_size):
    vals = np.empty(batch_size)
    for b in prange(batch_size):
        energy = 0.0
        for i in range(n):
            bi = fixed_vars[b, i]
            for r in range(G_rows[i], G_rows[i + 1]):
                j = G_cols[r]
                val = G_data[r]
                energy += -val if bi == fixed_vars[b, j] else val
        vals[b] = energy
    return vals


@njit(cache=True)
def _influence_scores_sparse(G_data, G_rows, G_cols, fixed_row, n):
    scores = np.full(n, -1.0)
    free_set = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        if fixed_row[i] < 0:
            free_set[i] = True
    for i in range(n):
        if not free_set[i]:
            continue
        s = 0.0
        for r in range(G_rows[i], G_rows[i + 1]):
            j = G_cols[r]
            if free_set[j]:
                s += abs(G_data[r])
        # reverse edges (j < i stored in row j)
        for j in range(i):
            if not free_set[j]:
                continue
            for r in range(G_rows[j], G_rows[j + 1]):
                if G_cols[r] == i:
                    s += abs(G_data[r])
                    break
        scores[i] = s
    return scores


# ---------------------------------------------------------------------------
# B&B loop
# ---------------------------------------------------------------------------

def _branch_and_bound_sparse(G_data, G_rows, G_cols, warm_theta, warm_energy,
                              n, is_spin_glass, verbose=True, time_limit=None):
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

        ubs = _upper_bound_batch_sparse(G_data, G_rows, G_cols,
                                        batch_arr, n, batch_size)

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
                _eval_leaves_energy_sparse(G_data, G_rows, G_cols,
                                           leaf_arr, n, len(leaves))
                if is_spin_glass
                else _eval_leaves_cut_sparse(G_data, G_rows, G_cols,
                                             leaf_arr, n, len(leaves))
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
            scores = _influence_scores_sparse(G_data, G_rows, G_cols, row, n)
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
    Exact MAXCUT/spin-glass solver: warm-start from spin_glass_solver_sparse
    then certify via branch and bound with parallel Numba kernels.

    Accepts the same G input as spin_glass_solver_sparse (NetworkX graph or
    scipy CSR matrix) and the same warm-start formats (str, int, list, None).

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

    best_theta = np.array([b == "1" for b in list(bitstring)], dtype=np.bool_)
    if is_spin_glass:
        max_energy = compute_energy_sparse(best_theta, G_m.data, G_m.indptr, G_m.indices, n_qubits)
    elif cut_value is None:
        max_energy = compute_cut_sparse(best_theta, G_m.data, G_m.indptr, G_m.indices, n_qubits)
    else:
        max_energy = cut_value

    if verbose:
        print("Starting branch and bound...")

    best_theta, max_energy, certified = _branch_and_bound_sparse(
        G_data, G_rows, G_cols, best_theta, max_energy, n_qubits, is_spin_glass,
        verbose=verbose, time_limit=time_limit,
    )

    bitstring, l, r = get_cut(best_theta, nodes, n_qubits)
    if is_spin_glass:
        cut_value = compute_cut_sparse(best_theta, G_data, G_rows, G_cols, n_qubits)
        min_energy = -max_energy
    else:
        cut_value = max_energy
        min_energy = compute_energy_sparse(best_theta, G_data, G_rows, G_cols, n_qubits)

    return bitstring, float(cut_value), (l, r), float(min_energy), certified
