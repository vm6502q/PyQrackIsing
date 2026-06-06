"""
kawasaki_chain_sampler.py
=========================
Kawasaki-Metropolis chain samplers for dense, sparse, and streaming
MaxCut solver variants.

Memory: O(n) per thread regardless of shot count.

Proposal biasing
----------------
Swap candidate (i, j) is chosen using repulsion_base ** (-G[i,j]) weighting,
matching the existing local_repulsion_choice convention. This naturally
handles sign: positive edges repel (downweight), negative edges attract
(upweight), magnitude scales continuously. No abs() discontinuity.

Authors: Claude (Anthropic) and D. Strano
"""

import numpy as np
from numba import njit, prange

from .maxcut_tfim_util import sample_mag, compute_cut_diff_between, compute_energy
from .maxcut_tfim_util import sample_mag, compute_cut_sparse, compute_energy_sparse
from .maxcut_tfim_util import sample_mag, compute_cut_diff_between_streaming, compute_energy_streaming


@njit(cache=True)
def _lcg(rng):
    return (rng * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)) & np.uint64(0xFFFFFFFFFFFFFFFF)


@njit(cache=True)
def _init_chain(n, m, rng):
    state = np.zeros(n, dtype=np.bool_)
    perm  = np.arange(n, dtype=np.int64)
    for k in range(n - 1, 0, -1):
        rng = _lcg(rng)
        idx = int(rng % np.uint64(k + 1))
        perm[k], perm[idx] = perm[idx], perm[k]
    for k in range(m):
        state[perm[k]] = True
    ones  = np.empty(m,     dtype=np.int64)
    zeros = np.empty(n - m, dtype=np.int64)
    oi = 0; zi = 0
    for k in range(n):
        if state[k]: ones[oi]  = k; oi += 1
        else:         zeros[zi] = k; zi += 1
    return state, ones, zeros, rng


@njit(cache=True)
def _do_swap(state, ones, zeros, m, n, i, j):
    state[i] = False; state[j] = True
    for k in range(m):
        if ones[k] == i: ones[k] = j; break
    for k in range(n - m):
        if zeros[k] == j: zeros[k] = i; break


# ── Biased proposals ───────────────────────────────────────────────────────────

@njit(cache=True)
def _biased_proposal_dense(state, G_m, n, ones, zeros, m, repulsion_base, rng):
    """
    Propose swap (i, j) with i sampled from ones and j from zeros,
    weighted by repulsion_base ** (-G_m[i, j]).
    Positive edges: repel (downweight). Negative edges: attract (upweight).
    Falls back to uniform if all weights collapse to zero.
    """
    # Score each one-bit by its total weight to zero-bits
    i_weights = np.zeros(m, dtype=np.float64)
    for oi in range(m):
        i_node = ones[oi]
        w = 0.0
        for zi in range(n - m):
            j_node = zeros[zi]
            val = G_m[i_node, j_node]
            if val != 0.0:
                w += repulsion_base ** (-val)
        i_weights[oi] = w

    total_i = 0.0
    for oi in range(m): total_i += i_weights[oi]

    if total_i <= 0.0:
        rng = _lcg(rng); i_node = ones[int(rng % np.uint64(m))]
        rng = _lcg(rng); j_node = zeros[int(rng % np.uint64(n - m))]
        return i_node, j_node, rng

    rng = _lcg(rng)
    r = float(rng) / float(np.uint64(0xFFFFFFFFFFFFFFFF)) * total_i
    cumsum = 0.0; chosen_oi = 0
    for oi in range(m):
        cumsum += i_weights[oi]
        if cumsum >= r: chosen_oi = oi; break
    i_node = ones[chosen_oi]

    j_weights = np.zeros(n - m, dtype=np.float64)
    total_j = 0.0
    for zi in range(n - m):
        j_node = zeros[zi]
        val = G_m[i_node, j_node]
        w = repulsion_base ** (-val) if val != 0.0 else 1.0
        j_weights[zi] = w
        total_j += w

    rng = _lcg(rng)
    r = float(rng) / float(np.uint64(0xFFFFFFFFFFFFFFFF)) * total_j
    cumsum = 0.0; chosen_zi = 0
    for zi in range(n - m):
        cumsum += j_weights[zi]
        if cumsum >= r: chosen_zi = zi; break
    j_node = zeros[chosen_zi]

    return i_node, j_node, rng


@njit(cache=True)
def _biased_proposal_sparse(state, G_data, G_rows, G_cols, n, ones, zeros, m, repulsion_base, rng):
    """
    Propose swap (i, j) weighted by repulsion_base ** (-G[i,j]).
    Only stored edges are weighted; unconnected pairs get weight 1.0 (uniform).
    """
    i_weights = np.zeros(m, dtype=np.float64)
    for oi in range(m):
        i_node = ones[oi]
        w = float(n - m)   # base weight: unconnected pairs contribute 1.0 each
        for col in range(G_rows[i_node], G_rows[i_node + 1]):
            j_node = G_cols[col]
            if state[j_node]: continue   # one-bit, not a swap candidate
            val = G_data[col]
            w += repulsion_base ** (-val) - 1.0   # delta from baseline
        i_weights[oi] = max(w, 0.0)

    total_i = 0.0
    for oi in range(m): total_i += i_weights[oi]

    if total_i <= 0.0:
        rng = _lcg(rng); i_node = ones[int(rng % np.uint64(m))]
        rng = _lcg(rng); j_node = zeros[int(rng % np.uint64(n - m))]
        return i_node, j_node, rng

    rng = _lcg(rng)
    r = float(rng) / float(np.uint64(0xFFFFFFFFFFFFFFFF)) * total_i
    cumsum = 0.0; chosen_oi = 0
    for oi in range(m):
        cumsum += i_weights[oi]
        if cumsum >= r: chosen_oi = oi; break
    i_node = ones[chosen_oi]

    # Sample j: stored edges get repulsion_base**(-val), rest get 1.0
    j_weights = np.ones(n - m, dtype=np.float64)
    for col in range(G_rows[i_node], G_rows[i_node + 1]):
        j_node = G_cols[col]
        if state[j_node]: continue
        val = G_data[col]
        for zi in range(n - m):
            if zeros[zi] == j_node:
                j_weights[zi] = repulsion_base ** (-val)
                break

    total_j = 0.0
    for zi in range(n - m): total_j += j_weights[zi]

    rng = _lcg(rng)
    r = float(rng) / float(np.uint64(0xFFFFFFFFFFFFFFFF)) * total_j
    cumsum = 0.0; chosen_zi = 0
    for zi in range(n - m):
        cumsum += j_weights[zi]
        if cumsum >= r: chosen_zi = zi; break
    j_node = zeros[chosen_zi]

    return i_node, j_node, rng


@njit(cache=True)
def _biased_proposal_streaming(state, G_func, nodes, n, ones, zeros, m, repulsion_base, rng):
    i_weights = np.zeros(m, dtype=np.float64)
    for oi in range(m):
        i_node = ones[oi]
        w = 0.0
        for zi in range(n - m):
            j_node = zeros[zi]
            val = G_func(nodes[i_node], nodes[j_node])
            w += repulsion_base ** (-val) if val != 0.0 else 1.0
        i_weights[oi] = w

    total_i = 0.0
    for oi in range(m): total_i += i_weights[oi]

    if total_i <= 0.0:
        rng = _lcg(rng); i_node = ones[int(rng % np.uint64(m))]
        rng = _lcg(rng); j_node = zeros[int(rng % np.uint64(n - m))]
        return i_node, j_node, rng

    rng = _lcg(rng)
    r = float(rng) / float(np.uint64(0xFFFFFFFFFFFFFFFF)) * total_i
    cumsum = 0.0; chosen_oi = 0
    for oi in range(m):
        cumsum += i_weights[oi]
        if cumsum >= r: chosen_oi = oi; break
    i_node = ones[chosen_oi]

    j_weights = np.zeros(n - m, dtype=np.float64)
    total_j = 0.0
    for zi in range(n - m):
        j_node = zeros[zi]
        val = G_func(nodes[i_node], nodes[j_node])
        w = repulsion_base ** (-val) if val != 0.0 else 1.0
        j_weights[zi] = w; total_j += w

    rng = _lcg(rng)
    r = float(rng) / float(np.uint64(0xFFFFFFFFFFFFFFFF)) * total_j
    cumsum = 0.0; chosen_zi = 0
    for zi in range(n - m):
        cumsum += j_weights[zi]
        if cumsum >= r: chosen_zi = zi; break
    j_node = zeros[chosen_zi]

    return i_node, j_node, rng


# ── Delta functions ────────────────────────────────────────────────────────────

@njit(cache=True)
def _kawasaki_delta_dense(state, G_m, n, i, j):
    si = state[i]; sj = state[j]
    delta = 0.0
    for v in range(n):
        if v == i or v == j: continue
        sv = state[v]
        vi = G_m[i, v]
        if vi != 0.0 and (si != sv) != (sj != sv):
            delta += vi if (sj != sv) else -vi
        vj = G_m[j, v]
        if vj != 0.0 and (sj != sv) != (si != sv):
            delta += vj if (si != sv) else -vj
    return delta


@njit(cache=True)
def _kawasaki_delta_sparse(state, G_data, G_rows, G_cols, n, i, j):
    si = state[i]; sj = state[j]
    delta = 0.0
    for col in range(G_rows[i], G_rows[i + 1]):
        v = G_cols[col]
        if v == j: continue
        sv = state[v]; val = G_data[col]
        if (si != sv) != (sj != sv):
            delta += val if (sj != sv) else -val
    for col in range(G_rows[j], G_rows[j + 1]):
        v = G_cols[col]
        if v == i: continue
        sv = state[v]; val = G_data[col]
        if (sj != sv) != (si != sv):
            delta += val if (si != sv) else -val
    for v in range(i):
        for col in range(G_rows[v], G_rows[v + 1]):
            if G_cols[col] == i:
                sv = state[v]; val = G_data[col]
                if (si != sv) != (sj != sv):
                    delta += val if (sj != sv) else -val
                break
    for v in range(j):
        if v == i: continue
        for col in range(G_rows[v], G_rows[v + 1]):
            if G_cols[col] == j:
                sv = state[v]; val = G_data[col]
                if (sj != sv) != (si != sv):
                    delta += val if (si != sv) else -val
                break
    return delta


@njit(cache=True)
def _kawasaki_delta_streaming(state, G_func, nodes, n, i, j):
    si = state[i]; sj = state[j]
    delta = 0.0
    for v in range(n):
        if v == i or v == j: continue
        sv = state[v]
        vi = G_func(nodes[i], nodes[v])
        if vi != 0.0 and (si != sv) != (sj != sv):
            delta += vi if (sj != sv) else -vi
        vj = G_func(nodes[j], nodes[v])
        if vj != 0.0 and (sj != sv) != (si != sv):
            delta += vj if (si != sv) else -vj
    return delta


# ── Samplers ───────────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True)
def sample_measurement_kawasaki_dense(G_m, shots, thread_count, thresholds, repulsion_base, thinning, is_spin_glass):
    n = len(G_m)
    shot_segment = (max(1, shots >> 1) + thread_count - 1) // thread_count

    solutions = np.zeros((thread_count, n), dtype=np.bool_)
    if is_spin_glass:
        energies = np.array([compute_energy(solutions[i], G_m, n)
                             for i in range(thread_count)], dtype=np.float64)
    else:
        energies = np.zeros(thread_count, dtype=np.float64)

    best_solution = solutions[0].copy()
    best_energy   = energies[0]

    improved = True
    while improved:
        improved = False
        for t in prange(thread_count):
            m = sample_mag(thresholds)
            rng = _lcg(np.uint64(t * 2654435761 + 1))
            state, ones, zeros, rng = _init_chain(n, m, rng)

            for _ in range(10 * n):
                i, j, rng = _biased_proposal_dense(state, G_m, n, ones, zeros, m, repulsion_base, rng)
                if _kawasaki_delta_dense(state, G_m, n, i, j) > 0.0:
                    _do_swap(state, ones, zeros, m, n, i, j)

            for _ in range(shot_segment):
                for _s in range(thinning):
                    i, j, rng = _biased_proposal_dense(state, G_m, n, ones, zeros, m, repulsion_base, rng)
                    if _kawasaki_delta_dense(state, G_m, n, i, j) > 0.0:
                        _do_swap(state, ones, zeros, m, n, i, j)
                energy = compute_cut_diff_between(solutions[t], state, G_m, n)
                if energy > 0.0:
                    energies[t] += energy
                    for k in range(n): solutions[t][k] = state[k]

        best_index = np.argmax(energies)
        energy = energies[best_index]
        if is_spin_glass: energy *= 2.0
        if energy > best_energy:
            best_energy = energy
            best_solution = solutions[best_index].copy()
            improved = True

    return best_solution, best_energy


@njit(parallel=True, cache=True)
def sample_measurement_kawasaki_sparse(G_data, G_rows, G_cols, shots, thread_count, thresholds, repulsion_base, thinning, is_spin_glass):
    n = G_rows.shape[0] - 1
    shot_segment = (max(1, shots >> 1) + thread_count - 1) // thread_count

    solutions = np.zeros((thread_count, n), dtype=np.bool_)
    energies  = np.full(thread_count, np.finfo(np.float32).min, dtype=np.float32)
    best_solution = solutions[0].copy()
    best_energy   = -np.inf

    improved = True
    while improved:
        improved = False
        for t in prange(thread_count):
            m = sample_mag(thresholds)
            rng = _lcg(np.uint64(t * 2654435761 + 1))
            state, ones, zeros, rng = _init_chain(n, m, rng)

            for _ in range(10 * n):
                i, j, rng = _biased_proposal_sparse(state, G_data, G_rows, G_cols, n, ones, zeros, m, repulsion_base, rng)
                if _kawasaki_delta_sparse(state, G_data, G_rows, G_cols, n, i, j) > 0.0:
                    _do_swap(state, ones, zeros, m, n, i, j)

            for _ in range(shot_segment):
                for _s in range(thinning):
                    i, j, rng = _biased_proposal_sparse(state, G_data, G_rows, G_cols, n, ones, zeros, m, repulsion_base, rng)
                    if _kawasaki_delta_sparse(state, G_data, G_rows, G_cols, n, i, j) > 0.0:
                        _do_swap(state, ones, zeros, m, n, i, j)
                energy = compute_cut_sparse(state, G_data, G_rows, G_cols, n) if not is_spin_glass \
                         else compute_energy_sparse(state, G_data, G_rows, G_cols, n)
                if energy > energies[t]:
                    energies[t] = energy
                    for k in range(n): solutions[t][k] = state[k]

        best_index = np.argmax(energies)
        energy = energies[best_index]
        if energy > best_energy:
            best_energy = energy
            best_solution = solutions[best_index].copy()
            improved = True

    return best_solution, best_energy


@njit(parallel=True, cache=True)
def sample_measurement_kawasaki_streaming(G_func, nodes, shots, thread_count, thresholds, n, repulsion_base, thinning, is_spin_glass):
    shot_segment = (max(1, shots >> 1) + thread_count - 1) // thread_count

    solutions = np.zeros((thread_count, n), dtype=np.bool_)
    if is_spin_glass:
        energies = np.array([compute_energy_streaming(solutions[i], G_func, nodes, n)
                             for i in range(thread_count)], dtype=np.float64)
    else:
        energies = np.zeros(thread_count, dtype=np.float64)

    best_solution = solutions[0].copy()
    best_energy   = energies[0]

    improved = True
    while improved:
        improved = False
        for t in prange(thread_count):
            m = sample_mag(thresholds)
            rng = _lcg(np.uint64(t * 2654435761 + 1))
            state, ones, zeros, rng = _init_chain(n, m, rng)

            for _ in range(10 * n):
                i, j, rng = _biased_proposal_streaming(state, G_func, nodes, n, ones, zeros, m, repulsion_base, rng)
                if _kawasaki_delta_streaming(state, G_func, nodes, n, i, j) > 0.0:
                    _do_swap(state, ones, zeros, m, n, i, j)

            for _ in range(shot_segment):
                for _s in range(thinning):
                    i, j, rng = _biased_proposal_streaming(state, G_func, nodes, n, ones, zeros, m, repulsion_base, rng)
                    if _kawasaki_delta_streaming(state, G_func, nodes, n, i, j) > 0.0:
                        _do_swap(state, ones, zeros, m, n, i, j)
                energy = compute_cut_diff_between_streaming(solutions[t], state, G_func, nodes, n)
                if energy > 0.0:
                    energies[t] += energy
                    for k in range(n): solutions[t][k] = state[k]

        best_index = np.argmax(energies)
        energy = energies[best_index]
        if is_spin_glass: energy *= 2.0
        if energy > best_energy:
            best_energy = energy
            best_solution = solutions[best_index].copy()
            improved = True

    return best_solution, best_energy
