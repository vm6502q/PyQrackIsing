from .maxcut_tfim import maxcut_tfim
from .spin_glass_solver_sparse import spin_glass_solver_sparse

import networkx as nx


def spin_glass_solver_hybrid(
    G,
    quality=None,
    shots=None,
    is_maxcut_gpu=True,
    is_spin_glass=True,
    anneal_t=None,
    anneal_h=None,
    repulsion_base=None,
    is_log=False,
    gray_iterations=None,
    gray_seed_multiple=None,
    bp_scale=None,
    bp_damping=0.5,
):
    if not isinstance(G, nx.Graph):
        raise TypeError("G graph must be a networkx.Graph for maxcut_hybrid!")

    guess, _, _ = maxcut_tfim(
        G,
        quality=quality,
        shots=shots,
        repulsion_base=repulsion_base,
        is_maxcut_gpu=is_maxcut_gpu,
        is_spin_glass=is_spin_glass,
        anneal_t=anneal_t,
        anneal_h=anneal_h
    )

    return spin_glass_solver_sparse(
        G,
        best_guess=guess,
        is_maxcut_gpu=is_maxcut_gpu,
        is_spin_glass=is_spin_glass,
        is_log=is_log,
        gray_iterations=gray_iterations,
        gray_seed_multiple=gray_seed_multiple,
        bp_scale=bp_scale,
        bp_damping=bp_damping
    )
