import tfim_sampler


def maxcut_tfim(G_edges, n_qubits):
    return tfim_sampler._maxcut_tfim(G_edges, n_qubits)
