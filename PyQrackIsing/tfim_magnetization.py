import tfim_sampler


def tfim_magnetization(J=-1.0, h=2.0, z=4, theta=0.174532925199432957, t=5, n_qubits=56):
    return tfim_sampler._tfim_magnetization(J, h, z, theta, t, n_qubits)
