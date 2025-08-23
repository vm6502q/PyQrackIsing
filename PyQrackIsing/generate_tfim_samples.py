import tfim_sampler


def generate_tfim_samples(
    J=-1.0, h=2.0, z=4, theta=0.174532925199432957, t=5, n_qubits=56, shots=100
):
    return [int(s) for s in tfim_sampler._generate_tfim_samples(J, h, z, theta, t, n_qubits, shots)]
