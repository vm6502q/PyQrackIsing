from .maxcut_tfim_util import probability_by_hamming_weight, opencl_context
from numba import njit


epsilon = opencl_context.epsilon


@njit
def tfim_square_magnetization(J=-1.0, h=2.0, z=4, theta=0.174532925199432957, t=5, n_qubits=56):
    if abs(t) <= epsilon:
        return np.cos(theta) ** 2

    bias = probability_by_hamming_weight(J, h, z, theta, t, n_qubits + 1)
    bias /= bias.sum()
    square_magnetization = 0.0
    nqs = int(n_qubits)
    nqd = float(n_qubits)
    for q in range(n_qubits + 1):
        mag = (nqs - 2 * q) / nqd
        square_magnetization += bias[q] * mag * mag
    return square_magnetization
