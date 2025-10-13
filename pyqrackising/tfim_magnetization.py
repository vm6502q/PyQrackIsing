from .maxcut_tfim_util import probability_by_hamming_weight
from numba import njit


@njit
def tfim_magnetization(J=-1.0, h=2.0, z=4, theta=0.174532925199432957, t=5, n_qubits=56):
    bias = probability_by_hamming_weight(J, h, z, theta, t, n_qubits + 1)
    bias /= bias.sum()
    magnetization = 0.0
    nqs = int(n_qubits)
    nqd = float(n_qubits)
    for q in range(n_qubits + 1):
        mag = (nqs - 2 * q) / nqd
        magnetization += bias[q] * mag
    return magnetization
