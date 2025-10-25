from .maxcut_tfim_util import probability_by_hamming_weight, sample_mag
from numba import njit
import numpy as np
import random


def get_otocs_hamming_distribution(J=-1.0, h=2.0, z=4, theta=0.174532925199432957, t=5, n_qubits=56, cycles=1, pauli_string = 'X' + 'I' * 55):
    pauli_string = list(pauli_string)
    if len(pauli_string) != n_qubits:
        raise ValueError("OTOCS pauli_string must be same length as n_qubits! (Use 'I' for qubits that aren't changed.)")

    fwd = probability_by_hamming_weight(J, h, z, theta, t, n_qubits + 1, False)
    rev = probability_by_hamming_weight(-J, -h, z, -theta, t, n_qubits + 1, False)
    diff_theta = rev - fwd

    phi = theta - np.pi / 2
    fwd = probability_by_hamming_weight(J, h, z, phi, t, n_qubits + 1, False)
    rev = probability_by_hamming_weight(-J, -h, z, -phi, t, n_qubits + 1, False)
    diff_phi = rev - fwd

    diff_theta *= cycles
    diff_phi *= cycles
    diff_lam = 0.5 * (diff_theta + diff_phi)

    n_bias = n_qubits + 1
    diff_z = np.zeros(n_bias, dtype=np.float64)
    diff_x = np.zeros(n_bias, dtype=np.float64)
    diff_y = np.zeros(n_bias, dtype=np.float64)
    for b in pauli_string:
        match b:
            case 'X':
                diff_z += diff_theta
                diff_x += diff_lam
                diff_y += diff_phi
            case 'Z':
                diff_z += diff_phi
                diff_x += diff_theta
                diff_y += diff_lam
            case 'Y':
                diff_z += diff_lam
                diff_x += diff_phi
                diff_y += diff_theta

    diff_z -= diff_z.min()
    diff_x -= diff_x.min()
    diff_y -= diff_y.min()

    diff_z[0] += n_bias
    diff_x[0] += n_bias
    diff_y[0] += n_bias

    diff_z /= diff_z.sum()
    diff_x /= diff_x.sum()
    diff_y /= diff_y.sum()

    return { 'X': diff_x, 'Y': diff_y, 'Z': diff_z }


@njit
def fix_cdf(hamming_prob):
    tot_prob = 0.0
    n_bias = len(hamming_prob)
    cum_prob = np.empty(n_bias, dtype=np.float64)
    for i in range(n_bias):
        tot_prob += hamming_prob[i]
        cum_prob[i] = tot_prob
    cum_prob[-1] = 2.0

    return cum_prob


@njit
def take_all(b, basis, sample):
    for i in range(len(basis)):
        if basis[i] == b:
            sample |= (1 << i)

    return sample


def take_sample(b, basis, sample, m):
    indices = []
    for i in range(len(basis)):
        if basis[i] == b:
            indices.append(i)
    indices = random.sample(indices, m)
    for i in indices:
        sample |= 1 << i

    return sample


def generate_otocs_samples(J=-1.0, h=2.0, z=4, theta=0.174532925199432957, t=5, n_qubits=56, cycles=1, pauli_string = 'X' + 'I' * 55, shots=100, measurement_basis='Z' * 56):
    pauli_string = list(pauli_string)
    if len(pauli_string) != n_qubits:
        raise ValueError("OTOCS pauli_string must be same length as n_qubits! (Use 'I' for qubits that aren't changed.)")

    measurement_basis = list(measurement_basis)
    if len(measurement_basis) != n_qubits:
        raise ValueError("OTOCS measurement_basis must be same length as measurement_basis! (Use 'I' for excluded qubits.)")

    basis_x, basis_y, basis_z = [], [], []
    for b in pauli_string:
        if b == 'Z':
            basis_z.append('X')
            basis_y.append('Z')
            basis_x.append('I')
        elif b == 'X':
            basis_z.append('Z')
            basis_y.append('I')
            basis_x.append('X')
        elif b == 'Y':
            basis_z.append('I')
            basis_y.append('I')
            basis_x.append('X')
        else:
            basis_z.append('I')
            basis_y.append('X')
            basis_x.append('Z')

    bases = { 'X': basis_x, 'Y': basis_y, 'Z': basis_z }
    thresholds = { key: fix_cdf(value) for key, value in get_otocs_hamming_distribution(J, h, z, theta, t, n_qubits, cycles, pauli_string).items() }

    samples_3_axis = {}
    for key, value in thresholds.items():
        basis = bases[key]

    samples = []
    for _ in range(shots):
        sample_3_axis = { 'X': 0, 'Y': 0, 'Z': 0 }
        for key, value in thresholds.items():
            basis = bases[key]

            # First dimension: Hamming weight
            m = sample_mag(value)
            if m >= n_qubits:
                sample_3_axis[key] = (1 << n_qubits) - 1
                continue

            # Second dimension: permutation within Hamming weight
            z_count = basis.count('Z')
            if z_count > m:
                sample_3_axis[key] = take_sample('Z', basis, sample_3_axis[key], m)
                continue
            m -= z_count
            sample_3_axis[key] = take_all('Z', basis, sample_3_axis[key])
            if m == 0:
                continue

            i_count = basis.count('I')
            if i_count > m:
                sample_3_axis[key] = take_sample('I', basis, sample_3_axis[key], m)
                continue
            m -= i_count
            sample_3_axis[key] = take_all('I', basis, sample_3_axis[key])
            if m == 0:
                continue

            sample_3_axis[key] = take_sample('X', basis, sample_3_axis[key], m)

        sample = 0
        j = 0
        for i in range(n_qubits):
            base = measurement_basis[i]
            if base not in ['X', 'Y', 'Z']:
                continue
            if (sample_3_axis[base] >> i) & 1:
                sample |= 1 << j
            j += 1

        samples.append(sample)

    return samples
