import math
import os
import sys
import statistics
import time

from collections import Counter

import numpy as np

from scipy.stats import distributions as dists

import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate
from qiskit.compiler import transpile
from qiskit_aer.backends import AerSimulator
from qiskit.quantum_info import Statevector

from pyqrack import QrackSimulator
from pyqrackising import generate_fermi_hubbard_samples


def factor_width(width, is_transpose=False):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (col_len, row_len) if is_transpose else (row_len, col_len)


def index(i, j, n_cols):
    return i * n_cols + j


def zz_rotation(qc, q1, q2, theta):
    # Implements exp(-i theta Z⊗Z)
    qc.cx(q1, q2)
    qc.rz(2 * theta, q2)
    qc.cx(q1, q2)

def first_order_tfim(qc, n_rows, n_cols, J, h, dt):
    theta_zz = J * dt

    # ---- Horizontal even bonds ----
    for i in range(n_rows):
        for j in range(0, n_cols - 1, 2):
            q1 = index(i, j, n_cols)
            q2 = index(i, j + 1, n_cols)
            zz_rotation(qc, q1, q2, theta_zz)

    # ---- Horizontal odd bonds ----
    for i in range(n_rows):
        for j in range(1, n_cols - 1, 2):
            q1 = index(i, j, n_cols)
            q2 = index(i, j + 1, n_cols)
            zz_rotation(qc, q1, q2, theta_zz)

    # ---- Vertical even bonds ----
    for j in range(n_cols):
        for i in range(0, n_rows - 1, 2):
            q1 = index(i, j, n_cols)
            q2 = index(i + 1, j, n_cols)
            zz_rotation(qc, q1, q2, theta_zz)

    # ---- Vertical odd bonds ----
    for j in range(n_cols):
        for i in range(1, n_rows - 1, 2):
            q1 = index(i, j, n_cols)
            q2 = index(i + 1, j, n_cols)
            zz_rotation(qc, q1, q2, theta_zz)

def brick_wall_tfim_step(n_rows, n_cols, J, h, dt):
    """
    Single first-order Trotter step for 2D TFIM
    using brick-wall decomposition.
    """
    n_qubits = n_rows * n_cols
    qc = QuantumCircuit(n_qubits)
    theta_x = h * dt

    first_order_tfim(qc, n_rows, n_cols, J, h, dt / 2)

    # ---- Transverse field ----
    for q in range(n_qubits):
        qc.rx(2 * theta_x, q)

    first_order_tfim(qc, n_rows, n_cols, J, h, dt / 2)

    return qc


def init_beta(n_qubits):
    n_bias = n_qubits + 1
    thresholds = np.empty(n_bias, dtype=np.float64)
    normalizer = 0
    for q in range(n_qubits >> 1):
        normalizer += math.comb(n_qubits, q) << 1
    if n_qubits & 1:
        normalizer += math.comb(n_qubits, n_qubits >> 1)
    p = 1
    for q in range(n_qubits >> 1):
        val = p / normalizer
        thresholds[q] = val
        thresholds[n_bias - (q + 1)] = val
        p = math.comb(n_qubits, q + 1)
    if n_qubits & 1:
        thresholds[n_qubits >> 1] = p / normalizer

    return thresholds


def calc_stats(ideal_probs, pqi_probs, shots):
    # For QV, we compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    n_pow = len(ideal_probs)
    n = int(round(math.log2(n_pow)))
    threshold = statistics.median(ideal_probs)
    u_u = 1 / n_pow
    diff_sqr = 0
    noise = 0
    numer = 0
    denom = 0
    sum_hog_prob = 0
    ideal_sqr_mag = 0
    exp_sqr_mag = 0
    for i in range(n_pow):
        exp = pqi_probs.get(i, 0)
        ideal = ideal_probs[i]

        # L2 distance
        diff_sqr += (ideal - exp) ** 2
        noise += exp * (1 - exp) / shots

        # XEB / EPLG
        denom += (ideal - u_u) ** 2
        numer += (ideal - u_u) * (exp - u_u)

        # QV / HOG
        if ideal > threshold:
            sum_hog_prob += exp

        perm = i
        m = 0
        for _ in range(n):
            m += -1 if (perm & 1) else 1
            perm >>= 1
        m /= n
        m *= m 
        ideal_sqr_mag += ideal * m
        exp_sqr_mag += exp * m

    l2_diff = diff_sqr ** (1 / 2)
    l2_diff_debiased = math.sqrt(max(diff_sqr - noise, 0.0))
    xeb = numer / denom
    if xeb > 1.0:
        xeb = 2.0 - xeb
    if xeb < 0.0:
        xeb = 0.0

    return {
        "qubits": n,
        "l2_difference": float(l2_diff),
        "l2_difference_debiased": float(l2_diff_debiased),
        "xeb_rectified": float(xeb),
        "hog_prob": float(sum_hog_prob),
        "ideal_sqr_mag": float(ideal_sqr_mag),
        "sqr_mag_diff": float(exp_sqr_mag - ideal_sqr_mag),
    }


def normalize_counts(counts, shots):
    return {k: v / shots for k, v in counts.items()}


def main():
    n_qubits = 6
    depth = 40
    z = 4

    # Quantinuum settings
    J, h, dt = -1.0, 2.0, 0.125
    theta = math.pi / 18

    # Pure ferromagnetic
    # J, h, dt = -1.0, 0.0, 0.25
    # theta = 0

    # Pure transverse field
    # J, h, dt = 0.0, 2.0, 0.25
    # theta = -math.pi / 2

    # Critical point (symmetry breaking)
    # J, h, dt = -1.0, 1.0, 0.25
    # theta = -math.pi / 4

    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])
    if len(sys.argv) > 2:
        depth = int(sys.argv[2])
    if len(sys.argv) > 3:
        dt = float(sys.argv[3])
    if len(sys.argv) > 4:
        shots = int(sys.argv[4])
    else:
        shots = max(65536, 1 << (n_qubits + 2))

    print(f"Qubits: {n_qubits}")

    n_rows, n_cols = factor_width(n_qubits, False)
    qubits = list(range(n_qubits))

    # Set the initial temperature by theta.
    qc_aer = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc_aer.ry(theta, q)

    qc_step = brick_wall_tfim_step(n_rows, n_cols, J, h, dt)
    qc_step = transpile(
        qc_step,
        basis_gates=QrackSimulator.get_qiskit_basis_gates(),
    )

    control = AerSimulator(method="statevector")

    r_squared = 0.0
    r_squared_db = 0.0
    r_squared_xeb = 0.0
    ss = 0.0
    ssr = 0.0

    for d in range(1, depth + 1):
        t = d * dt

        # Run the Trotterized simulation with Aer and get the marginal probabilities.
        qc_aer = qc_aer.compose(qc_step)
        qc_aer_sv = qc_aer.copy()
        qc_aer_sv.save_statevector()
        job = control.run(qc_aer_sv)
        ideal_probs = Statevector(job.result().get_statevector()).probabilities()

        # The magnetization components are weighted by (n+1) symmetric "bias" terms over possible Hamming weights.
        pqi_probs = normalize_counts(dict(Counter(
            generate_fermi_hubbard_samples(J=J, h=h, z=z, theta=theta, t=t, n_qubits=n_qubits, shots=shots)
        )), shots)

        result = calc_stats(ideal_probs, pqi_probs, shots)

        # Add up the square residuals:
        r_squared += result["l2_difference"] ** 2
        r_squared_db += result["l2_difference_debiased"] ** 2
        r_squared_xeb += (1.0 - result["xeb_rectified"]) ** 2

        # Save the sum of squares and sum of square residuals on the magnetization curve values.
        ss += result["ideal_sqr_mag"] ** 2
        ssr += result["sqr_mag_diff"] ** 2

    # R^2 and RMSE are elementary and standard measures of goodness-of-fit with simple definitions.
    # Ideal marginal probability would be 1.0, each depth step. Squared and summed, that's depth.
    r_squared = 1.0 - r_squared / depth
    r_squared_db = 1.0 - r_squared_db / depth
    r_squared_xeb = 1.0 - r_squared_xeb / depth
    rmse = (ssr / depth) ** (1 / 2)
    sm_r_squared = 1.0 - (ssr / ss)

    print("L2 norm similarity R^2: " + str(r_squared))
    print("L2 norm debiased similarity R^2: " + str(r_squared_db))
    print("XEB (rectified) R^2: " + str(r_squared_xeb))
    print("Square magnetization RMSE: " + str(rmse))
    print("Square magnetization R^2: " + str(sm_r_squared))

    return 0


if __name__ == "__main__":
    sys.exit(main())
