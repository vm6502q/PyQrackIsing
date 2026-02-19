import math
import os
import sys
import time

from collections import Counter

import numpy as np

from scipy.stats import distributions as dists

import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.compiler import transpile

from pyqrack import QrackSimulator
from pyqrackising import get_tfim_hamming_distribution


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


def main():
    n_qubits = 16
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

    if os.environ['QRACK_MAX_PAGING_QB'] and (int(os.environ['QRACK_MAX_PAGING_QB']) < n_qubits):
        alpha = 0.0
        beta = 0.0
    else:
        alpha = 1.0
        beta = 0.0

    if len(sys.argv) > 2:
        depth = int(sys.argv[2])
    if len(sys.argv) > 3:
        dt = float(sys.argv[3])
    if len(sys.argv) > 4:
        alpha = float(sys.argv[4])
        alpha = min(max(alpha, 0), 1)
    if len(sys.argv) > 5:
        beta = float(sys.argv[5])
        beta = min(max(beta, 0), 1)
    if len(sys.argv) > 6:
        t2 = float(sys.argv[6])
    else:
        t2 = 1.0
    if len(sys.argv) > 7:
        shots = int(sys.argv[7])
    else:
        shots = max(65536, 1 << (n_qubits + 2))

    dt_h = dt / t2

    print(f"Qubits: {n_qubits}")
    print(f"Subsystem size: {os.environ['QRACK_MAX_PAGING_QB']}")
    print(f"alpha: {alpha}")
    print(f"beta: {beta}")
    print(f"t2: {t2}")

    depths = list(range(1, depth + 1))
    results = []
    magnetizations = []

    n_rows, n_cols = factor_width(n_qubits, False)
    qubits = list(range(n_qubits))

    bias_h = init_beta(n_qubits)
    bias_magnetization, bias_sqr_magnetization = 0, 0
    for hamming_weight, value in enumerate(bias_h):
        m = 1.0 - 2 * hamming_weight / n_qubits
        bias_magnetization += value * m
        bias_sqr_magnetization += value * m * m

    # Set the initial temperature by theta.
    qc_aer = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc_aer.ry(theta, q)

    qc_step = brick_wall_tfim_step(n_rows, n_cols, J, h, dt_h)
    qc_step = transpile(
        qc_step,
        basis_gates=QrackSimulator.get_qiskit_basis_gates(),
    )

    start = time.perf_counter()

    experiment = QrackSimulator(n_qubits)
    experiment.run_qiskit_circuit(qc_aer)

    for d in depths:
        t = d * dt
        t_h = t / t2

        experiment.run_qiskit_circuit(qc_step)
        experiment_counts = dict(Counter(experiment.measure_shots(qubits, shots)))

        # The magnetization components are weighted by (n+1) symmetric "bias" terms over possible Hamming weights.
        bias_z = get_tfim_hamming_distribution(J=J, h=h, z=z, theta=theta, t=t, n_qubits=n_qubits)
        bias_x = get_tfim_hamming_distribution(J=-h, h=-J, z=z, theta=theta+np.pi/2, t=t, n_qubits=n_qubits)
        bias_y = get_tfim_hamming_distribution(J=J, h=h, z=z, theta=theta+np.pi/2, t=t, n_qubits=n_qubits)
        bias = [(z + x + y) / 3 for z, x, y in zip(bias_z, bias_x, bias_y)]

        magnetization, sqr_magnetization = 0, 0
        for key, value in experiment_counts.items():
            m = 0
            for _ in range(n_qubits):
                m += -1 if (key & 1) else 1
                key >>= 1
            m /= n_qubits
            magnetization += m * value
            sqr_magnetization += m * m * value
        magnetization /= shots
        sqr_magnetization /= shots
        
        b_magnetization, b_sqr_magnetization = 0.0, 0.0
        for hamming_weight, value in enumerate(bias):
            m = 1.0 - 2 * hamming_weight / n_qubits
            b_magnetization += value * m
            b_sqr_magnetization += value * m * m

        magnetization = (1.0 - beta) * (alpha * magnetization + (1.0 - alpha) * b_magnetization) + beta * bias_magnetization
        sqr_magnetization = (1.0 - beta) * (alpha * sqr_magnetization +  (1.0 - alpha) * b_sqr_magnetization) + beta * bias_sqr_magnetization

        seconds = time.perf_counter() - start

        results.append(
            {
                "width": n_qubits,
                "depth": d,
                "magnetization": float(magnetization),
                "square_magnetization": float(sqr_magnetization),
                "seconds": seconds,
            }
        )
        magnetizations.append(sqr_magnetization)
        print(results[-1])

    # Plotting (contributed by Elara, an OpenAI custom GPT)
    plt.figure(figsize=(14, 14))

    plt.plot(depths, magnetizations, marker='o', linestyle='-')

    plt.xlabel("step")
    plt.ylabel(r"$\langle Z^2_{tot} \rangle$")
    plt.title("Square Magnetization vs Trotter Depth")
    # plt.legend()
    plt.grid(True)
    plt.xticks(depths)
    # plt.ylim(0.05, 0.7)
    plt.tight_layout()
    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
