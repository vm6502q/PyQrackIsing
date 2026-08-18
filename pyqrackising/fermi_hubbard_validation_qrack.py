# Fermi-Hubbard / XXZ-Heisenberg validation, v2
# by Dan Strano and Claude (Anthropic)
#
# TFIM -> Fermi-Hubbard disclaimer):
#   - This is a mean-field / entropy-based heuristic correction, not derived
#     from the Heisenberg Hamiltonian in closed form. It was benchmarked at
#     ONE parameter point (n_qubits=6, Quantinuum J/h/dt/theta defaults) and
#     has not been swept across parameter space or system size.
#   - Direct numerical comparison (see hamming_conservation_test.py) shows
#     that RXX+RYY+RZZ terms, despite each individually commuting with total
#     Hamming weight (Sz) in isolation, measurably shift BOTH the mean and
#     the variance of the true Hamming-weight marginal relative to pure TFIM
#     once interleaved with the transverse-field Trotter steps -- this is a
#     real correction, not just added entropy/noise, and the entropy-blend
#     approach here only captures it directionally, via a proxy signal, not
#     from the actual mean-shift mechanism. A first-principles closed-form
#     ansatz (analogous to the original TFIM derivation, this time fit
#     against RZZ+RXX+RYY+RX Trotter data) is the right next step and has
#     NOT been done here.

import math
import numpy as np
import statistics
import sys

from collections import Counter

from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate, RXXGate, RYYGate
from qiskit.compiler import transpile

from pyqrack import QrackSimulator

from pyqrackising.generate_tfim_samples import generate_fermi_hubbard_samples


def factor_width(width, is_transpose=False):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (col_len, row_len) if is_transpose else (row_len, col_len)


def edge_layers(n_rows, n_cols):
    horiz_even = [(r * n_cols + c, r * n_cols + (c + 1) % n_cols) for r in range(n_rows) for c in range(0, n_cols, 2)]
    horiz_odd = [(r * n_cols + c, r * n_cols + (c + 1) % n_cols) for r in range(n_rows) for c in range(1, n_cols, 2)]
    vert_even = [(r * n_cols + c, ((r + 1) % n_rows) * n_cols + c) for r in range(1, n_rows, 2) for c in range(n_cols)]
    vert_odd = [(r * n_cols + c, ((r + 1) % n_rows) * n_cols + c) for r in range(0, n_rows, 2) for c in range(n_cols)]
    return [horiz_even, horiz_odd, vert_even, vert_odd]


def trotter_step_heisenberg(circ, qubits, lattice_shape, J, h, dt, Jxy=None):
    """The CORRECT ideal circuit for Fermi-Hubbard/Heisenberg validation:
    RZZ + RXX + RYY (isotropic hopping+interaction) + RX (transverse field).
    Jxy defaults to J (isotropic Heisenberg + field)."""
    if Jxy is None:
        Jxy = J
    n_rows, n_cols = lattice_shape
    for q in qubits:
        circ.rx(h * dt, q)
    for layer in edge_layers(n_rows, n_cols):
        for q1, q2 in layer:
            circ.append(RZZGate(2 * J * dt), [q1, q2])
            circ.append(RXXGate(2 * Jxy * dt), [q1, q2])
            circ.append(RYYGate(2 * Jxy * dt), [q1, q2])
    for q in qubits:
        circ.rx(h * dt, q)
    return circ


def normalize_counts(counts, shots):
    return {k: v / shots for k, v in counts.items()}


def calc_stats(ideal_probs, pqi_probs, shots):
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
        diff_sqr += (ideal - exp) ** 2
        noise += exp * (1 - exp) / shots
        denom += (ideal - u_u) ** 2
        numer += (ideal - u_u) * (exp - u_u)
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

    l2_diff = diff_sqr ** 0.5
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
        shots = 1 << (n_qubits + 2)

    print(f"Qubits: {n_qubits}  (ideal = RZZ+RXX+RYY+RX Heisenberg+field circuit)")

    n_rows, n_cols = factor_width(n_qubits)
    qubits = list(range(n_qubits))

    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.ry(theta, q)
    control = QrackSimulator(n_qubits)
    basis_gates = QrackSimulator.get_qiskit_basis_gates()
    qc = transpile(qc, basis_gates=basis_gates)
    control.run_qiskit_circuit(qc)

    r_squared = r_squared_db = r_squared_xeb = ss = ssr = 0.0
    for d in range(1, depth + 1):
        t = d * dt
        step_circ = QuantumCircuit(n_qubits)
        trotter_step_heisenberg(step_circ, qubits, (n_rows, n_cols), J, h, dt)
        step_circ = transpile(step_circ, basis_gates=basis_gates)
        control.run_qiskit_circuit(step_circ)
        control_probs = control.out_probs()

        pqi_probs = normalize_counts(dict(Counter(
            generate_fermi_hubbard_samples(J=J, h=h, z=z, theta=theta, t=t, n_qubits=n_qubits, shots=shots)
        )), shots)

        result = calc_stats(control_probs, pqi_probs, shots)
        r_squared += result["l2_difference"] ** 2
        r_squared_db += result["l2_difference_debiased"] ** 2
        r_squared_xeb += (1.0 - result["xeb_rectified"]) ** 2
        ss += result["ideal_sqr_mag"] ** 2
        ssr += result["sqr_mag_diff"] ** 2

    r_squared = 1.0 - r_squared / depth
    r_squared_db = 1.0 - r_squared_db / depth
    r_squared_xeb = 1.0 - r_squared_xeb / depth
    rmse = (ssr / depth) ** 0.5
    sm_r_squared = 1.0 - (ssr / ss)

    print("L2 norm similarity R^2: " + str(r_squared))
    print("L2 norm debiased similarity R^2: " + str(r_squared_db))
    print("XEB (rectified) R^2: " + str(r_squared_xeb))
    print("Square magnetization RMSE: " + str(rmse))
    print("Square magnetization R^2: " + str(sm_r_squared))
    return 0


if __name__ == "__main__":
    sys.exit(main())
