# How good are Google's own "patch circuits" and "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)

import math
import random
import statistics
import sys

from collections import Counter

from scipy.stats import binom

from pyqrack import QrackSimulator

from qiskit import QuantumCircuit
from qiskit_aer.backends import AerSimulator
from qiskit.quantum_info import Statevector


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len
    if col_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")

    return (row_len, col_len)


def cx(sim, q1, q2):
    sim.cx(q1, q2)


def cy(sim, q1, q2):
    sim.cy(q1, q2)


def cz(sim, q1, q2):
    sim.cz(q1, q2)


def acx(sim, q1, q2):
    sim.x(q1)
    sim.cx(q1, q2)
    sim.x(q1)


def acy(sim, q1, q2):
    sim.x(q1)
    sim.cy(q1, q2)
    sim.x(q1)


def acz(sim, q1, q2):
    sim.x(q1)
    sim.cz(q1, q2)
    sim.x(q1)


def bench_qrack(width, depth, cycles):
    # This is a "nearest-neighbor" coupler random circuit.

    lcv_range = range(width)
    all_bits = list(lcv_range)

    # Nearest-neighbor couplers:
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    two_bit_gates = cx, cy, cz, acx, acy, acz

    row_len, col_len = factor_width(width)

    rcs = QuantumCircuit(width)
    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            rcs.u(th, ph, lm, i)

        # Nearest-neighbor couplers:
        ############################
        gate = gateSequence.pop(0)
        gateSequence.append(gate)
        for row in range(1, row_len, 2):
            for col in range(col_len):
                temp_row = row
                temp_col = col
                temp_row = temp_row + (1 if (gate & 2) else -1)
                temp_col = temp_col + (1 if (gate & 1) else 0)

                if temp_row < 0:
                    temp_row = temp_row + row_len
                if temp_col < 0:
                    temp_col = temp_col + col_len
                if temp_row >= row_len:
                    temp_row = temp_row - row_len
                if temp_col >= col_len:
                    temp_col = temp_col - col_len

                b1 = row * row_len + col
                b2 = temp_row * row_len + temp_col

                if (b1 >= width) or (b2 >= width) or (b1 == b2):
                    continue

                if d & 1:
                    t = b1
                    b1 = b2
                    b2 = t

                g = random.choice(two_bit_gates)
                g(rcs, b1, b2)

    ops = ['I', 'X', 'Y', 'Z']
    pauli_strings = []

    otoc = QuantumCircuit(width)
    for cycle in range(cycles):
        otoc &= rcs
        string = []
        for b in range(width):
            string.append(random.choice(ops))
        pauli_strings.append("".join(string))
        act_string(otoc, string)
        otoc &= rcs.inverse()


    experiment = QrackSimulator(width, isTensorNetwork=False)
    experiment.run_qiskit_circuit(otoc)

    otoc_aer = otoc.copy()
    otoc_aer.save_statevector()
    control = AerSimulator(method="statevector")
    job = control.run(otoc_aer)

    shots = 1 << (width + 2)
    experiment_counts = dict(Counter(experiment.measure_shots(all_bits, shots)))
    control_probs = Statevector(job.result().get_statevector()).probabilities()

    return calc_stats(control_probs, experiment_counts, d + 1, shots), pauli_strings


def act_string(otoc, string):
    for i in range(len(string)):
        match string[i]:
            case 'X':
                otoc.x(i)
            case 'Y':
                otoc.y(i)
            case 'Z':
                otoc.z(i)
            case _:
                pass


def calc_stats(ideal_probs, counts, depth, shots):
    # For QV, we compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    n_pow = len(ideal_probs)
    n = int(round(math.log2(n_pow)))
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)
    numer = 0
    denom = 0
    sum_hog_counts = 0
    for i in range(n_pow):
        count = counts[i] if i in counts else 0
        ideal = ideal_probs[i]

        # XEB / EPLG
        denom += (ideal - u_u) ** 2
        numer += (ideal - u_u) * ((count / shots) - u_u)

        # QV / HOG
        if ideal > threshold:
            sum_hog_counts += count

    hog_prob = sum_hog_counts / shots
    xeb = numer / denom
    # p-value of heavy output count, if method were actually 50/50 chance of guessing
    p_val = (
        (1 - binom.cdf(sum_hog_counts - 1, shots, 1 / 2)) if sum_hog_counts > 0 else 1
    )

    return {
        "qubits": n,
        "depth": depth,
        "xeb": float(xeb),
        "hog_prob": float(hog_prob),
        "p-value": float(p_val),
    }


def main():
    if len(sys.argv) < 4:
        raise RuntimeError(
            "Usage: python3 fc_qiskit_validation.py [width] [depth] [cycles]"
        )

    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    cycles = int(sys.argv[3])

    # Run the benchmarks
    print(bench_qrack(width, depth, cycles))

    return 0


if __name__ == "__main__":
    sys.exit(main())
