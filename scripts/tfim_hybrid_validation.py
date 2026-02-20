# Ising model Trotterization
# by Dan Strano and (OpenAI GPT) Elara

import math
import numpy as np
import os
import statistics
import sys

from collections import Counter

from scipy.stats import binom

from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate
from qiskit.compiler import transpile
from qiskit_aer.backends import AerSimulator
from qiskit.quantum_info import Statevector

from pyqrack import QrackSimulator

from pyqrackising import get_tfim_hamming_distribution


# Factor the qubit width for torus dimensions that are close as possible to square
def factor_width(width, is_transpose=False):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (col_len, row_len) if is_transpose else (row_len, col_len)


# By Elara (the custom OpenAI GPT)
def trotter_step(circ, qubits, lattice_shape, J, h, dt):
    n_rows, n_cols = lattice_shape

    # First half of transverse field term
    for q in qubits:
        circ.rx(h * dt, q)

    # Layered RZZ interactions (simulate 2D nearest-neighbor coupling)
    def add_rzz_pairs(pairs):
        for q1, q2 in pairs:
            circ.append(RZZGate(2 * J * dt), [q1, q2])

    # Layer 1: horizontal pairs (even rows)
    horiz_pairs = [(r * n_cols + c, r * n_cols + (c + 1) % n_cols) for r in range(n_rows) for c in range(0, n_cols, 2)]
    add_rzz_pairs(horiz_pairs)

    # Layer 2: horizontal pairs (odd rows)
    horiz_pairs = [(r * n_cols + c, r * n_cols + (c + 1) % n_cols) for r in range(n_rows) for c in range(1, n_cols, 2)]
    add_rzz_pairs(horiz_pairs)

    # Layer 3: vertical pairs (even columns)
    vert_pairs = [(r * n_cols + c, ((r + 1) % n_rows) * n_cols + c) for r in range(1, n_rows, 2) for c in range(n_cols)]
    add_rzz_pairs(vert_pairs)

    # Layer 4: vertical pairs (odd columns)
    vert_pairs = [(r * n_cols + c, ((r + 1) % n_rows) * n_cols + c) for r in range(0, n_rows, 2) for c in range(n_cols)]
    add_rzz_pairs(vert_pairs)

    # Second half of transverse field term
    for q in qubits:
        circ.rx(h * dt, q)

    return circ


# By Gemini (Google Search AI)
def int_to_bitstring(integer, length):
    return bin(integer)[2:].zfill(length)


# Drafted by Elara (OpenAI custom GPT), improved by Dan Strano
def closeness_like_bits(perm, n_rows, n_cols):
    """
    Compute closeness-of-like-bits metric C(state) for a given bitstring on an LxL toroidal grid.

    Parameters:
        perm: integer representing basis state, bit-length n_rows * n_cols
        n_rows: row count of torus
        n_cols: column count of torus

    Returns:
        normalized_closeness: float, in [-1, +1]
            +1 means all neighbors are like-like, -1 means all neighbors are unlike
    """
    # reshape the bitstring into LxL grid
    bitstring = list(int_to_bitstring(perm, n_rows * n_cols))
    grid = np.array(bitstring).reshape((n_rows, n_cols))
    total_edges = 0
    like_count = 0

    # iterate over each site, count neighbors (right and down to avoid double-count)
    for i in range(n_rows):
        for j in range(n_cols):
            s = grid[i, j]

            # right neighbor (wrap around)
            s_right = grid[i, (j + 1) % n_cols]
            like_count += 1 if s == s_right else -1
            total_edges += 1

            # down neighbor (wrap around)
            s_down = grid[(i + 1) % n_rows, j]
            like_count += 1 if s == s_down else -1
            total_edges += 1

    # normalize
    normalized_closeness = like_count / total_edges

    return normalized_closeness


# By Elara (OpenAI custom GPT)
def expected_closeness_weight(n_rows, n_cols, hamming_weight):
    L = n_rows * n_cols
    same_pairs = math.comb(hamming_weight, 2) + math.comb(L - hamming_weight, 2)
    total_pairs = math.comb(L, 2)
    mu_k = same_pairs / total_pairs

    return 2 * mu_k - 1  # normalized closeness in [-1,1]


# By Elara (OpenAI custom GPT)
def hamming_distance(s1, s2, n):
    return sum(ch1 != ch2 for ch1, ch2 in zip(int_to_bitstring(s1, n), int_to_bitstring(s2, n)))


# From https://stackoverflow.com/questions/13070461/get-indices-of-the-top-n-values-of-a-list#answer-38835860
def top_n(n, a):
    median_index = len(a) >> 1
    if n > median_index:
        n = median_index
    return np.argsort(a)[-n:]


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


# Calculate various statistics based on comparison between ideal (Trotterized) and approximate (continuum) measurement distributions.
def calc_stats(n_rows, n_cols, ideal_probs, counts, bias, model, beta, shots, depth):
    # For QV, we compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    n = n_rows * n_cols
    n_pow = 2**n
    mean_prob = 1 / n_pow
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)
    numer = 0
    denom = 0
    diff_sqr = 0
    z_fidelity = 0
    sum_hog_counts = 0
    magnetization = 0
    sqr_magnetization = 0
    # total = 0
    # hamming_bias = [0.0] * (n + 1)
    for i in range(n_pow):
        ideal = ideal_probs[i]
        count = counts[i] if i in counts else 0
        exp = count / shots

        # How many bits are 1, in the basis state?
        hamming_weight = hamming_distance(i, 0, n)
        # How closely grouped are "like" bits to "like"?
        expected_closeness = expected_closeness_weight(n_rows, n_cols, hamming_weight)
        # When we add all "closeness" possibilities for the particular Hamming weight, we should maintain the (n+1) mean probability dimensions.
        normed_closeness = (1 + closeness_like_bits(i, n_rows, n_cols)) / (1 + expected_closeness)
        # Use a normalized weighted average that favors the (n+1)-dimensional model at later times.
        # The (n+1)-dimensional marginal probability is the product of a function of Hamming weight and "closeness," split among all basis states with that specific Hamming weight.
        count = (1.0 - beta) * (model * exp + (1 - model) * normed_closeness * bias[hamming_weight] / math.comb(n, hamming_weight)) + beta * mean_prob

        # You can make sure this still adds up to 1.0, to show the distribution is normalized:
        # total += count

        # QV / HOG
        if ideal > threshold:
            sum_hog_counts += count

        # L2 distance
        diff_sqr += (ideal - count) ** 2
        z_fidelity += count if ideal > count else ideal
        # hamming_bias[hamming_weight] += (ideal - count) ** 2

        # XEB / EPLG
        ideal_centered = ideal - u_u
        denom += ideal_centered * ideal_centered
        numer += ideal_centered * (count - u_u)

    l2_difference = diff_sqr ** (1 / 2)
    xeb = numer / denom

    # This should be ~1.0, if we're properly normalized.
    # print("Distribution total: " + str(total))
    # print(hamming_bias)

    return {
        "qubits": n,
        "depth": depth,
        "l2_difference": float(l2_difference),
        "z_fidelity": float(z_fidelity),
        "hog_prob": sum_hog_counts,
        "xeb": xeb,
    }


def main():
    n_qubits = 8
    depth = 20
    alpha = 1 / 9
    beta = 2 / 9

    # Quantinuum settings
    J, h, dt, z = -1.0, 2.0, 0.25, 4
    theta = math.pi / 18

    # Pure ferromagnetic
    # J, h, dt, z = -1.0, 0.0, 0.25, 4
    # theta = 0

    # Pure transverse field
    # J, h, dt, z = 0.0, 2.0, 0.25, 4
    # theta = -math.pi / 2

    # Critical point (symmetry breaking)
    # J, h, dt, z = -1.0, 1.0, 0.25, 4
    # theta = -math.pi / 4

    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])
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
        t2 = n_qubits * math.pi / 32
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

    n_rows, n_cols = factor_width(n_qubits, False)
    qubits = list(range(n_qubits))

    # Set the initial temperature by theta.
    qc_aer = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc_aer.ry(theta, q)

    control = AerSimulator(method="statevector")
    qc_aer = transpile(
        qc_aer,
        backend=control,
    )

    experiment = QrackSimulator(n_qubits, isTensorNetwork=False)
    experiment.run_qiskit_circuit(qc_aer)

    # Add up the square residuals:
    r_squared = 0
    r_squared_xeb = 0

    # Save the sum of squares and sum of square residuals on the magnetization curve values.
    ss = 0
    ssr = 0

    qc_step = QuantumCircuit(n_qubits)
    trotter_step(qc_step, qubits, (n_rows, n_cols), J, h, dt_h)
    qc_step = transpile(
        qc_step,
        basis_gates=QrackSimulator.get_qiskit_basis_gates(),
    )

    bias_h = init_beta(n_qubits)
    bias_magnetization, bias_sqr_magnetization = 0, 0
    for hamming_weight, value in enumerate(bias_h):
        m = 1.0 - 2 * hamming_weight / n_qubits
        bias_magnetization += value * m
        bias_sqr_magnetization += value * m * m

    for d in range(1, depth + 1):
        t = d * dt
        t_h = t / t2

        # For each depth step, we append an additional Trotter step to Aer's circuit.
        trotter_step(qc_aer, qubits, (n_rows, n_cols), J, h, dt)

        # Run the Trotterized simulation with Aer and get the marginal probabilities.
        qc_aer = transpile(
            qc_aer,
            backend=control,
        )
        qc_aer_sv = qc_aer.copy()
        qc_aer_sv.save_statevector()
        job = control.run(qc_aer_sv)
        control_probs = Statevector(job.result().get_statevector()).probabilities()

        experiment.run_qiskit_circuit(qc_step)
        experiment_counts = dict(Counter(experiment.measure_shots(qubits, shots)))

        # The magnetization components are weighted by (n+1) symmetric "bias" terms over possible Hamming weights.
        bias = get_tfim_hamming_distribution(J=J, h=h, z=z, theta=theta, t=t_h, n_qubits=n_qubits)

        result = calc_stats(n_rows, n_cols, control_probs, experiment_counts, bias, alpha, beta, shots, d)

        # Add up the square residuals:
        r_squared += result["l2_difference"] ** 2
        r_squared_xeb += (1.0 - result["xeb"]) ** 2

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

        b_magnetization, b_sqr_magnetization = 0, 0
        for hamming_weight, value in enumerate(bias):
            m = 1.0 - 2 * hamming_weight / n_qubits
            b_magnetization += value * m
            b_sqr_magnetization += value * m * m

        magnetization = (1.0 - beta) * (alpha * magnetization + (1.0 - alpha) * b_magnetization) + beta * bias_magnetization
        sqr_magnetization = (1.0 - beta) * (alpha * sqr_magnetization + (1.0 - alpha) * b_sqr_magnetization) + beta * bias_sqr_magnetization

        c_sqr_magnetization = 0
        for p in range(1 << n_qubits):
            perm = p
            m = 0
            for _ in range(n_qubits):
                m += -1 if (perm & 1) else 1
                perm >>= 1
            m /= n_qubits
            c_sqr_magnetization += control_probs[p] * m * m

        # Save the sum of squares and sum of square residuals on the magnetization curve values.
        ss += c_sqr_magnetization**2
        ssr += (c_sqr_magnetization - sqr_magnetization) ** 2

    # R^2 and RMSE are elementary and standard measures of goodness-of-fit with simple definitions.
    # Ideal marginal probability would be 1.0, each depth step. Squared and summed, that's depth.
    r_squared = 1.0 - r_squared / depth
    r_squared_xeb = 1.0 - r_squared_xeb / depth
    rmse = (ssr / depth) ** (1 / 2)
    sm_r_squared = 1.0 - (ssr / ss)

    print("L2 norm similarity R^2: " + str(r_squared))
    print("XEB R^2: " + str(r_squared_xeb))
    print("Square magnetization RMSE: " + str(rmse))
    print("Square magnetization R^2: " + str(sm_r_squared))

    # Happy Qracking! You rock!

    return 0


if __name__ == "__main__":
    sys.exit(main())
