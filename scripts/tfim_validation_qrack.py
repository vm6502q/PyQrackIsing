# Ising model Trotterization
# by Dan Strano and (OpenAI GPT) Elara

# We reduce transverse field Ising model for globally uniform J and h parameters from a 2^n-dimensional problem to an (n+1)-dimensional approximation that suffers from no Trotter error. Upon noticing most time steps for Quantinuum's parameters had roughly a quarter to a third (or thereabouts) of their marginal probability in |0> state, it became obvious that transition to and from |0> state should dominate the mechanics. Further, the first transition tends to be to or from any state with Hamming weight of 1 (in other words, 1 bit set to 1 and the rest reset 0, or n bits set for Hamming weight of n). Further, on a torus, probability of all states with Hamming weight of 1 tends to be exactly symmetric. Assuming approximate symmetry in every respective Hamming weight, the requirement for the overall probability to converge to 1.0 or 100% in the limit of an infinite-dimensional Hilbert space suggests that Hamming weight marginal probability could be distributed like a geometric series. A small correction to exact symmetry should be made to favor closeness of "like" bits to "like" bits (that is, geometric closeness on the torus of "1" bits to "1" bits and "0" bits to "0" bits), but this does not affect average global magnetization. Adding an oscillation component with angular frequency proportional to J, we find excellent agreement with Trotterization approaching the limit of infinitesimal time step, for R^2 (coefficient of determination) of normalized marginal probability distribution of ideal Trotterized simulation as described by the (n+1)-dimensional approximate model, as well as for R^2 and RMSE (root-mean-square error) of global magnetization curve values.

import math
import numpy as np
import statistics
import sys

from collections import Counter

from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate, RXGate
from qiskit.compiler import transpile

from pyqrack import QrackSimulator

from pyqrackising import generate_tfim_samples, get_tfim_hamming_distribution


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
    horiz_pairs = [
        (r * n_cols + c, r * n_cols + (c + 1) % n_cols)
        for r in range(n_rows)
        for c in range(0, n_cols, 2)
    ]
    add_rzz_pairs(horiz_pairs)

    # Layer 2: horizontal pairs (odd rows)
    horiz_pairs = [
        (r * n_cols + c, r * n_cols + (c + 1) % n_cols)
        for r in range(n_rows)
        for c in range(1, n_cols, 2)
    ]
    add_rzz_pairs(horiz_pairs)

    # Layer 3: vertical pairs (even columns)
    vert_pairs = [
        (r * n_cols + c, ((r + 1) % n_rows) * n_cols + c)
        for r in range(1, n_rows, 2)
        for c in range(n_cols)
    ]
    add_rzz_pairs(vert_pairs)

    # Layer 4: vertical pairs (odd columns)
    vert_pairs = [
        (r * n_cols + c, ((r + 1) % n_rows) * n_cols + c)
        for r in range(0, n_rows, 2)
        for c in range(n_cols)
    ]
    add_rzz_pairs(vert_pairs)

    # Second half of transverse field term
    for q in qubits:
        circ.rx(h * dt, q)

    return circ


# Calculate various statistics based on comparison between ideal (Trotterized) and approximate (continuum) measurement distributions.
def calc_stats(n_rows, n_cols, ideal_probs, bias, depth):
    # For QV, we compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    n = n_rows * n_cols
    n_pow = 2**n
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

        # How many bits are 1, in the basis state?
        hamming_weight = hamming_distance(i, 0, n)
        # How closely grouped are "like" bits to "like"?
        expected_closeness = expected_closeness_weight(n_rows, n_cols, hamming_weight)
        # When we add all "closeness" possibilities for the particular Hamming weight, we should maintain the (n+1) mean probability dimensions.
        normed_closeness = (1 + closeness_like_bits(i, n_rows, n_cols)) / (1 + expected_closeness)
        # Use a normalized weighted average that favors the (n+1)-dimensional model at later times.
        # The (n+1)-dimensional marginal probability is the product of a function of Hamming weight and "closeness," split among all basis states with that specific Hamming weight.
        count = normed_closeness * bias[hamming_weight] / math.comb(n, hamming_weight)

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
        "xeb": xeb
    }


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
    return sum(
        ch1 != ch2 for ch1, ch2 in zip(int_to_bitstring(s1, n), int_to_bitstring(s2, n))
    )


# From https://stackoverflow.com/questions/13070461/get-indices-of-the-top-n-values-of-a-list#answer-38835860
def top_n(n, a):
    median_index = len(a) >> 1
    if n > median_index:
        n = median_index
    return np.argsort(a)[-n:]


def main():
    n_qubits = 8
    depth = 20
    t1 = 0

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
        t1 = float(sys.argv[4])
    if len(sys.argv) > 5:
        shots = int(sys.argv[5])
    else:
        shots = max(65536, 1 << (n_qubits + 2))
    if len(sys.argv) > 6:
        trials = int(sys.argv[6])
    else:
        trials = 8 if t1 > 0 else 1

    print("t1: " + str(t1))

    n_rows, n_cols = factor_width(n_qubits, False)
    qubits = list(range(n_qubits))

    # Set the initial temperature by theta.
    qc_aer = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc_aer.ry(theta, q)

    control = QrackSimulator(n_qubits, isTensorNetwork=False)
    basis_gates = QrackSimulator.get_qiskit_basis_gates()
    qc_aer = transpile(
        qc_aer,
        basis_gates=basis_gates
    )
    control.run_qiskit_circuit(qc_aer)
    control_probs = control.out_probs()

    bias_0 = get_tfim_hamming_distribution(J=J, h=h, z=z, theta=theta, t=0, n_qubits=n_qubits)

    c_sqr_magnetization = 0
    for p in range(1 << n_qubits):
        perm = p
        m = 0
        for _ in range(n_qubits):
            m += -1 if (perm & 1) else 1
            perm >>= 1
        m /= n_qubits
        c_sqr_magnetization += control_probs[p] * m * m

    result = calc_stats(
        n_rows,
        n_cols,
        control_probs,
        bias_0,
        0
    )

    # Add up the square residuals:
    r_squared = result["l2_difference"] ** 2
    xeb = result["xeb"]
    if xeb > 1.0:
        # Rectify, to penalize values greater than 1.0
        xeb -= xeb - 1.0
    r_squared_xeb = 1.0 - xeb

    magnetization_0, sqr_magnetization_0 = 0, 0
    for hamming_weight, value in enumerate(bias_0):
        m = 1.0 - 2 * hamming_weight / n_qubits
        magnetization_0 += value * m
        sqr_magnetization_0 += value * m * m

    # Save the sum of squares and sum of square residuals on the magnetization curve values.
    ss = c_sqr_magnetization**2
    ssr = (c_sqr_magnetization - sqr_magnetization_0) ** 2

    qc_aer = QuantumCircuit(n_qubits)
    trotter_step(qc_aer, qubits, (n_rows, n_cols), J, h, dt)

    # Run the Trotterized simulation with Aer and get the marginal probabilities.
    qc_aer = transpile(
        qc_aer,
        basis_gates=basis_gates
    )

    for d in range(1, depth + 1):
        t = d * dt

        # For each depth step, we append an additional Trotter step to Aer's circuit.
        control.run_qiskit_circuit(qc_aer)
        control_probs = control.out_probs()

        # The magnetization components are weighted by (n+1) symmetric "bias" terms over possible Hamming weights.
        bias = get_tfim_hamming_distribution(J=J, h=h, z=z, theta=theta, t=t, n_qubits=n_qubits)

        # The full 2^n marginal probabilities will be produced in the statistics calculation,
        # but notice that the global magnetization value only requires (n+1) dimensions of marginal probability,
        # the marginal probability per each Hilbert space basis dimension is trivial to calculate by closed form,
        # and we simply need to be thoughtful in how to extract expectation values to maximize similar symmetries.
        result = calc_stats(
            n_rows,
            n_cols,
            control_probs,
            bias,
            d
        )

        # Add up the square residuals:
        r_squared += result["l2_difference"] ** 2
        xeb = result["xeb"]
        if xeb > 1.0:
            # Rectify, to penalize values greater than 1.0
            xeb -= xeb - 1.0
        r_squared_xeb += 1.0 - xeb

        magnetization, sqr_magnetization = 0, 0
        for hamming_weight, value in enumerate(bias):
            m = 1.0 - 2 * hamming_weight / n_qubits
            magnetization += value * m
            sqr_magnetization += value * m * m

        # Calculate the "control-case" magnetization values, from Aer's samples.
        c_magnetization, c_sqr_magnetization = 0, 0
        for p in range(1 << n_qubits):
            perm = p
            m = 0
            for _ in range(n_qubits):
                m += -1 if (perm & 1) else 1
                perm >>= 1
            m /= n_qubits
            c_magnetization += control_probs[p] * m
            c_sqr_magnetization += control_probs[p] * m * m

        # Save the sum of squares and sum of square residuals on the magnetization curve values.
        ss += c_sqr_magnetization**2
        ssr += (c_sqr_magnetization - sqr_magnetization) ** 2

    # R^2 and RMSE are elementary and standard measures of goodness-of-fit with simple definitions.
    # Ideal marginal probability would be 1.0, each depth step. Squared and summed, that's depth.
    r_squared = 1.0 - r_squared / (depth + 1)
    r_squared_xeb = 1.0 - r_squared_xeb / (depth + 1)
    rmse = (ssr / depth) ** (1 / 2)
    sm_r_squared = 1.0 - (ssr / ss)

    print("L2 norm similarity R^2: " + str(r_squared))
    print("Rectified XEB R^2: " + str(r_squared_xeb))
    print("Square magnetization RMSE: " + str(rmse))
    print("Square magnetization R^2: " + str(sm_r_squared))

    # Happy Qracking! You rock!

    return 0


if __name__ == "__main__":
    sys.exit(main())
