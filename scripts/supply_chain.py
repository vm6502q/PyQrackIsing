# supply_chain.py
# Provided by Elara (the custom OpenAI GPT)

import itertools
import math
import random
import numpy as np
import matplotlib.pyplot as plt

from PyQrackIsing import tfim_magnetization


def simulate_tfim(
    J_func,
    h_func,
    n_qubits=64,
    n_steps=20,
    delta_t=0.1,
    theta=[],
    z=[],
):
    t2 = 1
    omega = 3 * math.pi / 2

    magnetizations = []

    for step in range(n_steps):
        t = step * delta_t
        J_t = J_func(t)
        h_t = h_func(t)

        # compute magnetization per qubit, then average
        mag_per_qubit = []

        for q in range(n_qubits):
            # gather local couplings for qubit q
            J_eff = sum(J_t[q, j] for j in range(n_qubits) if (j != q)) / z[q]
            h_eff = h_t[q]
            mag_per_qubit.append(tfim_magnetization(J_eff, h_eff, z[q], theta[q], t, n_qubits))

        # combine per-qubit magnetizations (e.g., average)
        step_mag = float(np.mean(mag_per_qubit))
        magnetizations.append(step_mag)

    return magnetizations


# Dynamic J(t) generator
def generate_Jt(n_nodes, t):
    J = np.zeros((n_nodes, n_nodes))

    # Base ring topology
    for i in range(n_nodes):
        J[i, (i + 1) % n_nodes] = -1.0
        J[(i + 1) % n_nodes, i] = -1.0

    # Simulate disruption:
    if t >= 0.5 and t < 1.5:
        # "Port 3" temporarily fails → remove its coupling
        J[2, 3] = J[3, 2] = 0
        J[3, 4] = J[4, 3] = 0
    if t >= 1.0 and t < 1.5:
        # Alternate weak link opens between 1 and 4
        J[1, 4] = J[4, 1] = -0.3
    # Restoration: after step 15, port 3 recovers

    return J


def generate_ht(n_nodes, t):
    # We can program h(q, t) for spatial-temporal locality.
    h = np.zeros(n_nodes)
    # Time-varying transverse field
    c = 0.5 + 0.5 * np.cos(t * math.pi / 10)
    # We can program for spatial locality, but we don't.
    #  n_sqrt = math.sqrt(n_nodes)
    for i in range(n_nodes):
        # "Longitude"-dependent severity (arbitrary)
        # h[i] = ((i % n_sqrt) / n_sqrt) * c
        h[i] = c

    return h


if __name__ == "__main__":
    # Example usage

    # Qubit count
    n_qubits = 64
    # Trotter step count
    n_steps = 40
    # Simulated time per Trotter step
    delta_t = 0.1
    # Initial temperatures (per qubit)
    theta = [math.pi / 18] * n_qubits
    # Number of nearest neighbors:
    z = [2] * n_qubits
    J_func = lambda t: generate_Jt(n_qubits, t)
    h_func = lambda t: generate_ht(n_qubits, t)

    mag = simulate_tfim(J_func, h_func, n_qubits, n_steps, delta_t, theta, z)
    ylim = ((min(mag) * 100) // 10) / 10
    plt.figure(figsize=(14, 14))
    plt.plot(list(range(1, n_steps + 1)), mag, marker="o", linestyle="-")
    plt.title(
        "Supply Chain Resilience over Time (Magnetization vs Trotter Depth, "
        + str(n_qubits)
        + " Qubits)"
    )
    plt.xlabel("Trotter Depth")
    plt.ylabel("Magnetization")
    plt.ylim(ylim, 1.0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
