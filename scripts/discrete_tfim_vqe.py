# Quantum chemistry example
# Developed with help from (OpenAI custom GPT) Elara
# (Requires PennyLane and OpenFermion)

import pennylane as qml
from pennylane import numpy as np
import openfermion as of
from openfermionpyscf import run_pyscf
from openfermion.transforms import jordan_wigner

from PyQrackIsing import tfim_magnetization

# Step 0: Set environment variables (before running script)

# On command line or by .env file, you can set the following variables
# QRACK_DISABLE_QUNIT_FIDELITY_GUARD=1: For large circuits, automatically "elide," for approximation
# QRACK_NONCLIFFORD_ROUNDING_THRESHOLD=[0 to 1]: Sacrifices near-Clifford accuracy to reduce overhead
# QRACK_QUNIT_SEPARABILITY_THRESHOLD=[0 to 1]: Rounds to separable states more aggressively
# QRACK_QBDT_SEPARABILITY_THRESHOLD=[0 to 0.5]: Rounding for QBDD, if actually used

# Step 1: Define the molecule (Hydrogen, Helium, Lithium, Carbon, Nitrogen, Oxygen)

basis = "sto-3g"  # Minimal Basis Set
# basis = '6-31g'  # Larger basis set
# basis = 'cc-pVDZ' # Even larger basis set!
multiplicity = 1  # singlet, closed shell, all electrons are paired (neutral molecules with full valence)
# multiplicity = 2  # doublet, one unpaired electron (ex.: OH- radical)
# multiplicity = 3  # triplet, two unpaired electrons (ex.: O2)
charge = 0  # Excess +/- elementary charge, beyond multiplicity

# Hydrogen (and lighter):

geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))]  # H2 Molecule

# Helium (and lighter):

# geometry = [('He', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 7.74))]  # HeH Molecule

# Lithium (and lighter):

# geometry = [('Li', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 15.9))]  # LiH Molecule

# Carbon (and lighter):

# Methane (CH4):
# geometry = [
#     ('C', (0.0000, 0.0000, 0.0000)),  # Central carbon
#     ('H', (1.0900, 0.0000, 0.0000)),  # Hydrogen 1
#     ('H', (-0.3630, 1.0270, 0.0000)),  # Hydrogen 2
#     ('H', (-0.3630, -0.5130, 0.8890)),  # Hydrogen 3
#     ('H', (-0.3630, -0.5130, -0.8890))  # Hydrogen 4
# ]

# Nitrogen (and lighter):

# geometry = [('N', (0.0, 0.0, 0.0)), ('N', (0.0, 0.0, 10.9))]  # N2 Molecule

# Ammonia:
# geometry = [
#     ('N', (0.0000, 0.0000, 0.0000)),  # Nitrogen at center
#     ('H', (0.9400, 0.0000, -0.3200)),  # Hydrogen 1
#     ('H', (-0.4700, 0.8130, -0.3200)), # Hydrogen 2
#     ('H', (-0.4700, -0.8130, -0.3200)) # Hydrogen 3
# ]

# Oxygen (and lighter):

# geometry = [('O', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 9.6))]  # OH- Radical
# geometry = [('O', (0.0000, 0.0000, 0.0000)), ('H', (0.7586, 0.0000, 0.5043)),  ('H', (-0.7586, 0.0000, 0.5043))]  # H2O Molecule
# geometry = [('C', (0.0000, 0.0000, 0.0000)), ('O', (0.0000, 0.0000, 1.128))]  # CO Molecule
# geometry = [('C', (0.0000, 0.0000, 0.0000)), ('O', (0.0000, 0.0000, 1.16)), ('O', (0.0000, 0.0000, -1.16))]  # CO2 Molecule
# geometry = [('O', (0.0, 0.0, 0.0)), ('N', (0.0, 0.0, 1.55))]  # NO Molecule
# geometry = [('O', (0.0, 0.0, 0.0)), ('N', (0.0, 0.0, 11.5))]  # NO+ Radical

# Nitrogen dioxide (toxic pollutant)
# geometry = [
#     ('N', (0.0000,  0.0000,  0.0000)),
#     ('O', (1.2000,  0.0000,  0.0000)),
#     ('O', (-1.2000 * np.cos(np.deg2rad(134)), 1.2000 * np.sin(np.deg2rad(134)), 0.0000))
# ]
# geometry = [('N', (0.0000, 0.0000, 0.0000)), ('N', (1.128, 0.0000, 0.0000)), ('O', (2.241, 0.0000, 0.0000))]  # N2O Molecule

# Ozone:
# geometry = [
#     ('O', (0.0000,  0.0000,  0.0000)),  # Central oxygen
#     ('O', (1.2780,  0.0000,  0.0000)),  # One oxygen
#     ('O', (-1.2780 * np.cos(np.deg2rad(117)), 1.2780 * np.sin(np.deg2rad(117)), 0.0000))  # The other oxygen
# ]

# H3O+ (Hydronium Ion):
# bond_length = 10.3  # Scaled bond length (1.03 Å in script units)
# bond_angle = 113.0  # Approximate bond angle of H3O+
# Convert angles to radians
# theta = np.deg2rad(bond_angle)
# Final geometry list
# geometry = [
#     ('O', (0.0, 0.0, 0.0)),
#     ('H', (bond_length, 0.0, 0.0)),
#     ('H', (-bond_length * np.cos(theta), bond_length * np.sin(theta), 0.0)),
#     ('H', (-bond_length * np.cos(theta), -bond_length * np.sin(theta), 0.0))
# ]

# Carbonate ion (CO3--):
# geometry = [
#     ('C', (0.0000, 0.0000, 0.0000)),  # Carbon at center
#     ('O', (1.2900, 0.0000, 0.0000)),  # Oxygen 1
#     ('O', (-0.6450, 1.1180, 0.0000)),  # Oxygen 2
#     ('O', (-0.6450, -1.1180, 0.0000))  # Oxygen 3
# ]

# Bicarbonate ion (HCO3-):
# geometry = [
#     ('C', (0.0000, 0.0000, 0.0000)),  # Carbon center
#     ('O', (1.2200, 0.0000, 0.0000)),  # Oxygen (C=O)
#     ('O', (-0.6100, 1.0550, 0.0000)),  # Oxygen (-O⁻)
#     ('O', (-0.6100, -1.0550, 0.0000)),  # Oxygen (OH)
#     ('H', (-1.2200, -1.0550, 0.0000))  # Hydrogen in OH group
# ]

# Sulfur chemistry:

# Hydrogen sulfide (rotten egg smell, major biologic sulfur compound):
# geometry = [
#     ('S', (0.0000, 0.0000, 0.0000)),
#     ('H', (1.3400, 0.0000, 0.0000)),
#     ('H', (-1.3400 * np.cos(np.deg2rad(92)), 1.3400 * np.sin(np.deg2rad(92)), 0.0000))
# ]

# Sulfur dioxide (major volcanic gas and pollutant):
# geometry = [
#     ('S', (0.0000, 0.0000, 0.0000)),  # Sulfur at center
#     ('O', (1.4300, 0.0000, 0.0000)),  # Oxygen 1
#     ('O', (-1.4300 * np.cos(np.deg2rad(119)), 1.4300 * np.sin(np.deg2rad(119)), 0.0000))  # Oxygen 2
# ]

# Sulfur trioxide (key in acid rain, forms H₂SO₄):
# geometry = [
#     ('S', (0.0000, 0.0000, 0.0000)),
#     ('O', (1.4200, 0.0000, 0.0000)),
#     ('O', (-0.7100, 1.2290, 0.0000)),
#     ('O', (-0.7100, -1.2290, 0.0000))
# ]

# Sulfate ion (SO4--, major oceanic anion, ionically bonds to Mg++):
# geometry = [
#     ('S', (0.0000, 0.0000, 0.0000)),
#     ('O', (1.4900, 0.0000, 0.0000)),
#     ('O', (-0.7450, 1.2900, 0.0000)),
#     ('O', (-0.7450, -1.2900, 0.0000)),
#     ('O', (0.0000, 0.0000, 1.4900))
# ]

# Oceanic electrolytes (consider isolating cations and anions as single atoms with excess charge):

# Sodium chloride:
# geometry = [
#     ('Na', (0.0000, 0.0000, 0.0000)),
#     ('Cl', (2.3600, 0.0000, 0.0000))
# ]

# Potassium chloride (biologically important):
# geometry = [
#     ('K', (0.0000, 0.0000, 0.0000)),
#     ('Cl', (2.6700, 0.0000, 0.0000))
# ]

# Calcium chloride:
# geometry = [
#     ('Ca', (0.0000, 0.0000, 0.0000)),
#     ('Cl', (2.7800, 0.0000, 0.0000)),
#     ('Cl', (-2.7800, 0.0000, 0.0000))
# ]

# Diatomic halogens:

# Fluorine gas (F2)
# geometry = [('F', (0.0000, 0.0000, 0.0000)), ('F', (1.4200, 0.0000, 0.0000))]

# Chlorine gas (Cl2)
# geometry = [('Cl', (0.0000, 0.0000, 0.0000)), ('Cl', (1.9900, 0.0000, 0.0000))]

# Silicon dioxide (quartz, sand, granite):
# geometry = [
#     ('Si', (0.0000, 0.0000, 0.0000)),  # Silicon at center
#     ('O', (1.6200, 0.0000, 0.0000)),  # Oxygen 1
#     ('O', (-1.6200, 0.0000, 0.0000))  # Oxygen 2
# ]

# The above are major atmospheric, oceanic, and soil components on Earth.
# Proper organic chemistry is beyond the scope of this script,
# but we give a memorable token example of a carbon ring.

# Benzene (C6H6)

# Define bond lengths (in angstroms, converted to script units)
# C_C = 13.9  # Carbon-carbon bond (1.39 Å)
# C_H = 10.9  # Carbon-hydrogen bond (1.09 Å)

# Angle of 120° between C-C bonds in the hexagonal ring
# theta = np.deg2rad(120)

# Define carbon positions (hexagonal ring)
# geometry = [
#     ('C', (C_C, 0.0, 0.0)),  # First carbon at x-axis
#     ('C', (C_C * np.cos(theta), C_C * np.sin(theta), 0.0)),
#     ('C', (-C_C * np.cos(theta), C_C * np.sin(theta), 0.0)),
#     ('C', (-C_C, 0.0, 0.0)),
#     ('C', (-C_C * np.cos(theta), -C_C * np.sin(theta), 0.0)),
#     ('C', (C_C * np.cos(theta), -C_C * np.sin(theta), 0.0))
# ]

# Define hydrogen positions (bonded to carbons)
# for i in range(6):
#     x, y, z = geometry[i][1]  # Get carbon position
#     hydrogen_x = x + (C_H * (x / C_C))  # Extend outward along C-C axis
#     hydrogen_y = y + (C_H * (y / C_C))
#     hydrogen_z = z  # Planar
#     geometry.append(('H', (hydrogen_x, hydrogen_y, hydrogen_z)))

# Now, `geometry` contains all 6 carbons and 6 hydrogens!

# Step 2: Compute the Molecular Hamiltonian
molecule = of.MolecularData(geometry, basis, multiplicity, charge)
molecule = run_pyscf(molecule, run_scf=True, run_fci=True)
fermionic_hamiltonian = molecule.get_molecular_hamiltonian()
n_qubits = molecule.n_qubits  # Auto-detect qubit count
print(str(n_qubits) + " qubits...")

# Step 3: Convert to Qubit Hamiltonian (Jordan-Wigner)
qubit_hamiltonian = jordan_wigner(fermionic_hamiltonian)

# Step 4: Setup localized TFIM from Hamiltonian

def estimate_local_parameters(qubit_hamiltonian, n_qubits):
    z = np.zeros(n_qubits, dtype=int)
    J = np.zeros(n_qubits)
    h = np.zeros(n_qubits)

    for term, coeff in qubit_hamiltonian.terms.items():
        if len(term) == 1:
            q, pauli = term[0]
            if pauli == "X":
                h[q] += np.abs(coeff)
            elif pauli == "Y":
                h[q] += 0.5 * np.abs(coeff)
        elif len(term) == 2:
            (q1, p1), (q2, p2) = term
            if {p1, p2} <= {"Z"}:
                J[q1] -= np.abs(coeff) / 2
                J[q2] -= np.abs(coeff) / 2
                z[q1] += 1
                z[q2] += 1
            else:
                h[q1] += 0.25 * np.abs(coeff)
                h[q2] += 0.25 * np.abs(coeff)
    return z, J, h

def tfim_ground_state_angles(n_qubits, J_vec, h_vec, z_vec, t=10.0):
    ry_angles = np.zeros(n_qubits)
    for i in range(n_qubits):
        J=J_vec[i]
        h=h_vec[i]
        z=z_vec[i]
        ry_angles[i] = np.arcsin(max(min(1, abs(h) / (z * J)) if np.isclose(z * J, 0) else (1 if J > 0 else -1), -1))
    return ry_angles

def hybrid_tfim_vqe(qubit_hamiltonian, n_qubits, dev=None):
    """
    Estimate energy from TFIM-predicted RY angles.
    """
    z, J, h = estimate_local_parameters(qubit_hamiltonian, n_qubits)
    theta = tfim_ground_state_angles(n_qubits, J, h, z)

    if dev is None:
        dev = qml.device("default.qubit", wires=n_qubits)

    coeffs = []
    observables = []
    for term, coefficient in qubit_hamiltonian.terms.items():
        pauli_operators = []
        for qubit_idx, pauli in term:
            if pauli == "X":
                pauli_operators.append(qml.PauliX(qubit_idx))
            elif pauli == "Y":
                pauli_operators.append(qml.PauliY(qubit_idx))
            elif pauli == "Z":
                pauli_operators.append(qml.PauliZ(qubit_idx))
        if pauli_operators:
            observable = pauli_operators[0]
            for op in pauli_operators[1:]:
                observable = observable @ op
            observables.append(observable)
        else:
            observables.append(qml.Identity(0))  # Default identity if no operators
        coeffs.append(qml.numpy.array(coefficient, requires_grad=False))

    hamiltonian = qml.Hamiltonian(coeffs, observables)

    @qml.qnode(dev)
    def circuit(delta):
        for i in range(n_qubits):
            qml.RY(theta[i] + delta[i], wires=i)
        for i in range(n_qubits - 1):
            qml.CZ(wires=[i, i + 1])
        return qml.expval(hamiltonian)

    return circuit

# Step 5: Setup Qrack simulator and calculate energy expectation value
dev = qml.device("qrack.simulator", wires=n_qubits)
circuit = hybrid_tfim_vqe(qubit_hamiltonian, n_qubits)
weights = np.zeros(n_qubits)
min_energy = circuit(weights)
for i in range(n_qubits):
    w = 0
    energy = circuit(weights)
    if energy < min_energy:
        min_energy = energy

    weights[i] = np.pi / 2
    energy = circuit(weights)
    if energy < min_energy:
        min_energy = energy
        w = np.pi / 2

    weights[i] = -np.pi / 2
    energy = circuit(weights)
    if energy < min_energy:
        min_energy = energy
        w = -np.pi / 2

    weights[i] = w

    print(f"Step {i+1}: Energy = {min_energy}")

print(f"Optimized Ground State Energy: {min_energy} Ha")
print("Optimized parameters:")
print(weights)
