#%%
import qiskit
import numpy as np
from qiskit.aqua import Operator
from qiskit.quantum_info import Pauli

#%%
# Functions useful if you're using Qiskit
def pauli_x(qubit, coeff, n_qubits=4):
    eye = np.eye((n_qubits))
    return Operator([[coeff, Pauli(np.zeros(n_qubits), eye[qubit])]])

def pauli_z(qubit, coeff, n_qubits=4):
    eye = np.eye((n_qubits))
    return Operator([[coeff, Pauli(eye[qubit], np.zeros(n_qubits))]])

def product_pauli_z(q1, q2, coeff, n_qubits=4):
    eye = np.eye((n_qubits))
    return Operator([[coeff, Pauli(eye[q1], np.zeros(n_qubits)) * 
                      Pauli(eye[q2], np.zeros(n_qubits))]])

#%%
p = pauli_x(1, 8)
p.to_matrix()

#%%
p.print_operators()

#%%
def get_hamiltonian(rates, m1, m2): # ordered dict
    assets = set(k for k, v in rates.keys())
    operators = []
    for i, r in enumerate(rates):
        operators.append(pauli_z(log(r), i))
    for a in assets:
        for i, (x1, y1) in rates.keys:
            if x1 != a: continue
            for j, (x2, y2) in rates.keys:
                if x2 != a: continue
                operators.append(pauli_z(m1, i, j))
            for j, (x2, y2) in rates.keys:
                if y2 != a: continue
                operators.append(pauli_z(-2 * m1, i, j))
        for i, (x1, y1) in rates.keys:
            if y1 != a: continue
            for j, (x2, y2) in rates.keys:
                if y2 != a: continue
                operators.append(pauli_z(m1, i, j))
    for a in assets:
        for i, (x1, y1) in rates.keys:
            if x1 != a: continue
            operators.append(pauli_z(-m2, i, j))
            for j, (x2, y2) in rates.keys:
                if x2 != a: continue
                operators.append(pauli_z(m2, i, j))
        
    
        



