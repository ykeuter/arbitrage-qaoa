#%%
import numpy as np
from qiskit import BasicAer as Aer
from qiskit import QuantumRegister, execute
from qiskit.quantum_info import Pauli
from qiskit.aqua import get_aer_backend
from qiskit.aqua.components.initial_states import Custom
from scipy.optimize import minimize
from collections import OrderedDict
from qiskit.aqua.operators import WeightedPauliOperator
from docplex.mp.model import Model
from qiskit.aqua.translators.ising import docplex
from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit.aqua import QuantumInstance
from qiskit import BasicAer, Aer
from qiskit.aqua.algorithms import QAOA
from qiskit.aqua.components.optimizers import COBYLA
#%%
def get_cost_hamiltonian(rates, m1=1., m2=1.): # ordered dict
    mdl = Model(name="arbitrage")
    x = {(a, b): mdl.binary_var(name=a + b) for (a, b) in rates.keys()}
    fun = mdl.sum(np.log(r) * x[k] for k, r in rates.items())

    assets = set(x for x, y in rates.keys())
    for a in assets:
        for x1, y1 in rates.keys():
            if x1 != a: continue
            for x2, y2 in rates.keys():
                if x2 != a: continue
                fun -= m1 * x[(x1, y1)] * x[(x2, y2)]
            for x2, y2 in rates.keys():
                if y2 != a: continue
                fun += 2. * m1 * x[(x1, y1)] * x[(x2, y2)]
        for x1, y1 in rates.keys():
            if y1 != a: continue
            for x2, y2 in rates.keys():
                if y2 != a: continue
                fun -= m1 * x[(x1, y1)] * x[(x2, y2)]
    for a in assets:
        for x1, y1 in rates.keys():
            if x1 != a: continue
            fun += m2 * x[(x1, y1)]
            for x2, y2 in rates.keys():
                if x2 != a: continue
                fun -= m2 * x[(x1, y1)] * x[(x2, y2)]
    mdl.maximize(fun)
    operator, _ = docplex.get_qubitops(mdl)
    return operator

#%%
rates = OrderedDict((
    # (("USD", "EUR"), 1.),
    # (("EUR", "USD"), 1.1),
    # (("GBP", "EUR"), 1.),
    # (("EUR", "GBP"), 1.3),
    # no arb
    # (("EUR", "GBP"), 0.88),
    # (("GBP", "EUR"), 1.13),
    # (("EUR", "CAD"), 1.47),
    # (("CAD", "EUR"), 0.68),
    # (("GBP", "CAD"), 1.65),
    # (("CAD", "GBP"), 0.6),
    # GBP -> EUR -> CAD -> GBP makes you money. The other cycles do not.
    (("EUR", "GBP"), 0.88),
    (("GBP", "EUR"), 1.13),
    (("EUR", "CAD"), 1.58),
    (("CAD", "EUR"), 0.61),
    (("GBP", "CAD"), 1.65),
    (("CAD", "GBP"), 0.6),
))
op = get_cost_hamiltonian(rates, 1, 1)
#%%
ee = ExactEigensolver(op)
result = ee.run()
print(bin(result['eigvecs'][0].argmax()))
#%%

p = 1
optimizer = COBYLA()
qaoa = QAOA(op, optimizer, p)
backend = BasicAer.get_backend('statevector_simulator')
quantum_instance = QuantumInstance(backend)
r2 = qaoa.run(quantum_instance)
print(bin(np.absolute(r2['eigvecs'][0]).argmax()))
