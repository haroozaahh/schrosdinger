from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

result = AerSimulator().run(qc, shots=1000).result()
print(result.get_counts())
