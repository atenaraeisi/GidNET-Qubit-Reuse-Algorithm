Reuse audit with a quantum-only correction for GidNET's classical-wire candidate matrix bug.

In the SAME activated Qiskit 0.45.3 virtual environment, from the extracted folder:
  python .\reuse_feasibility_audit_fixed.py --circuits cnt3-5_179.qasm,rd53_311.qasm,sym6_316.qasm --out-dir .\fixed_smoke
  python .\reuse_feasibility_audit_fixed.py --out-dir .\fixed_full

Do not mix results with the previous output files. This script patches only
the loaded GidNET class in memory and leaves qubitreuse.py unchanged.
The fix restricts DAG input/output maps to QuantumCircuit qubits.
All generated plans are checked with the independent static validator;
the matrix must agree pairwise with the validator before mutation.
No routing or model training is performed.
