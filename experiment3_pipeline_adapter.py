"""
Experiment 4A adapter matching the physical compiler helpers used in Experiment 3.

Expected environment: Qiskit 0.45.3, matching the saved Experiment3_B1 validation notebook.
Place this file beside experiment4A_physical_value.py and run the commands in
RUN_EXPERIMENT4A_PARITY.txt from the GidNET-Qubit-Reuse-Algorithm project root.
"""
import copy

import qiskit
from qiskit import QuantumCircuit, ClassicalRegister
from gidnet.qubitreuse import GidNET

EXPECTED_QISKIT_VERSION = "0.45.3"
if qiskit.__version__ != EXPECTED_QISKIT_VERSION:
    raise RuntimeError(
        "Experiment-3 parity adapter expects Qiskit "
        f"{EXPECTED_QISKIT_VERSION}, but this Python uses {qiskit.__version__}. "
        "Activate the same Python environment used by Experiment3_B1_Fresh30SeedValidation."
    )


def build_dynamic_circuit(qc, chains):
    """
    Exact Experiment-3 fixed-plan compilation pattern:
      - add inactive declared qubits as singleton chains
      - replace only GidNET's reuse-sequence selector
      - call compile_to_dynamic_circuit(iterations=1, draw=False)

    Note: Experiment 4A canonicalizes chain order before this adapter is called.
    """
    active_chains = [tuple(int(q) for q in chain) for chain in chains]
    used = {q for chain in active_chains for q in chain}
    full_chains = [list(chain) for chain in active_chains]
    full_chains.extend(
        [[q] for q in range(qc.num_qubits) if q not in used]
    )

    worker = GidNET(copy.deepcopy(qc))
    worker.compute_optimized_reuse_sequences = (
        lambda iterations: copy.deepcopy(full_chains)
    )
    return worker.compile_to_dynamic_circuit(iterations=1, draw=False)


def normalize_for_routing(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Verbatim reset-lowering logic from Experiment 3's normalize_active_reset.
    A reset is replaced with a measurement-based conditional X.
    """
    new_qc = QuantumCircuit(*qc.qregs, *qc.cregs)

    last_measure_clbit = {}
    clbit_owner = {}

    total_resets = sum(
        1 for inst in qc.data if inst.operation.name == "reset"
    )

    aux_reg = None
    aux_next = 0

    for item in qc.data:
        instr = item.operation
        qargs = list(item.qubits)
        cargs = list(item.clbits)

        q_indices = [qc.find_bit(q).index for q in qargs]

        if instr.name == "measure":
            q = qargs[0]
            c = cargs[0]
            q_idx = qc.find_bit(q).index
            c_idx = qc.find_bit(c).index

            old_owner = clbit_owner.get(c_idx)
            if old_owner is not None:
                last_measure_clbit.pop(old_owner, None)

            new_qc.append(instr, qargs, cargs)
            new_cbit = new_qc.clbits[c_idx]
            last_measure_clbit[q_idx] = new_cbit
            clbit_owner[c_idx] = q_idx
            continue

        if instr.name == "reset":
            q = qargs[0]
            q_idx = qc.find_bit(q).index

            if q_idx in last_measure_clbit:
                cbit = last_measure_clbit[q_idx]
            else:
                if aux_reg is None:
                    aux_reg = ClassicalRegister(
                        max(total_resets, 1), "reset_aux"
                    )
                    new_qc.add_register(aux_reg)

                cbit = aux_reg[aux_next]
                aux_next += 1
                new_qc.measure(q, cbit)

            new_qc.x(q).c_if(cbit, 1)
            last_measure_clbit.pop(q_idx, None)
            continue

        new_qc.append(instr, qargs, cargs)
        for q_idx in q_indices:
            last_measure_clbit.pop(q_idx, None)

    return new_qc
