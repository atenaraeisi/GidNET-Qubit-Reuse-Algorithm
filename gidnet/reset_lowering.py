"""
Reset lowering helper for GidNET experiments.

Compatible with Qiskit 0.45.x.

This module replaces Qiskit's native ``reset`` instruction with the
measurement-based active-reset equivalent:

    measure q -> c
    x q  if c == 1

If the last operation touching the same qubit is already an unconditional
measurement and its classical result has not been overwritten, that existing
measurement result is reused.  Therefore a GidNET reuse boundary of the form

    measure q -> c
    reset q

becomes

    measure q -> c
    x q if c == 1

instead of adding a duplicate measurement.
"""

import copy
from typing import Dict

from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, Clbit, Qubit


def lower_reset_to_measure_x(circuit: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of ``circuit`` with every native reset lowered.

    Rules
    -----
    1. ``measure q -> c; reset q`` becomes
       ``measure q -> c; x(q).c_if(c, 1)`` when that measurement is still
       the most recent operation touching ``q`` and ``c`` has not been
       overwritten.

    2. A standalone ``reset q`` becomes
       ``measure q -> aux; x(q).c_if(aux, 1)``.

    Notes
    -----
    - The input circuit is not modified.
    - Existing measurements are preserved.
    - No qubits are added or removed.
    - New classical 1-bit registers are created only when a reset does not
      already have a usable preceding measurement.
    - This helper is intended for GidNET benchmark circuits on Qiskit 0.45.x.
    """

    # GidNET circuits are register-based.  Reusing the same register objects
    # also keeps any existing classical conditions valid in Qiskit 0.45.
    lowered = QuantumCircuit(
        *circuit.qregs,
        *circuit.cregs,
        name=circuit.name,
    )

    lowered.global_phase = circuit.global_phase
    lowered.metadata = copy.deepcopy(circuit.metadata)

    # Map:
    #   qubit -> classical bit containing the result of the most recent
    #            valid unconditional measurement of that same qubit.
    last_measurement: Dict[Qubit, Clbit] = {}

    existing_creg_names = {reg.name for reg in lowered.cregs}
    aux_counter = 0

    def new_aux_register() -> ClassicalRegister:
        """Create a unique 1-bit classical register for active reset."""
        nonlocal aux_counter

        while True:
            name = f"reset_aux_{aux_counter}"
            aux_counter += 1

            if name not in existing_creg_names:
                existing_creg_names.add(name)
                reg = ClassicalRegister(1, name)
                lowered.add_register(reg)
                return reg

    def invalidate_clbit(bit: Clbit) -> None:
        """Forget measurements whose stored result has been overwritten."""
        stale_qubits = [
            qubit
            for qubit, measured_bit in last_measurement.items()
            if measured_bit == bit
        ]

        for qubit in stale_qubits:
            del last_measurement[qubit]

    for instruction in circuit.data:
        op = instruction.operation
        qargs = list(instruction.qubits)
        cargs = list(instruction.clbits)

        # --------------------------------------------------------------
        # RESET
        # --------------------------------------------------------------
        if op.name == "reset":
            if len(qargs) != 1:
                raise ValueError(
                    f"Expected a one-qubit reset, got {len(qargs)} qubits."
                )

            # We only lower ordinary unconditional resets here.
            if getattr(op, "condition", None) is not None:
                raise NotImplementedError(
                    "Conditional reset encountered. "
                    "This helper only lowers unconditional reset instructions."
                )

            qubit = qargs[0]

            # Case 1:
            # The most recent operation on this qubit was an unconditional
            # measurement and its result has not been overwritten.
            condition_bit = last_measurement.get(qubit)

            # Case 2:
            # No reusable measurement exists, so implement reset by adding one.
            if condition_bit is None:
                aux_reg = new_aux_register()
                condition_bit = aux_reg[0]
                lowered.measure(qubit, condition_bit)

            # Qiskit 0.45 supports classical conditions via InstructionSet.c_if.
            lowered.x(qubit).c_if(condition_bit, 1)

            # The conditional X is now the most recent operation touching q,
            # so an older measurement must not be reused for a later reset.
            last_measurement.pop(qubit, None)
            continue

        # --------------------------------------------------------------
        # ALL NON-RESET OPERATIONS
        # --------------------------------------------------------------

        # Preserve the original operation exactly.
        lowered.append(op, qargs, cargs)

        is_plain_single_qubit_measurement = (
            op.name == "measure"
            and len(qargs) == 1
            and len(cargs) == 1
            and getattr(op, "condition", None) is None
        )

        if is_plain_single_qubit_measurement:
            qubit = qargs[0]
            clbit = cargs[0]

            # If some earlier measurement result used this same classical bit,
            # the new measurement overwrites it.
            invalidate_clbit(clbit)

            # This measurement is now reusable by a following reset on q,
            # provided nothing else touches q in between.
            last_measurement[qubit] = clbit

        else:
            # Any other operation touching a qubit invalidates a previously
            # stored measurement for that qubit.  This is deliberately
            # conservative and avoids reusing stale measurement results.
            for qubit in qargs:
                last_measurement.pop(qubit, None)

            # Conservatively treat explicit classical operands as writes that
            # may invalidate an old stored measurement result.
            for clbit in cargs:
                invalidate_clbit(clbit)

    return lowered


def count_reset_lowering_changes(
    original: QuantumCircuit,
    lowered: QuantumCircuit,
) -> dict:
    """Return a small diagnostic summary useful in benchmark scripts."""
    original_ops = original.count_ops()
    lowered_ops = lowered.count_ops()

    return {
        "original_resets": int(original_ops.get("reset", 0)),
        "lowered_resets": int(lowered_ops.get("reset", 0)),
        "original_measurements": int(original_ops.get("measure", 0)),
        "lowered_measurements": int(lowered_ops.get("measure", 0)),
        "conditional_x_added_estimate": int(original_ops.get("reset", 0)),
        "same_num_qubits": original.num_qubits == lowered.num_qubits,
    }
