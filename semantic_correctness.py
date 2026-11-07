"""Fixed-plan GidNET compilation and independent end-to-end semantic auditing.

Scope: Qiskit 0.46-era *static* input circuits with unitary gates, barriers,
terminal measurements, and no classical control. This matches the existing
ReuseValidityChecker supported-input gate. A fixed active-qubit reuse plan is
injected into GidNET's *existing* compilation path; its algorithm is unchanged.

Three separate layers:
  1. graph/width admissibility via reuse_validity.py
  2. exact per-wire operation sequence + reset boundary structural audit
  3. ideal, exhaustive branching simulation of classical output distributions
     on small circuits (not a proof of general channel equivalence).

This module never performs routing and never changes any GidNET source files.
"""
from __future__ import annotations

import copy
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from reuse_validity import ReuseValidityChecker


class SemanticAuditError(ValueError):
    """Input is invalid, unsupported, or a compiled artifact violates invariants."""


class SimulationLimit(RuntimeError):
    """Exact simulation was explicitly limited (not a validation failure)."""


@dataclass
class FixedCompilation:
    circuit: Any
    original: Any
    active_chains: tuple[tuple[int, ...], ...]
    full_chains: tuple[tuple[int, ...], ...]
    original_to_wire: dict[int, int]
    expected_width: int
    reuse_boundaries: int


def _full_chains(original, active_chains):
    active = {int(q) for c in active_chains for q in c}
    full = [tuple(int(q) for q in chain) for chain in active_chains]
    full.extend((q,) for q in range(original.num_qubits) if q not in active)
    return tuple(full)


def compile_from_reuse_plan(original, active_chains: Sequence[Sequence[int]],
                            *, target_width: int | None = None) -> FixedCompilation:
    """Compile this *specific* static-valid plan via GidNET's original pipeline.

    Validate first. `target_width` is active width, not width counting inactive
    declared qubits. Do not call the optimizer and do not monkeypatch globally.
    """
    from gidnet.qubitreuse import GidNET

    checker = ReuseValidityChecker(original)
    chains = tuple(tuple(c) for c in active_chains)
    if target_width is None:
        raise SemanticAuditError('target_width must be explicit for fixed-width repair')
    validity = checker.validate(chains, target_width=target_width)
    if not validity.is_valid:
        raise SemanticAuditError(f'Not a static-valid reuse plan: {validity.reason} {dict(validity.details)}')
    if len(original.qregs) != 1:
        raise SemanticAuditError('GidNET compiler currently assumes a single quantum register')

    full = _full_chains(original, chains)
    mapping = {q: wire for wire, chain in enumerate(full) for q in chain}
    if set(mapping) != set(range(original.num_qubits)):
        raise SemanticAuditError('Incomplete original-to-physical mapping')
    # A multi-qubit gate must never collapse two of its operands onto one wire.
    
    for index, inst in enumerate(original.data):
        # Barriers are directives, not physical multi-qubit gates.
        if inst.operation.name == "barrier":
            continue

        if len(inst.qubits) <= 1:
            continue

        wires = [
            mapping[original.find_bit(q).index]
            for q in inst.qubits
        ]

        if len(wires) != len(set(wires)):
            raise SemanticAuditError(f'Multi-qubit gate collision at original op {index}')

    class _FixedPlanGidNET(GidNET):
        def __init__(self, circuit, fixed_chains):
            super().__init__(circuit)
            self._fixed_chains = [list(c) for c in fixed_chains]

        def compute_optimized_reuse_sequences(self, iterations):
            return copy.deepcopy(self._fixed_chains)

    gidnet = _FixedPlanGidNET(copy.deepcopy(original), full)
    dynamic = gidnet.compile_to_dynamic_circuit(iterations=1, draw=False)
    if dynamic.num_qubits != len(full):
        raise SemanticAuditError(f'Compiled width {dynamic.num_qubits}, expected {len(full)}')
    return FixedCompilation(dynamic, original, chains, full, mapping,
                            len(full), sum(len(c) - 1 for c in full))


def _op_key(circuit, inst, *, remap=None):
    op = inst.operation
    qargs = tuple(circuit.find_bit(q).index for q in inst.qubits)
    if remap is not None:
        qargs = tuple(remap[q] for q in qargs)
    cargs = tuple(circuit.find_bit(c).index for c in inst.clbits)
    # The parameters are not changed by GidNET; represent them identically.
    parameters = tuple(repr(p) for p in op.params)
    return (op.name, qargs, cargs, parameters, repr(getattr(op, 'condition', None)))


def audit_compilation_structure(compilation: FixedCompilation) -> dict:
    """Independently check operation conservation, order, and reset placement.

    For every reused physical wire, the exact mapped per-logical-qubit op trace
    must be concatenated in chain order, with one reset at *every* boundary.
    This catches dropped or misplaced resets even if operation counts match.
    """
    original, dynamic = compilation.original, compilation.circuit
    remap = compilation.original_to_wire
    if dynamic.num_qubits != compilation.expected_width:
        raise SemanticAuditError('compiled_width_mismatch')
    if dynamic.num_clbits != original.num_clbits:
        raise SemanticAuditError('classical_bit_count_changed')

    orig_ops = Counter(_op_key(original, inst, remap=remap)
                       for inst in original.data if inst.operation.name != 'barrier')
    dyn_ops = Counter(_op_key(dynamic, inst)
                      for inst in dynamic.data if inst.operation.name not in {'barrier', 'reset'})
    if orig_ops != dyn_ops:
        missing = list((orig_ops - dyn_ops).items())[:5]
        extra = list((dyn_ops - orig_ops).items())[:5]
        raise SemanticAuditError(f'logical_operations_not_conserved missing={missing} extra={extra}')

    original_traces = defaultdict(list)
    for inst in original.data:
        if inst.operation.name == 'barrier':
            continue
        key = _op_key(original, inst, remap=remap)
        for q in inst.qubits:
            original_traces[original.find_bit(q).index].append(key)

    actual_traces = defaultdict(list)
    for inst in dynamic.data:
        if inst.operation.name == 'barrier':
            continue
        if inst.operation.name == 'reset':
            if len(inst.qubits) != 1:
                raise SemanticAuditError('invalid_reset_arity')
            token = ('__RESET__',)
        else:
            token = _op_key(dynamic, inst)
        for q in inst.qubits:
            actual_traces[dynamic.find_bit(q).index].append(token)

    for wire, chain in enumerate(compilation.full_chains):
        expected = []
        for j, q in enumerate(chain):
            if j:
                expected.append(('__RESET__',))
            expected.extend(original_traces[q])
        actual = actual_traces[wire]
        if expected != actual:
            # Restrict message size (long QASM benchmark circuits).
            idx = next((i for i, (x, y) in enumerate(zip(expected, actual)) if x != y),
                       min(len(expected), len(actual)))
            raise SemanticAuditError(
                f'wire_trace_mismatch wire={wire} first_diff={idx} '
                f'expected_len={len(expected)} actual_len={len(actual)} '
                f'expected_at_diff={expected[idx:idx+1]} actual_at_diff={actual[idx:idx+1]}'
            )

    resets = sum(inst.operation.name == 'reset' for inst in dynamic.data)
    if resets != compilation.reuse_boundaries:
        raise SemanticAuditError(f'reset_count_mismatch {resets} != {compilation.reuse_boundaries}')
    return {
        'status': 'structurally_verified', 'expected_width': compilation.expected_width,
        'actual_width': dynamic.num_qubits, 'original_operations': sum(orig_ops.values()),
        'preserved_original_operations': sum(dyn_ops.values()),
        'expected_resets': compilation.reuse_boundaries, 'actual_resets': resets,
        'num_measurements': sum(inst.operation.name == 'measure' for inst in dynamic.data),
    }


def _condition_holds(circuit, condition, cbits):
    if condition is None:
        return True
    if not isinstance(condition, tuple) or len(condition) != 2:
        raise SemanticAuditError(f'Unsupported classical condition: {condition!r}')
    classical_operand, expected = condition
    if hasattr(classical_operand, 'bits') or hasattr(classical_operand, '__iter__'):
        try:
            bits = list(classical_operand)
        except TypeError:
            bits = [classical_operand]
    else:
        bits = [classical_operand]
    value = sum(int(cbits[circuit.find_bit(bit).index]) << i for i, bit in enumerate(bits))
    return value == int(expected)


def _measurement_split(state, qubit):
    """Return nonzero (probability, normalized projected state, outcome) branches."""
    mask = 1 << qubit
    index = np.arange(state.size)
    ones = (index & mask) != 0
    for outcome, keep in ((0, ~ones), (1, ones)):
        p = float(np.vdot(state[keep], state[keep]).real)
        if p > 1e-13:
            projected = state.copy()
            projected[~keep] = 0
            projected /= np.sqrt(p)
            yield p, projected, outcome


def exact_classical_distribution(circuit, *, max_qubits=7, max_branches=4096,
                                 max_operations=250, output_clbits=None) -> dict[str, float]:
    """Ideal exact branch simulation supporting gates, measurement, native reset,
    and register/Clbit-controlled gates from reset lowering.

    This is independent of Aer and routing. Measurements branch explicitly,
    reset is a nonselective measurement followed by a conditional X. Supports
    fixed computational-basis |0...0> inputs (the source benchmark convention).
    Treat an unsupported instruction as an ERROR, never a silent skip.
    """
    from qiskit.quantum_info import Statevector
    n = circuit.num_qubits
    if n > max_qubits:
        raise SimulationLimit(f'{n} qubits > max_qubits={max_qubits}')
    if len(circuit.data) > max_operations:
        raise SimulationLimit(f'{len(circuit.data)} operations > max_operations={max_operations}')
    if output_clbits is None:
        output_clbits = circuit.num_clbits
    if output_clbits > circuit.num_clbits:
        raise SemanticAuditError('output_clbits exceeds available classical bits')

    initial = np.zeros(1 << n, dtype=complex)
    initial[0] = 1
    branches = [(1.0, initial, (0,) * circuit.num_clbits)]
    for index, inst in enumerate(circuit.data):
        op = inst.operation
        name = op.name
        if name == 'barrier':
            continue
        qinds = [circuit.find_bit(q).index for q in inst.qubits]
        cinds = [circuit.find_bit(c).index for c in inst.clbits]
        next_branches = []
        for weight, state, cbits in branches:
            condition = getattr(op, 'condition', None)
            if not _condition_holds(circuit, condition, cbits):
                next_branches.append((weight, state, cbits))
                continue
            if name == 'measure':
                if len(qinds) != 1 or len(cinds) != 1:
                    raise SemanticAuditError('unsupported_measure_arity')
                for p, projected, outcome in _measurement_split(state, qinds[0]):
                    updated = list(cbits)
                    updated[cinds[0]] = outcome
                    next_branches.append((weight * p, projected, tuple(updated)))
            elif name == 'reset':
                if len(qinds) != 1:
                    raise SemanticAuditError('unsupported_reset_arity')
                for p, projected, outcome in _measurement_split(state, qinds[0]):
                    if outcome:
                        ids = np.arange(projected.size) ^ (1 << qinds[0])
                        projected = projected[ids]
                    next_branches.append((weight * p, projected, cbits))
            else:
                try:
                    # Classical condition is evaluated above. Evolve the underlying
                    # *unconditioned* unitary, as Statevector cannot interpret
                    # classical conditions itself.
                    unitary_op = op
                    if condition is not None:
                        unitary_op = op.to_mutable() if hasattr(op, 'to_mutable') else copy.deepcopy(op)
                        unitary_op.condition = None
                    updated_state = Statevector(state).evolve(unitary_op, qargs=qinds).data
                except Exception as exc:
                    raise SemanticAuditError(
                        f'Unsupported gate at instruction {index}: {name}, {exc}'
                    ) from exc
                next_branches.append((weight, np.asarray(updated_state), cbits))
        branches = next_branches
        if len(branches) > max_branches:
            raise SimulationLimit(f'branch count {len(branches)} > max_branches={max_branches}')

    distribution = defaultdict(float)
    for weight, _state, cbits in branches:
        value = sum(int(cbits[i]) << i for i in range(output_clbits))
        key = format(value, f'0{max(output_clbits, 1)}b')
        distribution[key] += float(weight)
    total = sum(distribution.values())
    if not np.isclose(total, 1.0, atol=1e-8):
        raise SemanticAuditError(f'Simulator probabilities sum to {total}, expected 1')
    return dict(sorted(distribution.items()))


def compare_exact_semantics(original, compiled, *, max_qubits=7,
                            max_branches=4096, max_operations=250,
                            tolerance=1e-9) -> dict:
    """Compare exact ideal classical bit distributions (not full quantum channels)."""
    if original.num_clbits == 0 or not any(x.operation.name == 'measure' for x in original.data):
        raise SemanticAuditError('Need observable classical measurements for semantic comparison')
    kwargs = dict(max_qubits=max_qubits, max_branches=max_branches,
                  max_operations=max_operations, output_clbits=original.num_clbits)
    p = exact_classical_distribution(original, **kwargs)
    q = exact_classical_distribution(compiled, **kwargs)
    keys = set(p) | set(q)
    tvd = 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in keys)
    return {'equal': bool(tvd <= tolerance), 'total_variation_distance': tvd,
            'original_distribution': p, 'compiled_distribution': q,
            'tolerance': tolerance, 'semantics_scope': 'ideal classical output from |0...0> input'}


def compare_aer_counts(original, compiled, *, shots=16384, seed=17,
                       max_tvd=0.055, original_clbits=None) -> dict:
    """Optional independent Aer shot check (sampling noise: not exact/proof)."""
    from qiskit_aer import AerSimulator
    if original_clbits is None:
        original_clbits = original.num_clbits
    sim = AerSimulator(method='automatic')
    c1 = sim.run(original, shots=shots, seed_simulator=seed).result().get_counts()
    c2 = sim.run(compiled, shots=shots, seed_simulator=seed + 1).result().get_counts()

    def mapped(counts):
        d = defaultdict(float)
        for key, count in counts.items():
            value = int(key.replace(' ', ''), 2) & ((1 << original_clbits) - 1)
            d[format(value, f'0{max(original_clbits, 1)}b')] += count / shots
        return dict(d)

    p, q = mapped(c1), mapped(c2)
    tvd = .5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q))
    return {'equal_within_sampling_tolerance': tvd <= max_tvd, 'tvd': tvd,
            'shots_per_circuit': shots, 'max_tvd': max_tvd}
