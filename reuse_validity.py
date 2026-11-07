"""Generator-independent, fixed-ACTIVE-width qubit-reuse plan checker.

Designed for the static-circuit subset used by the fixed-width GidNET repair
experiments (Qiskit 0.45.x): unitary gates, barriers and terminal measurements.

A plan is a sequence of ordered chains of ACTIVE logical-qubit indices, e.g.
    ((0, 4), (1,), (2, 3))
Inactive declared qubits are NOT listed and are accounted for as singleton
chains when computing the full declared width.

The static checker verifies structural validity, individually admissible reuse
boundaries, and joint acyclicity of the ORIGINAL dependency DAG with all reuse
edges added. It does NOT prove a concrete compiled dynamic circuit preserves
quantum/classical semantics. Test the compiler/reset insertion separately.

No imports from GidNET; safe to use repeatedly for repair neighborhoods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Iterable, Mapping, Optional, Sequence

import networkx as nx


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    reason: str
    active_width: Optional[int] = None
    full_width: Optional[int] = None
    num_reuse_boundaries: int = 0
    boundaries: tuple[tuple[int, int], ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def legal(self) -> bool:
        """Compatibility alias for Experiment 3's result['legal']."""
        return self.is_valid

    def as_dict(self) -> dict[str, Any]:
        return {
            'legal': self.is_valid,
            'is_valid': self.is_valid,
            'reason': self.reason,
            'active_width': self.active_width,
            'full_width': self.full_width,
            'full_chain_count': self.full_width,
            'num_reuse_boundaries': self.num_reuse_boundaries,
            'boundaries': self.boundaries,
            **dict(self.details),
        }


def boundaries_from_chains(chains: Sequence[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    """Derive directed reuse edges: terminal of a -> input of b."""
    return tuple((int(a), int(b)) for chain in chains for a, b in zip(chain, chain[1:]))


def active_qubits_from_circuit(circuit: Any) -> frozenset[int]:
    """Match Experiment 3's active-qubit convention (ignore measure/barrier)."""
    active: set[int] = set()
    for item in circuit.data:
        if item.operation.name in {'measure', 'barrier'}:
            continue
        for bit in item.qubits:
            active.add(circuit.find_bit(bit).index)
    return frozenset(active)


def _validate_supported_input(circuit: Any) -> Optional[ValidationResult]:
    """Fail closed for constructs the current compilation model does not justify.

    Gate is Qiskit's unitary-gate abstraction; measurements are allowed only at
    the end of each measured qubit's lifetime. We deliberately do not pretend
    to validate general mid-circuit control flow or arbitrary instructions.
    """
    try:
        from qiskit.circuit import Gate
    except ImportError as exc:
        raise ImportError('Qiskit is required to validate QuantumCircuit inputs.') from exc

    measured_qubits: set[int] = set()
    for position, item in enumerate(circuit.data):
        op = item.operation
        name = op.name
        qinds = {circuit.find_bit(q).index for q in item.qubits}
        if getattr(op, 'condition', None) is not None:
            return ValidationResult(False, 'unsupported_classical_condition',
                                    details={'operation_index': position, 'name': name})
        if name == 'barrier':
            continue
        if name == 'measure':
            if len(qinds) != 1 or len(item.clbits) != 1:
                return ValidationResult(False, 'unsupported_measurement',
                                        details={'operation_index': position})
            measured_qubits.update(qinds)
            continue
        if measured_qubits & qinds:
            return ValidationResult(False, 'mid_circuit_measurement',
                                    details={'operation_index': position,
                                             'qubits': tuple(sorted(measured_qubits & qinds))})
        if not isinstance(op, Gate):
            return ValidationResult(False, 'unsupported_nonunitary_instruction',
                                    details={'operation_index': position, 'name': name})
    return None


def _circuit_to_dependency_graph(circuit: Any, *, omit_final_measurements: bool) -> tuple[nx.DiGraph, dict[int, Any], dict[int, Any]]:
    """Build a dependency DAG via Qiskit's public DAGCircuit APIs.

    Do not use GidNET's candidate_matrix, or its private _multi_graph.
    """
    try:
        from qiskit.converters import circuit_to_dag
    except ImportError as exc:
        raise ImportError('Install Qiskit 0.45.x in the project environment.') from exc

    qc = circuit.copy()
    if omit_final_measurements:
        qc.remove_final_measurements(inplace=True)
    dag = circuit_to_dag(qc)
    for node in tuple(dag.op_nodes()):
        if node.op.name == 'barrier':
            dag.remove_op_node(node)

    g = nx.DiGraph()
    g.add_nodes_from(dag.nodes())
    for source, target, _wire in dag.edges():
        g.add_edge(source, target)

    ins = {i: dag.input_map[bit] for i, bit in enumerate(dag.qubits)}
    outs = {i: dag.output_map[bit] for i, bit in enumerate(dag.qubits)}
    return g, ins, outs


class ReuseValidityChecker:
    """Cache circuit DAGs once; validate arbitrarily many fixed-width neighbors.

    For tests or research on abstract dependencies, use `from_graphs` to avoid
    requiring Qiskit. Real circuits should use the ordinary constructor.
    """

    def __init__(self, circuit: Any):
        self.source_circuit = circuit
        self.num_qubits = int(circuit.num_qubits)
        self.active_qubits = active_qubits_from_circuit(circuit)
        self.unsupported = _validate_supported_input(circuit)
        self.original_graph, self.input_nodes, self.output_nodes = _circuit_to_dependency_graph(
            circuit, omit_final_measurements=False)
        self.candidate_graph, self.candidate_inputs, self.candidate_outputs = _circuit_to_dependency_graph(
            circuit, omit_final_measurements=True)
        self._check_graph_setup()
        self._reachable_to_output = self._compute_candidate_forbidden()

    @classmethod
    def from_graphs(cls, *, num_qubits: int, active_qubits: Iterable[int],
                    full_graph: nx.DiGraph, full_inputs: Mapping[int, Any],
                    full_outputs: Mapping[int, Any], candidate_graph: Optional[nx.DiGraph] = None,
                    candidate_inputs: Optional[Mapping[int, Any]] = None,
                    candidate_outputs: Optional[Mapping[int, Any]] = None) -> 'ReuseValidityChecker':
        """Synthetic dependency graph constructor, useful for isolated unit tests."""
        obj = cls.__new__(cls)
        obj.source_circuit = None
        obj.num_qubits = num_qubits
        obj.active_qubits = frozenset(active_qubits)
        obj.unsupported = None
        obj.original_graph = full_graph.copy()
        obj.input_nodes = dict(full_inputs)
        obj.output_nodes = dict(full_outputs)
        obj.candidate_graph = (candidate_graph if candidate_graph is not None else full_graph).copy()
        obj.candidate_inputs = dict(candidate_inputs if candidate_inputs is not None else full_inputs)
        obj.candidate_outputs = dict(candidate_outputs if candidate_outputs is not None else full_outputs)
        obj._check_graph_setup()
        obj._reachable_to_output = obj._compute_candidate_forbidden()
        return obj

    def _check_graph_setup(self) -> None:
        expected = set(range(self.num_qubits))
        if (set(self.input_nodes) != expected or set(self.output_nodes) != expected or
                set(self.candidate_inputs) != expected or set(self.candidate_outputs) != expected):
            raise ValueError('DAG input/output maps must include every declared qubit.')
        if not nx.is_directed_acyclic_graph(self.original_graph):
            raise ValueError('Original circuit dependency graph has a cycle.')
        if not nx.is_directed_acyclic_graph(self.candidate_graph):
            raise ValueError('Candidate circuit dependency graph has a cycle.')
        if not self.active_qubits <= expected:
            raise ValueError('active_qubits contains out-of-range qubits.')

    def _compute_candidate_forbidden(self) -> frozenset[tuple[int, int]]:
        """Reuse a->b invalid individually if input(b) reaches output(a).

        This is the same reachability question as GidNET's initial candidate
        matrix complement, but computed independently from the DAG.
        """
        blocked: set[tuple[int, int]] = set()
        for b in range(self.num_qubits):
            descendants = nx.descendants(self.candidate_graph, self.candidate_inputs[b])
            descendants.add(self.candidate_inputs[b])
            for a in range(self.num_qubits):
                if a == b or self.candidate_outputs[a] in descendants:
                    blocked.add((a, b))
        return frozenset(blocked)

    def validate(self, active_chains: Sequence[Sequence[int]], *,
                 target_width: Optional[int] = None) -> ValidationResult:
        """Check a plan on the *active* logical qubits only.

        The `target_width` is active width, matching Experiment 3 normalization.
        All disconnected/inactive declared qubits become singleton chains in
        the full plan, with full width = active width + inactive count.
        """
        try:
            chains = tuple(tuple(chain) for chain in active_chains)
        except (TypeError, ValueError):
            return ValidationResult(False, 'malformed_chains')
        if not chains or any(not chain for chain in chains):
            return ValidationResult(False, 'empty_chain')
        if any((not isinstance(q, Integral)) or isinstance(q, bool)
               for chain in chains for q in chain):
            return ValidationResult(False, 'non_integer_qubit')

        flat = tuple(int(q) for chain in chains for q in chain)
        active_width = len(chains)
        full_width = active_width + self.num_qubits - len(self.active_qubits)
        if len(flat) != len(set(flat)):
            return ValidationResult(False, 'duplicate_logical_qubit', active_width, full_width)
        observed = set(flat)
        if observed != self.active_qubits:
            return ValidationResult(False, 'active_qubit_set_mismatch', active_width, full_width,
                                    details={'missing': tuple(sorted(self.active_qubits - observed)),
                                             'extra': tuple(sorted(observed - self.active_qubits))})
        if target_width is not None and (not isinstance(target_width, Integral) or
                                          int(target_width) != active_width):
            return ValidationResult(False, 'width_mismatch', active_width, full_width,
                                    details={'target_width': target_width})
        if self.unsupported is not None:
            return ValidationResult(False, self.unsupported.reason, active_width, full_width,
                                    details=self.unsupported.details)

        edges = boundaries_from_chains(chains)
        illegal = tuple((a, b) for a, b in edges if (a, b) in self._reachable_to_output)
        if illegal:
            return ValidationResult(False, 'illegal_reuse_boundary', active_width, full_width,
                                    len(edges), edges, {'illegal_boundaries': illegal})

        # A set of separately legal boundaries may still introduce a GLOBAL cycle.
        combined = self.original_graph.copy()
        for a, b in edges:
            combined.add_edge(self.output_nodes[a], self.input_nodes[b])
        if not nx.is_directed_acyclic_graph(combined):
            return ValidationResult(False, 'dependency_cycle', active_width, full_width,
                                    len(edges), edges)

        # This is static graph admissibility, not verified dynamic-circuit semantics.
        return ValidationResult(True, 'ok_static', active_width, full_width,
                                len(edges), edges,
                                {'static_validity_only': True,
                                 'inactive_qubits': tuple(sorted(set(range(self.num_qubits)) - self.active_qubits))})


def validate_reuse_plan(circuit: Any, active_chains: Sequence[Sequence[int]], *,
                        target_width: Optional[int] = None) -> ValidationResult:
    """Convenience API. In large searches, instantiate the checker once instead."""
    return ReuseValidityChecker(circuit).validate(active_chains, target_width=target_width)
