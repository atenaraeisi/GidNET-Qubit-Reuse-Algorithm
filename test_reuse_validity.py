"""Run: python -m pytest -q test_reuse_validity.py

Tests without Qiskit run on synthetic dependency DAGs. The Qiskit integration
smoke test is automatically skipped when Qiskit is not installed.
"""

import networkx as nx
import pytest

from reuse_validity import ReuseValidityChecker, boundaries_from_chains


def toy_checker(interactions=((0, 1), (2, 3))):
    """Four independent input/output wires with two interaction gates.

    Each interaction gate lies on both corresponding qubit lifetimes.
    A reuse edge out(0)->in(2) and out(3)->in(1) is individually legal,
    but together they make a dependency cycle between the two gates.
    """
    g = nx.DiGraph()
    inputs = {i: ('in', i) for i in range(4)}
    outputs = {i: ('out', i) for i in range(4)}
    for i in range(4):
        g.add_edge(inputs[i], outputs[i])
    for pair in interactions:
        gate = ('gate', pair)
        for i in pair:
            g.remove_edge(inputs[i], outputs[i])
            g.add_edge(inputs[i], gate)
            g.add_edge(gate, outputs[i])
    return ReuseValidityChecker.from_graphs(num_qubits=4, active_qubits=set(range(4)),
                                           full_graph=g, full_inputs=inputs,
                                           full_outputs=outputs)


def test_pair_construction():
    assert boundaries_from_chains([[0, 2, 3], [1]]) == ((0, 2), (2, 3))


def test_valid_same_width_plan():
    c = toy_checker()
    res = c.validate([[0, 2], [1], [3]], target_width=3)
    assert res.is_valid and res.reason == 'ok_static'
    assert res.active_width == 3
    assert res.full_width == 3
    assert res.num_reuse_boundaries == 1
    assert res.as_dict()['legal'] is True


def test_rejects_invalid_structure():
    c = toy_checker()
    assert c.validate([[0, 1], [1], [2, 3]]).reason == 'duplicate_logical_qubit'
    assert c.validate([[0, 2], [1]]).reason == 'active_qubit_set_mismatch'
    assert c.validate([[0, 2], [], [1], [3]]).reason == 'empty_chain'
    assert c.validate([[0, 2], [1], [3]], target_width=2).reason == 'width_mismatch'
    assert c.validate([[0, 2], [1], ['3']]).reason == 'non_integer_qubit'


def test_rejects_individually_illegal_boundary():
    c = toy_checker()
    # out(0)->in(1) conflicts with joint gate(0,1) causality.
    res = c.validate([[0, 1], [2], [3]])
    assert not res.is_valid
    assert res.reason == 'illegal_reuse_boundary'
    assert (0, 1) in res.details['illegal_boundaries']


def test_pairwise_valid_but_global_cycle():
    c = toy_checker()
    assert c.validate([[0, 2], [1], [3]]).is_valid
    assert c.validate([[3, 1], [0], [2]]).is_valid
    res = c.validate([[0, 2], [3, 1]], target_width=2)
    assert not res.is_valid and res.reason == 'dependency_cycle'


def test_inactive_singletons_do_not_change_active_width():
    g = nx.DiGraph()
    inputs = {i: ('in', i) for i in range(4)}
    outputs = {i: ('out', i) for i in range(4)}
    g.add_edges_from((inputs[i], outputs[i]) for i in range(4))
    c = ReuseValidityChecker.from_graphs(num_qubits=4, active_qubits={0, 1, 2},
                                        full_graph=g, full_inputs=inputs, full_outputs=outputs)
    res = c.validate([[0, 2], [1]], target_width=2)
    assert res.is_valid and res.active_width == 2 and res.full_width == 3
    assert res.details['inactive_qubits'] == (3,)


def test_qiskit_smoke_static_circuit():
    qiskit = pytest.importorskip('qiskit')
    qc = qiskit.QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(2)
    qc.measure([0, 1, 2], [0, 1, 2])
    checker = ReuseValidityChecker(qc)
    assert checker.validate([[0, 2], [1]], target_width=2).is_valid
    assert checker.validate([[0, 1], [2]], target_width=2).reason == 'illegal_reuse_boundary'
    assert checker.validate([[0, 2], [1]], target_width=3).reason == 'width_mismatch'


def test_qiskit_mid_circuit_measurement_rejected():
    qiskit = pytest.importorskip('qiskit')
    qc = qiskit.QuantumCircuit(2, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.x(0)
    qc.h(1)
    checker = ReuseValidityChecker(qc)
    res = checker.validate([[0], [1]], target_width=2)
    assert res.reason == 'mid_circuit_measurement'
