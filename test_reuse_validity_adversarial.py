"""Run: python -m pytest -q test_reuse_validity.py test_reuse_validity_adversarial.py"""
import itertools
import random

import networkx as nx
import pytest

from reuse_adversarial import (
    OPERATORS, canonical, malformed_variants, reference_static_feasible,
    unique_mutations,
)
from reuse_validity import ReuseValidityChecker


def example_qiskit_circuit():
    qiskit = pytest.importorskip('qiskit')
    qc = qiskit.QuantumCircuit(5, 5)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.h(4)
    qc.measure(range(5), range(5))
    return qc


def test_mutation_generators_preserve_partition_and_width():
    initial = ((0, 3), (1,), (2, 4))
    results = {}
    for kind in OPERATORS:
        neighbors = unique_mutations(initial, kind)
        results[kind] = len(neighbors)
        assert len({canonical(plan) for plan in neighbors}) == len(neighbors)
        for plan in neighbors:
            assert len(plan) == len(initial)
            assert all(plan)
            assert sorted(q for chain in plan for q in chain) == [0, 1, 2, 3, 4]
            assert canonical(plan) != canonical(initial)
    assert results['chain_transfer'] > 0
    assert results['suffix_swap'] > 0
    assert results['node_swap_reconnect'] > 0
    assert results['segment_reverse'] > 0


def test_known_bad_inputs_are_rejected():
    qc = example_qiskit_circuit()
    checker = ReuseValidityChecker(qc)
    plan = ((0, 2), (1,), (3,), (4,))
    for name, mutated, expected, width in malformed_variants(plan, qc.num_qubits):
        actual = checker.validate(mutated, target_width=width)
        assert not actual.is_valid, name
        assert actual.reason == expected, (name, actual.reason, expected)
        assert not reference_static_feasible(qc, mutated, width), name


def test_small_exhaustive_partitions_agree_with_reference():
    """Enumerate EVERY width-2 ordered-chain assignment of four active qubits."""
    qiskit = pytest.importorskip('qiskit')
    qc = qiskit.QuantumCircuit(4, 4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.measure(range(4), range(4))
    checker = ReuseValidityChecker(qc)
    seen = set()
    accepted = rejected = 0
    for perm in itertools.permutations(range(4)):
        for cut in (1, 2, 3):
            plan = (perm[:cut], perm[cut:])
            key = canonical(plan)
            if key in seen:
                continue
            seen.add(key)
            actual = checker.validate(plan, target_width=2).is_valid
            expected = reference_static_feasible(qc, plan, target_width=2)
            assert actual == expected, (plan, actual, expected)
            accepted += actual
            rejected += not actual
    assert accepted > 0 and rejected > 0
    assert len(seen) == 36


def test_metamorphic_chain_order_invariance():
    qc = example_qiskit_circuit()
    checker = ReuseValidityChecker(qc)
    plan = ((0, 2), (1,), (3,), (4,))
    base = checker.validate(plan, target_width=4)
    assert base.is_valid
    for reordered in itertools.permutations(plan):
        v = checker.validate(reordered, target_width=4)
        assert v.is_valid == base.is_valid
        assert v.reason == base.reason
        assert reference_static_feasible(qc, reordered, 4) == base.is_valid


def test_randomized_mutants_agree_with_reference():
    qc = example_qiskit_circuit()
    checker = ReuseValidityChecker(qc)
    rng = random.Random(20261011)
    base = ((0, 2), (1,), (3,), (4,))
    validated = 0
    for kind in OPERATORS:
        pool = unique_mutations(base, kind)
        rng.shuffle(pool)
        for candidate in pool[:70]:
            result = checker.validate(candidate, target_width=4)
            ref = reference_static_feasible(qc, candidate, 4)
            assert result.is_valid == ref, (kind, candidate, result.reason)
            validated += 1
    assert validated >= 15


def test_cycle_rejection_in_reference():
    qc = example_qiskit_circuit()
    checker = ReuseValidityChecker(qc)
    # q0 and q1 interact: forcing q0->q1 is forbidden.
    bad = ((0, 1), (2,), (3,), (4,))
    assert not checker.validate(bad, target_width=4).is_valid
    assert not reference_static_feasible(qc, bad, 4)
