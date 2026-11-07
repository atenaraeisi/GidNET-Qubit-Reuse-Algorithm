"""Regression checks for the real-plan terminal-readout probing helper.

Run: python -m pytest -q test_real_plan_semantic_probe.py
"""
from qiskit import QuantumCircuit

from run_real_plan_semantic_audit import prepare_readout_probe
from semantic_correctness import (
    compile_from_reuse_plan, audit_compilation_structure,
    compare_exact_semantics,
)


def test_final_readout_probe_preserves_original_and_passes_semantics():
    original = QuantumCircuit(3)
    original.h(0)
    original.cx(0, 1)  # outgoing q0's last gate has two operands
    original.x(2)

    probed, added, already_measured = prepare_readout_probe(original, probe_unmeasured=True)
    assert added and not already_measured
    assert original.num_clbits == 0  # did not mutate caller's source
    assert probed.num_clbits == 3
    assert all(inst.operation.name != "barrier" for inst in probed.data)
    
    compiled = compile_from_reuse_plan(probed, [[0, 2], [1]], target_width=2)
    audit_compilation_structure(compiled)
    outcome = compare_exact_semantics(probed, compiled.circuit)
    assert outcome['equal'], outcome


def test_existing_measurements_not_modified():
    original = QuantumCircuit(2, 2)
    original.h(0)
    original.measure([0, 1], [0, 1])
    probed, added, already_measured = prepare_readout_probe(original, probe_unmeasured=True)
    assert probed is original
    assert not added and already_measured


def test_probe_can_be_disabled():
    original = QuantumCircuit(2)
    output, added, already_measured = prepare_readout_probe(original, probe_unmeasured=False)
    assert output is original
    assert not added and not already_measured
