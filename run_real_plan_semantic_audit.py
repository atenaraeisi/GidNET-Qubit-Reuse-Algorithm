"""Audit saved original and mutated fixed-width reuse plans on real QASM circuits.

Run from the GidNET repository root, with these files also at its root:
  semantic_correctness.py
  run_semantic_correctness.py
  reuse_validity.py

This runner does NOT regenerate GidNET plans, route circuits, or edit QASM files.

If a source QASM has no measurements, `--probe-unmeasured` (default) adds final
readout measurements to every original logical qubit, in memory only. It then
checks the complete *classical output distribution* of this instrumented
circuit against the compiled fixed-plan circuit when exact simulation is within
limits. This is stronger than a structure-only audit, but is NOT a proof of
full quantum-channel equivalence for arbitrary inputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from run_semantic_correctness import load_cases, process_case



def prepare_readout_probe(circuit, *, probe_unmeasured: bool):
    """Add individual terminal measurements without a global barrier."""
    from qiskit.circuit import ClassicalRegister

    has_measurement = any(
        inst.operation.name == "measure"
        for inst in circuit.data
    )

    if has_measurement or not probe_unmeasured:
        return circuit, False, has_measurement

    measured = circuit.copy()

    probe = ClassicalRegister(
        measured.num_qubits,
        "semantic_probe"
    )
    measured.add_register(probe)

    for qubit, clbit in zip(measured.qubits, probe):
        measured.measure(qubit, clbit)

    return measured, True, False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench-dir', type=Path, required=True,
                        help='Directory containing the original QASM benchmark files')
    parser.add_argument('--original-csv', type=Path,
                        default=Path('experiment2_results/experiment2_plan_features.csv'))
    parser.add_argument('--mutation-csv', type=Path,
                        default=Path('validity_exhaustive_results/adversarial_mutation_cases.csv'))
    parser.add_argument('--out-dir', type=Path,
                        default=Path('semantic_realplan_results'))
    parser.add_argument('--original-per-circuit', type=int, default=2)
    parser.add_argument('--mutated-per-circuit', type=int, default=3)
    parser.add_argument('--max-sim-qubits', type=int, default=8,
                        help='Exact simulation limit (applies to original AND compiled width)')
    parser.add_argument('--max-branches', type=int, default=4096)
    parser.add_argument('--max-operations', type=int, default=300)
    parser.add_argument('--probe-unmeasured', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Append terminal readout to otherwise unmeasured QASMs (in memory only)')
    args = parser.parse_args()
    if args.original_per_circuit < 0 or args.mutated_per_circuit < 0:
        parser.error('Per-circuit case counts must be nonnegative')
    if args.max_sim_qubits < 1 or args.max_branches < 1 or args.max_operations < 1:
        parser.error('Simulation limits must be positive')
    for path in (args.bench_dir, args.original_csv, args.mutation_csv):
        if not path.exists():
            parser.error(f'Missing input path: {path}')
    return args


def main():
    args = parse_args()
    try:
        from qiskit import qasm2
    except ImportError as exc:
        raise SystemExit('Qiskit is required; activate .venv_gidnet first') from exc

    cases = load_cases(args.original_csv, args.mutation_csv,
                       original_per_circuit=args.original_per_circuit,
                       mutated_per_circuit=args.mutated_per_circuit)
    if not cases:
        raise SystemExit('No cases selected; check input CSVs and case limits')

    cache = {}
    results = []
    print(f'Selected {len(cases)} cases from {len(set(x["circuit"] for x in cases))} circuits')
    print(f'Probe unmeasured QASM: {args.probe_unmeasured}. Simulation max qubits: {args.max_sim_qubits}')
    for idx, row in enumerate(cases, 1):
        name = row['circuit']
        if name not in cache:
            original = qasm2.load(str(args.bench_dir / name))
            cache[name] = prepare_readout_probe(original,
                                                probe_unmeasured=args.probe_unmeasured)
        circuit, probed, source_has_measurements = cache[name]
        result = process_case(row, circuit,
                              max_sim_qubits=args.max_sim_qubits,
                              max_branches=args.max_branches,
                              max_operations=args.max_operations)
        result.update({
            'readout_probe_added': probed,
            'source_had_measurements': source_has_measurements,
            'test_num_clbits': circuit.num_clbits,
            'verification_target': ('instrumented_final_readout' if probed
                                    else 'original_classical_output' if source_has_measurements
                                    else 'structure_only_unmeasured_source'),
        })
        results.append(result)
        msg = f'[{idx}/{len(cases)}] {name} {row["kind"]} {row["plan_id"]} / {row["operator"]}: {result["status"]}'
        if result.get('detail'):
            msg += ' (' + str(result['detail'])[:120] + ')'
        print(msg, flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ['kind', 'circuit', 'plan_id', 'operator', 'saved_width',
                  'n_original_qubits', 'test_num_clbits', 'readout_probe_added',
                  'source_had_measurements', 'verification_target', 'static_valid',
                  'structurally_verified', 'compiled_width', 'reset_count',
                  'exact_semantics', 'tvd', 'status', 'detail', 'elapsed_sec']
    cases_path = args.out_dir / 'semantic_audit_cases.csv'
    with cases_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    statuses = Counter(r['status'] for r in results)
    semantic_checked = [r for r in results if r['exact_semantics'] in {'PASS', 'FAIL'}]
    skipped = [r for r in results if r['status'] == 'STRUCTURE_ONLY']
    failures = [r for r in results if r['status'] not in {'SEMANTIC_PASS', 'STRUCTURE_ONLY'}]
    by_operator = defaultdict(Counter)
    for r in results:
        by_operator[r['operator']][r['status']] += 1

    report = {
        'experiment': 'Real-plan end-to-end compilation and classical readout audit',
        'source_circuits': len(cache),
        'selected_cases': len(results),
        'original_cases': sum(r['kind'] == 'original' for r in results),
        'mutated_cases': sum(r['kind'] == 'mutation' for r in results),
        'status_counts': dict(statuses),
        'by_operator': {k: dict(v) for k, v in sorted(by_operator.items())},
        'probe_unmeasured': bool(args.probe_unmeasured),
        'probed_cases': sum(bool(r['readout_probe_added']) for r in results),
        'semantics_compared': len(semantic_checked),
        'semantic_passes': sum(r['exact_semantics'] == 'PASS' for r in semantic_checked),
        'semantic_mismatches': sum(r['exact_semantics'] == 'FAIL' for r in semantic_checked),
        'structure_only': len(skipped),
        'structural_failures': sum(not r['structurally_verified'] for r in results),
        'all_compilations_and_structures_pass': all(r['structurally_verified'] for r in results),
        'no_detected_failures': not failures,
        'simulation_limits': {
            'max_qubits': args.max_sim_qubits,
            'max_branches': args.max_branches,
            'max_operations': args.max_operations,
        },
        'caveats': [
            'STRUCTURE_ONLY cases have NOT been semantically verified.',
            'A terminal-readout probe tests the joint classical-output distribution '
            'of an instrumented source circuit, not its full quantum channel.',
            'Tests assume a |0...0> input and ideal operations, not arbitrary states, '
            'hardware noise, or routing.',
            'Mutations were selected from pre-validated static-valid cases; '
            'this audit independently tests their actual compilation.',
        ],
    }
    report_path = args.out_dir / 'semantic_audit_report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\nSaved:', cases_path)
    print('Saved:', report_path)
    print('Summary:', dict(statuses))
    if failures:
        print('Failures detected. Do not proceed to physical experiments until investigated.')
        raise SystemExit(1)
    if len(semantic_checked) == 0:
        print('WARNING: No exact semantic comparisons ran; all cases were structure-only.')


if __name__ == '__main__':
    main()
