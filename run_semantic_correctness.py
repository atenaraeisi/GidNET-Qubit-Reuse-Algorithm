"""Audit selected saved originals and VALID changed plans from real benchmarks.

No GidNET search, transpile, routing, hardware backend, or noisy simulation.
Run from repository root. See SEMANTIC_CORRECTNESS_GUIDE.md.
"""
from __future__ import annotations
import argparse
import ast
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from semantic_correctness import (
    SemanticAuditError, SimulationLimit, audit_compilation_structure,
    compare_exact_semantics, compile_from_reuse_plan,
)


def parse_chains(text):
    obj = ast.literal_eval(text)
    if not isinstance(obj, (tuple, list)) or any(not isinstance(c, (tuple, list)) for c in obj):
        raise ValueError('Expected a tuple/list of reuse chains')
    return tuple(tuple(int(v) for v in c) for c in obj)


def load_cases(original_csv, mutation_csv, *, original_per_circuit, mutated_per_circuit):
    cases, counts = [], Counter()
    with open(original_csv, encoding='utf-8-sig', newline='') as f:
        for r in sorted(csv.DictReader(f), key=lambda x: (x['circuit'], x['plan_id'])):
            key = r['circuit']
            if counts[key] >= original_per_circuit:
                continue
            counts[key] += 1
            cases.append({'kind': 'original', 'circuit': key, 'plan_id': r['plan_id'],
                          'operator': 'none', 'chains': parse_chains(r['reuse_chains']),
                          'saved_width': int(r['width'])})

    # Only candidate plans already confirmed static-valid by both checkers.
    # Deterministic diverse selection: chain_transfer, node_swap, suffix_swap.
    candidates = defaultdict(list)
    with open(mutation_csv, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if (r.get('checker_is_valid', '').lower() != 'true'
                    or r.get('reference_is_valid', '').lower() != 'true'
                    or r.get('oracle_agrees', '').lower() != 'true'
                    or r.get('preserved_width', '').lower() != 'true'
                    or r.get('preserved_partition', '').lower() != 'true'):
                continue
            candidates[r['circuit']].append(r)

    priority = {'chain_transfer': 0, 'node_swap_reconnect': 1, 'suffix_swap': 2}
    for circuit, rows in sorted(candidates.items()):
        # One distinct candidate from each operator if possible. This avoids
        # accidentally testing only chain_transfer in a mixed-operator suite.
        groups = defaultdict(list)
        for r in sorted(rows, key=lambda x: (x['operator'], x['plan_id'], x['candidate_signature'])):
            groups[r['operator']].append(r)
        order = sorted(groups, key=lambda op: (priority.get(op, 9), op))
        seen, chosen = set(), []
        while len(chosen) < mutated_per_circuit and any(groups[op] for op in order):
            progress = False
            for op in order:
                while groups[op] and groups[op][0]['candidate_signature'] in seen:
                    groups[op].pop(0)
                if not groups[op] or len(chosen) >= mutated_per_circuit:
                    continue
                r = groups[op].pop(0)
                seen.add(r['candidate_signature'])
                chosen.append(r)
                progress = True
            if not progress:
                break
        for r in chosen:
            chains = parse_chains(r['candidate_chains'])
            cases.append({'kind': 'mutation', 'circuit': circuit,
                          'plan_id': r['plan_id'], 'operator': r['operator'],
                          'chains': chains, 'saved_width': len(chains)})
    return cases


def process_case(row, qc, *, max_sim_qubits, max_branches, max_operations):
    t0 = time.perf_counter()
    result = {'kind': row['kind'], 'circuit': row['circuit'],
              'plan_id': row['plan_id'], 'operator': row['operator'],
              'saved_width': row['saved_width'], 'n_original_qubits': qc.num_qubits,
              'static_valid': False, 'structurally_verified': False,
              'exact_semantics': 'NOT_RUN', 'tvd': '', 'status': '', 'detail': ''}
    try:
        compiled = compile_from_reuse_plan(qc, row['chains'], target_width=row['saved_width'])
        result['static_valid'] = True
        result['compiled_width'] = compiled.expected_width
        structure = audit_compilation_structure(compiled)
        result['structurally_verified'] = True
        result['reset_count'] = structure['actual_resets']
        if qc.num_clbits == 0 or not any(x.operation.name == 'measure' for x in qc.data):
            result['exact_semantics'] = 'SKIPPED_NO_CLASSICAL_OUTPUT'
            result['status'] = 'STRUCTURE_ONLY'
        else:
            try:
                comp = compare_exact_semantics(qc, compiled.circuit,
                                               max_qubits=max_sim_qubits,
                                               max_branches=max_branches,
                                               max_operations=max_operations)
                result['tvd'] = comp['total_variation_distance']
                result['exact_semantics'] = 'PASS' if comp['equal'] else 'FAIL'
                result['status'] = 'SEMANTIC_PASS' if comp['equal'] else 'SEMANTIC_MISMATCH'
                if not comp['equal']:
                    result['detail'] = repr(comp)[:600]
            except SimulationLimit as exc:
                result['exact_semantics'] = 'SKIPPED_LIMIT'
                result['status'] = 'STRUCTURE_ONLY'
                result['detail'] = str(exc)
    except Exception as exc:
        result['status'] = 'COMPILE_OR_STRUCTURE_ERROR'
        result['detail'] = f'{type(exc).__name__}: {exc}'[:650]
    result['elapsed_sec'] = round(time.perf_counter() - t0, 4)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bench-dir', required=True, type=Path)
    p.add_argument('--original-csv', type=Path, default=Path('experiment2_results/experiment2_plan_features.csv'))
    p.add_argument('--mutation-csv', type=Path, default=Path('validity_exhaustive_results/adversarial_mutation_cases.csv'))
    p.add_argument('--out-dir', type=Path, default=Path('semantic_correctness_results'))
    p.add_argument('--original-per-circuit', type=int, default=2)
    p.add_argument('--mutated-per-circuit', type=int, default=3)
    p.add_argument('--max-sim-qubits', type=int, default=7)
    p.add_argument('--max-branches', type=int, default=4096)
    p.add_argument('--max-operations', type=int, default=250)
    args = p.parse_args()
    from qiskit import qasm2

    for path in (args.bench_dir, args.original_csv, args.mutation_csv):
        if not path.exists():
            p.error(f'File/directory does not exist: {path}')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.original_csv, args.mutation_csv,
                       original_per_circuit=args.original_per_circuit,
                       mutated_per_circuit=args.mutated_per_circuit)
    qc_cache, results = {}, []
    print(f'Selected {len(cases)} test cases across {len(set(x["circuit"] for x in cases))} circuits')
    for i, row in enumerate(cases, 1):
        name = row['circuit']
        if name not in qc_cache:
            qc_cache[name] = qasm2.load(str(args.bench_dir / name))
        res = process_case(row, qc_cache[name], max_sim_qubits=args.max_sim_qubits,
                           max_branches=args.max_branches,
                           max_operations=args.max_operations)
        results.append(res)
        print(f'[{i:02d}/{len(cases):02d}] {name} {row["kind"]} {row["plan_id"]} '
              f'{row["operator"]}: {res["status"]} {res["detail"][:100]}')

    output = args.out_dir / 'semantic_audit_cases.csv'
    columns = ['kind', 'circuit', 'plan_id', 'operator', 'saved_width',
               'n_original_qubits', 'static_valid', 'structurally_verified',
               'compiled_width', 'reset_count', 'exact_semantics', 'tvd',
               'status', 'detail', 'elapsed_sec']
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    status_counts = dict(Counter(x['status'] for x in results))
    summary = {
        'tested_cases': len(results), 'circuits': len(qc_cache),
        'statuses': status_counts,
        'all_structures_pass': all(x['structurally_verified'] for x in results),
        'semantic_cases_run': sum(x['exact_semantics'] in ('PASS', 'FAIL') for x in results),
        'semantic_failures': sum(x['exact_semantics'] == 'FAIL' for x in results),
        'overall_no_detected_failures': all(x['status'] in ('SEMANTIC_PASS', 'STRUCTURE_ONLY') for x in results),
        'interpretation': 'A STRUCTURE_ONLY result does not establish semantic equivalence.',
        'scope': 'fixed-plan GidNET compiler; static gates + terminal measurements; |0> basis initial states',
    }
    (args.out_dir / 'semantic_audit_report.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\nSaved:', output)
    print('Report:', args.out_dir / 'semantic_audit_report.json')
    print('Summary:', summary['statuses'])
    if not summary['overall_no_detected_failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
