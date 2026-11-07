"""Adversarial / property-based STATIC reuse-validity audit over saved plans.

Example PowerShell (in project root, environment activated):
    python run_validity_adversarial.py --bench-dir "C:\\path\\to\\CaQR-main\\benchmarks"

NO GidNET generation, routing, or quantum simulation is performed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd
from qiskit import qasm2

from reuse_adversarial import (
    OPERATORS, canonical, malformed_variants, reference_static_feasible,
    unique_mutations,
)
from reuse_validity import ReuseValidityChecker


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench-dir', type=Path, required=True,
                        help='Directory containing original QASM files')
    parser.add_argument('--plan-csv', type=Path, default=Path('experiment2_results/experiment2_plan_features.csv'))
    parser.add_argument('--validation-csv', type=Path,
                        default=Path('validity_checker_results/validation_of_experiment2_saved_plans.csv'))
    parser.add_argument('--out-dir', type=Path, default=Path('validity_adversarial_results'))
    parser.add_argument('--per-operator', type=int, default=12,
                        help='Max unique candidates from EACH operator per plan (default: 12)')
    parser.add_argument('--seed', type=int, default=20261011)
    return parser.parse_args(argv)


def parse_chains(raw):
    parsed = ast.literal_eval(raw) if isinstance(raw, str) else raw
    return tuple(tuple(int(q) for q in chain) for chain in parsed)


def _seed(seed, *parts):
    h = hashlib.sha256('|'.join(map(str, (seed,) + parts)).encode('utf8')).digest()
    return int.from_bytes(h[:8], 'big')


def _safe_error(value):
    return str(value)[:400]


def run(args):
    if args.per_operator < 1:
        raise ValueError('--per-operator must be >= 1')
    if not args.bench_dir.is_dir():
        raise FileNotFoundError('QASM directory does not exist: ' + str(args.bench_dir))
    if not args.plan_csv.is_file():
        raise FileNotFoundError('Plan CSV not found: ' + str(args.plan_csv))
    df = pd.read_csv(args.plan_csv)
    expected_columns = {'circuit', 'plan_id', 'width', 'reuse_chains'}
    if missing := expected_columns - set(df.columns):
        raise ValueError('Missing plan CSV columns: ' + ', '.join(sorted(missing)))
    if df.duplicated(['circuit', 'plan_id']).any():
        raise ValueError('Duplicate (circuit, plan_id) rows in saved data.')

    outdir = args.out_dir
    outdir.mkdir(parents=True, exist_ok=True)
    audit, cases, malformed_cases = [], [], []
    warnings = []
    old_validation = None
    if args.validation_csv.is_file():
        old_validation = pd.read_csv(args.validation_csv)
        if old_validation.duplicated(['circuit', 'plan_id']).any():
            raise ValueError('Duplicate keys in prior validation CSV')
        old_validation = old_validation.set_index(['circuit', 'plan_id'])
    else:
        warnings.append('Previous validation CSV not found. Compare stored-vs-actual widths only via new checker.')

    for circuit_name, rows in df.groupby('circuit', sort=True):
        qcpath = args.bench_dir / circuit_name
        if not qcpath.is_file():
            raise FileNotFoundError('Original QASM not found: ' + str(qcpath))
        qc = qasm2.load(str(qcpath))
        checker = ReuseValidityChecker(qc)
        if checker.unsupported is not None:
            raise RuntimeError(f'Unsupported original circuit: {circuit_name}, {checker.unsupported.reason}')
        print(f'Checking {circuit_name}: {len(rows)} saved plans', flush=True)

        for _, row in rows.iterrows():
            plan_id = str(row.plan_id)
            chains = parse_chains(row.reuse_chains)
            saved_width = int(row.width)
            result = checker.validate(chains, target_width=saved_width)
            oracle = reference_static_feasible(qc, chains, saved_width)
            old_width = None
            old_valid = None
            key = (circuit_name, row.plan_id)
            if old_validation is not None and key in old_validation.index:
                previous = old_validation.loc[key]
                old_width = int(previous['actual_active_width'])
                # Convert saved string values safely, without bool('False') pitfall.
                old_valid = str(previous['is_valid_static']).strip().lower() == 'true'

            # Metamorphic property: the order of whole reuse chains is irrelevant.
            reordered = tuple(reversed(chains))
            reordered_checker = checker.validate(reordered, target_width=saved_width)
            reordered_oracle = reference_static_feasible(qc, reordered, saved_width)
            order_invariant = (result.is_valid == reordered_checker.is_valid
                               and oracle == reordered_oracle)

            audit.append({
                'chain_order_invariance': order_invariant,
                'circuit': circuit_name, 'plan_id': plan_id, 'saved_width': saved_width,
                'chains_width': len(chains), 'checker_width': result.active_width,
                'previous_actual_width': old_width,
                'saved_equals_chains_width': saved_width == len(chains),
                'saved_equals_checker_width': saved_width == result.active_width,
                'saved_equals_previous_width': None if old_width is None else saved_width == old_width,
                'previous_is_valid_static': old_valid,
                'checker_is_valid': result.is_valid, 'reference_is_valid': oracle,
                'oracle_agrees': result.is_valid == oracle,
                'checker_reason': result.reason,
            })

            for operator in OPERATORS:
                pool = unique_mutations(chains, operator)
                size = len(pool)
                rng = random.Random(_seed(args.seed, circuit_name, plan_id, operator))
                selected = rng.sample(pool, min(size, args.per_operator))
                for idx, candidate in enumerate(selected):
                    answer = checker.validate(candidate, target_width=saved_width)
                    oracle = reference_static_feasible(qc, candidate, saved_width)
                    cases.append({
                        'circuit': circuit_name, 'plan_id': plan_id,
                        'operator': operator, 'sample_index': idx,
                        'available_unique_neighborhood': size,
                        'candidate_chains': repr(candidate),
                        'candidate_signature': repr(canonical(candidate)),
                        'checker_is_valid': answer.is_valid,
                        'checker_reason': answer.reason,
                        'reference_is_valid': oracle,
                        'oracle_agrees': answer.is_valid == oracle,
                        'preserved_width': answer.active_width == saved_width,
                        'preserved_partition': sorted(q for c in candidate for q in c)
                                                == sorted(q for c in chains for q in c),
                        'num_boundaries': answer.num_reuse_boundaries,
                    })
            for name, invalid, known_reason, width in malformed_variants(chains, qc.num_qubits):
                try:
                    actual = checker.validate(invalid, target_width=width)
                    ref = reference_static_feasible(qc, invalid, width)
                    passed = (not actual.is_valid and not ref and actual.reason == known_reason)
                    reason = actual.reason
                except Exception as exc:
                    passed = False
                    reason = f'EXCEPTION: {_safe_error(exc)}'
                malformed_cases.append({
                    'circuit': circuit_name, 'plan_id': plan_id,
                    'mutation': name, 'expected_reason': known_reason,
                    'actual_reason': reason, 'passed': passed,
                })

    width_df = pd.DataFrame(audit)
    case_df = pd.DataFrame(cases)
    bad_df = pd.DataFrame(malformed_cases)
    width_df.to_csv(outdir / 'width_and_original_plan_audit.csv', index=False)
    case_df.to_csv(outdir / 'adversarial_mutation_cases.csv', index=False)
    bad_df.to_csv(outdir / 'malformed_input_tests.csv', index=False)

    if len(case_df):
        summary = case_df.groupby(['circuit', 'operator'], as_index=False).agg(
            evaluated=('sample_index', 'size'),
            statically_valid=('checker_is_valid', 'sum'),
            oracle_matches=('oracle_agrees', 'sum'),
            all_mutations_preserve_width=('preserved_width', 'all'),
            all_mutations_preserve_partition=('preserved_partition', 'all'),
            mean_available_unique_neighborhood=('available_unique_neighborhood', 'mean'),
        )
        summary['valid_rate'] = summary['statically_valid'] / summary['evaluated']
        reasons = (case_df.groupby(['circuit', 'operator', 'checker_reason'])
                   .size().reset_index(name='count'))
    else:
        summary = pd.DataFrame()
        reasons = pd.DataFrame()
    summary.to_csv(outdir / 'neighborhood_diagnostic_summary.csv', index=False)
    reasons.to_csv(outdir / 'rejection_reasons.csv', index=False)

    mismatches = case_df.loc[~case_df['oracle_agrees']] if len(case_df) else pd.DataFrame()
    mismatches.to_csv(outdir / 'oracle_disagreements.csv', index=False)
    structural_failures = bad_df.loc[~bad_df['passed']]
    width_ok = bool(width_df['saved_equals_chains_width'].all() and
                    width_df['saved_equals_checker_width'].all())
    if old_validation is not None:
        width_ok = width_ok and bool(width_df['saved_equals_previous_width'].fillna(False).all())
    all_original_valid = bool(width_df['checker_is_valid'].all() and width_df['reference_is_valid'].all()
                              and width_df['oracle_agrees'].all()
                              and width_df['chain_order_invariance'].all())
    if old_validation is not None:
        all_original_valid = all_original_valid and bool(width_df['previous_is_valid_static'].fillna(False).all())
    mutation_ok = bool(len(case_df) == 0 or
                       (case_df['oracle_agrees'].all() and
                        case_df['preserved_width'].all() and
                        case_df['preserved_partition'].all()))
    malformed_ok = structural_failures.empty
    success = width_ok and all_original_valid and mutation_ok and malformed_ok
    report = {
        'success': success,
        'total_original_plans': len(width_df),
        'circuits': int(df.circuit.nunique()),
        'mutations_tested': len(case_df),
        'malformed_inputs_tested': len(bad_df),
        'saved_width_agreement': width_ok,
        'all_original_plans_valid_both_methods': all_original_valid,
        'oracle_disagreements': len(mismatches),
        'malformed_input_failures': len(structural_failures),
        'per_operator_cap': args.per_operator,
        'rng_seed': args.seed,
        'scope': 'STATIC dependency feasibility; no dynamic semantics or routing',
        'warnings': warnings,
    }
    (outdir / 'audit_report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print('\n' + json.dumps(report, indent=2))
    print('\nPer-circuit/operator results:\n', summary.to_string(index=False) if len(summary) else '(empty)')
    print(f'\nSaved all CSVs and report to: {outdir.resolve()}')
    if not success:
        print('\nFAIL: inspect oracle_disagreements.csv, width audit, and malformed_input_tests.csv.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(run(parse_args()))
