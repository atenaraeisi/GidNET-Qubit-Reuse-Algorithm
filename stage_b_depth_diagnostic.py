#!/usr/bin/env python3
"""Non-destructive diagnosis for Stage B frozen-plan logical-depth mismatches.

Run from the GidNET repository root, with Qiskit installed in .venv_gidnet.
This script does NOT route any circuits, change GidNET files, or change CSVs.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import inspect
import json
import sys
from collections import Counter
from pathlib import Path


def sha256(path):
    p = Path(path)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_table(root, filename):
    paths = list(root.rglob(filename))
    if len(paths) != 1:
        raise FileNotFoundError(f'Expected exactly one {filename} under {root}; found {len(paths)}: {paths}')
    with paths[0].open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f)), paths[0]


def op_items(qc):
    for item in qc.data:
        if hasattr(item, 'operation'):
            yield item.operation, list(item.qubits), list(item.clbits)
        else:
            yield item[0], list(item[1]), list(item[2])


def legacy_reset_normalization(qc):
    """Replicate Experiment's older reset->measure/conditional-X normalization convention.

    This is a DIFFERENTIAL DIAGNOSTIC ONLY, not the production compiler.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ClassicalRegister
    new_qc = QuantumCircuit(*qc.qregs, *qc.cregs)
    last_measure = {}
    clbit_owner = {}
    total_resets = sum(op.name == 'reset' for op, _, _ in op_items(qc))
    aux_reg = None
    aux_next = 0

    for op, qargs, cargs in op_items(qc):
        qindices = [qc.find_bit(q).index for q in qargs]
        if op.name == 'measure':
            if len(qargs) != 1 or len(cargs) != 1:
                raise ValueError('Unexpected multi-bit measurement in legacy normalization')
            qidx = qc.find_bit(qargs[0]).index
            cidx = qc.find_bit(cargs[0]).index
            oldowner = clbit_owner.get(cidx)
            if oldowner is not None:
                last_measure.pop(oldowner, None)
            new_qc.append(op, qargs, cargs)
            last_measure[qidx] = new_qc.clbits[cidx]
            clbit_owner[cidx] = qidx
        elif op.name == 'reset':
            if len(qargs) != 1:
                raise ValueError('Invalid reset instruction')
            qidx = qc.find_bit(qargs[0]).index
            cbit = last_measure.get(qidx)
            if cbit is None:
                if aux_reg is None:
                    aux_reg = ClassicalRegister(max(total_resets, 1), 'reset_aux')
                    new_qc.add_register(aux_reg)
                cbit = aux_reg[aux_next]
                aux_next += 1
                new_qc.measure(qargs[0], cbit)
            new_qc.x(qargs[0]).c_if(cbit, 1)
            last_measure.pop(qidx, None)
        else:
            new_qc.append(op, qargs, cargs)
            for qidx in qindices:
                last_measure.pop(qidx, None)
    return new_qc


def op_count(qc):
    return {str(k): int(v) for k,v in dict(qc.count_ops()).items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--qasm-dir',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,default=Path('stage_b_depth_diagnosis'))
    args=ap.parse_args()
    project_root=Path.cwd().resolve()
    pkg=project_root/'targeted_routing_diagnostics_package'
    if not (pkg/'targeted_routing_diagnostics.py').exists():
        ap.error('Run from repository root containing targeted_routing_diagnostics_package')
    sys.path.insert(0,str(pkg))
    sys.path.insert(0,str(project_root))

    import qiskit
    from qiskit import QuantumCircuit
    from gidnet.qubitreuse import GidNET
    from reuse_validity import ReuseValidityChecker
    from targeted_routing_diagnostics import (PAIR_SPECS, parse_chains, find_qasm,
                                               lower_active_resets, conditional_x,
                                               check_quotient_features)

    rows, summary_path=load_table(args.data_dir, 'experiment1_fresh30_plan_summary.csv')
    feat_rows, feat_path=load_table(args.data_dir, 'experiment2_plan_features.csv')
    lookup={(r['circuit'],r['plan_id']):r for r in rows}
    feats={(r['circuit'],r['plan_id']):r for r in feat_rows}
    source=Path(inspect.getfile(GidNET))
    results=[]
    for circuit,better,worse in PAIR_SPECS:
        path=find_qasm(args.qasm_dir,circuit)
        orig=QuantumCircuit.from_qasm_file(str(path))
        checker=ReuseValidityChecker(orig)
        print(f'\n{circuit} | QASM sha256={sha256(path)[:16]} | {orig.num_qubits} declared qubits')
        for pid in (better,worse):
            row=lookup[(circuit,pid)]
            chains=parse_chains(row['plan_signature'])
            validity=checker.validate(chains,target_width=int(row['width']))
            if not validity.is_valid:
                raise RuntimeError(f'Static validation failed for {circuit}:{pid}: {validity.reason}')
            check_quotient_features(orig,chains,feats[(circuit,pid)],circuit,pid)
            inst=GidNET(orig)
            inst.compute_optimized_reuse_sequences=lambda n, frozen=chains: [list(c) for c in frozen]
            native=inst.compile_to_dynamic_circuit(iterations=1,draw=False)
            if native is None or native.num_qubits!=len(chains):
                raise RuntimeError(f'Invalid compiled width {circuit}:{pid}')
            norm=lower_active_resets(native)
            legacy=legacy_reset_normalization(native)
            need_reset=sum(len(c)-1 for c in chains)
            src_counts=Counter({k:v for k,v in op_count(orig).items() if k not in {'measure','reset','barrier'}})
            native_counts=Counter({k:v for k,v in op_count(native).items() if k not in {'measure','reset','barrier'}})
            gate_counts_agree=src_counts==native_counts
            result={
                'circuit':circuit,'plan':pid,'original_qasm_sha256':sha256(path),
                'archived_raw_depth':int(row['raw_logical_depth']),
                'current_raw_depth':int(native.depth()),
                'raw_depth_drift':int(native.depth())-int(row['raw_logical_depth']),
                'archived_normalized_depth':int(row['normalized_logical_depth']),
                'current_normalized_depth':int(norm.depth()),
                'current_legacy_style_normalized_depth':int(legacy.depth()),
                'normalized_depth_drift':int(norm.depth())-int(row['normalized_logical_depth']),
                'normalizer_depth_difference':int(norm.depth())-int(legacy.depth()),
                'reuse_boundary_count':need_reset,
                'native_reset_count':op_count(native).get('reset',0),
                'normalized_measurement_count':op_count(norm).get('measure',0),
                'normalized_conditional_x_count':int(conditional_x(norm)),
                'legacy_measurement_count':op_count(legacy).get('measure',0),
                'legacy_conditional_x_count':int(conditional_x(legacy)),
                'gate_multiset_preserved':gate_counts_agree,
                'source_gate_counts':dict(src_counts),
                'compiled_gate_counts':dict(native_counts),
                'compiled_op_counts':op_count(native),
                'new_normalized_op_counts':op_count(norm),
                'legacy_normalized_op_counts':op_count(legacy),
                'width':int(norm.num_qubits),
                'status':'COUNTS_OK' if (gate_counts_agree and need_reset==op_count(native).get('reset',0)
                           and op_count(norm).get('measure',0)==need_reset
                           and conditional_x(norm)==need_reset) else 'STRUCTURAL_WARNING'
            }
            results.append(result)
            print(f"  {pid}: archived raw/norm={result['archived_raw_depth']}/{result['archived_normalized_depth']} | "
                  f"current raw/norm={result['current_raw_depth']}/{result['current_normalized_depth']} | "
                  f"old-style-normalization={result['current_legacy_style_normalized_depth']} | "
                  f"resets={result['native_reset_count']}/{need_reset} | {result['status']}")
    args.out_dir.mkdir(parents=True,exist_ok=True)
    report={'qiskit_version':qiskit.__version__,
            'gidnet_source':str(source),'gidnet_sha256':sha256(source),
            'archive_summary_sha256':sha256(summary_path),
            'experiment2_features_sha256':sha256(feat_path),
            'purpose':'Non-destructive depth drift diagnosis; no routing, no source modifications',
            'cases':results}
    target=args.out_dir/'depth_drift_report.json'
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'\nFinished: {target}')
    print('Do NOT bypass archived depth check until the source of the drift is understood.')

if __name__=='__main__':
    main()
