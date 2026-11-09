#!/usr/bin/env python3
"""Compare legacy vs validity-aware neighbor generation on the SAME source QASMs.

* No routing, no training, no changes to existing outputs.
* Compare every unique VALID neighbor for each original/operator (not only samples).
* Enforce exact ordered-list parity, under equal RAW mutation limits.
* Recheck all fast survivors using ReuseValidityChecker and a bounded number
  with the separate operation-level reference scheduler.
* Measure engine setup and mutation scans separately; run full/partial caps.

Works from pilot_code with Qiskit 0.45.3. Writes diagnostics in NEW out-dir.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import traceback
from pathlib import Path

from build_multistep_pilot_validity_aware import (CIRCUITS, read_sources,
    compact_active, make_checker, canonical)
from reuse_adversarial import OPERATORS, reference_static_feasible
from validity_aware_neighbors import SUPPORTED_OPERATORS, ValidityAwareEngine


def ms(): return time.perf_counter()*1000


def legacy_scan(plan, ck, op, seen, cap):
    valid=[]
    attempts=0
    for i, raw in enumerate(OPERATORS[op](plan)):
        if cap and i>=cap:
            break
        attempts+=1
        if not raw or not all(raw):
            continue
        cand=canonical(raw)
        if cand==plan or cand in seen:
            continue
        seen.add(cand)
        if ck.validate(cand, target_width=len(plan)).is_valid:
            valid.append(cand)
    return valid, attempts


def write_csv(path, rows, columns):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit-dirs', nargs='+',type=Path, default=[Path('fixed_full'),Path('wave2')])
    ap.add_argument('--out-dir',type=Path, default=Path('validity_parity_out'))
    ap.add_argument('--circuits',default='',help='Source::filename keys, comma-separated; default: the six pilot circuits')
    ap.add_argument('--max-originals',type=int,default=2)
    ap.add_argument('--max-attempts-per-op',type=int,default=180,
        help='Raw attempted mutations per op, 0 means exhaustive (may be expensive)')
    ap.add_argument('--operators',default='suffix_swap,chain_transfer',
        help='Can additionally include node_swap_reconnect,segment_reverse')
    ap.add_argument('--oracle-samples',type=int,default=8,
        help='Per original, verify up to this many fast survivors against reference_static_feasible')
    args=ap.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    ops=[s.strip() for s in args.operators.split(',') if s.strip()]
    if not ops or len(set(ops))!=len(ops) or not set(ops)<=set(SUPPORTED_OPERATORS):
        raise ValueError(f'Invalid operators: {ops}')
    chosen=CIRCUITS if not args.circuits else [tuple(s.split('::',1)) for s in args.circuits.split(',')]
    import qiskit
    from qiskit import QuantumCircuit
    if qiskit.__version__!='0.45.3':
        raise RuntimeError('Must use Qiskit 0.45.3, got '+qiskit.__version__)
    audits,originals=read_sources(args.audit_dirs)
    rows=[]; issues=[]
    for source,filename in chosen:
        cid=source+'::'+filename
        try:
            a=audits[(source,filename)]
            if a['status']!='screened':
                raise ValueError(f'Unexpected audit status={a["status"]}')
            raw=QuantumCircuit.from_qasm_file(a['qasm_path'])
            qc,_=compact_active(raw)
            ck=make_checker(qc)
            t0=ms();eng=ValidityAwareEngine(ck); setup_ms=ms()-t0
            source_plans=[]
            for r in sorted(originals[(source,filename)],key=lambda t:t['plan_id']):
                plan=canonical(json.loads(r['chains_json']))
                if len(plan)>=qc.num_qubits:continue
                if not ck.validate(plan,target_width=len(plan)).is_valid:
                    raise AssertionError('Invalid source plan: '+r['plan_id'])
                source_plans.append((r['plan_id'],plan))
            if not source_plans:
                raise ValueError('No reused original plans')
            for source_sig,plan in source_plans[:args.max_originals]:
                old_seen=set()
                prep=eng.prepare(plan)
                surviving=[]
                for op in ops:
                    t0=ms()
                    old,attempts=legacy_scan(plan,ck,op,old_seen,args.max_attempts_per_op)
                    t_old=ms()-t0
                    t1=ms()
                    new,stats=prep.scan(op,max_attempts=args.max_attempts_per_op,verify_survivors=True)
                    t_new=ms()-t1
                    matches=(new==old)
                    if not matches:
                        misses=[list(map(list,p)) for p in old if p not in set(new)]
                        extras=[list(map(list,p)) for p in new if p not in set(old)]
                        issues.append(dict(circuit_id=cid,original_sig=source_sig,operator=op,
                            issue='parity_mismatch',misses=misses[:4],extras=extras[:4]))
                    if attempts!=stats['attempted']:
                        issues.append(dict(circuit_id=cid,original_sig=source_sig,operator=op,
                            issue='attempt_count_mismatch',legacy=attempts,fast=stats['attempted']))
                    rows.append(dict(circuit_id=cid,original_sig=source_sig,width=len(plan),operator=op,
                        attempted=attempts,legacy_valid=len(old),fast_valid=len(new),
                        parity='PASS' if matches and attempts==stats['attempted'] else 'FAIL',
                        legacy_scan_ms=round(t_old,3),fast_scan_ms=round(t_new,3),
                        engine_setup_ms=round(setup_ms,3),
                        fast_reject_illegal=stats['fast_reject_illegal_boundary'],
                        fast_reject_cycle=stats['fast_reject_dependency_cycle'],
                        duplicate=stats['duplicate'],same_plan=stats['same_plan']))
                    surviving.extend(new)
                # Compare a deterministic sample against the wholly separate
                # operation-level reference, rather than the bitset fast model.
                rng=random.Random(311)
                surviving=list(dict.fromkeys(surviving))
                rng.shuffle(surviving)
                for candidate in surviving[:args.oracle_samples]:
                    if not reference_static_feasible(qc,candidate,len(plan)):
                        issues.append(dict(circuit_id=cid,original_sig=source_sig,
                            issue='independent_oracle_mismatch',plan=candidate))
            print(f'[parity] {cid}: {len(source_plans[:args.max_originals])} original(s); '
                  f'{sum(r["attempted"] for r in rows if r["circuit_id"]==cid)} attempts; '
                  f'{sum(r["fast_valid"] for r in rows if r["circuit_id"]==cid)} accepted; '
                  f'{"PASS" if not any(i["circuit_id"]==cid for i in issues) else "FAIL"}',flush=True)
        except Exception as e:
            issues.append(dict(circuit_id=cid,issue='exception',error=repr(e),trace=traceback.format_exc(limit=6)))
            print(f'[ERROR] {cid}: {type(e).__name__}: {e}',file=sys.stderr,flush=True)
    fields=['circuit_id','original_sig','width','operator','attempted','legacy_valid','fast_valid','parity',
            'legacy_scan_ms','fast_scan_ms','engine_setup_ms','fast_reject_illegal','fast_reject_cycle','duplicate','same_plan']
    write_csv(args.out_dir/'parity_report.csv',rows,fields)
    (args.out_dir/'parity_errors.json').write_text(json.dumps(issues,indent=2,ensure_ascii=False),encoding='utf-8')
    summary=dict(circuits_requested=len(chosen),circuits_checked=len(set(r['circuit_id'] for r in rows)),
        original_operator_checks=len(rows),failures=len(issues),
        all_pass=bool(rows) and not issues and all(r['parity']=='PASS' for r in rows),
        raw_attempts=sum(r['attempted'] for r in rows),
        legacy_valid_total=sum(r['legacy_valid'] for r in rows),
        fast_valid_total=sum(r['fast_valid'] for r in rows),
        fast_reject_illegal_total=sum(r['fast_reject_illegal'] for r in rows),
        fast_reject_cycle_total=sum(r['fast_reject_cycle'] for r in rows),
        legacy_scan_ms_total=round(sum(r['legacy_scan_ms'] for r in rows),3),
        fast_scan_ms_total=round(sum(r['fast_scan_ms'] for r in rows),3),
        setup_ms_total_per_circuit=round(sum({r['circuit_id']:r['engine_setup_ms'] for r in rows}.values()),3),
        explanation='Static-feasibility parity only; scan times exclude engine setup, which is reported separately. No routing done.')
    (args.out_dir/'parity_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    if not summary['all_pass']:
        sys.exit(1)

if __name__=='__main__':main()
