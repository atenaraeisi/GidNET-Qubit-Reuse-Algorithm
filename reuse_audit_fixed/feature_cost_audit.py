#!/usr/bin/env python3
"""Read-only, pre-routing audit of Experiment-4D plan features.

Consumes original_plans.csv and circuit_audit.csv from one or more
reuse_feasibility_audit_fixed.py output directories, plus exp4d_features.py.

For each supported circuit, validate up to N existing original plans and sample
up to K VALID one-step neighbours from the ORIGINAL operators. Profiles are
cached once per circuit; times for feature families are measured per plan.

This is NOT a physical-improvement evaluation, full plan-space enumeration, or
feature-selection result. Logical-depth-from-dynamic-compilation is excluded:
its extraction code and costs are not supplied here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

FAMILIES = {
    'quotient': ['quotient_unique_edges', 'quotient_density', 'max_degree',
                 'max_weighted_degree', 'max_edge_weight'],
    'boundary': ['mean_raw_jaccard_incompat','max_raw_jaccard_incompat',
                 'top2_raw_jaccard_incompat','mean_raw_weighted_incompat',
                 'max_raw_weighted_incompat','mean_chain_jaccard_incompat',
                 'max_chain_jaccard_incompat','top2_chain_jaccard_incompat',
                 'mean_chain_weighted_incompat','max_chain_weighted_incompat',
                 'mean_first_2q_local_delay','mean_first_2q_layer_delay'],
    'concentration': ['edge_weight_cv','max_edge_share', 'edge_weight_hhi',
                      'normalized_edge_entropy','edge_weight_concentration',
                      'weighted_degree_cv'],
    'extras': ['extra_internal_2q_frac','extra_cross_2q'],
}
FEATURES = [x for group in FAMILIES.values() for x in group]
REDUNDANT_PAIRS = [
    ('normalized_edge_entropy','edge_weight_concentration', 'second = 1 - first, for finite values'),
    ('quotient_unique_edges','quotient_density','second = first / choose(fixed_width,2), if width > 1'),
    ('extra_internal_2q_frac','extra_cross_2q','second = total_2q * (1 - first), for each fixed circuit'),
]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location('experiment4d_feature_functions', str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_inputs(dirs):
    original, circuits = [], []
    for d in dirs:
        p, q = d / 'original_plans.csv', d / 'circuit_audit.csv'
        if not p.exists() or not q.exists():
            raise FileNotFoundError(f'Expected {p} AND {q}.')
        original.append(pd.read_csv(p))
        circuits.append(pd.read_csv(q))
    orig = pd.concat(original, ignore_index=True).drop_duplicates(
        subset=['source','filename','plan_id'])
    cir = pd.concat(circuits, ignore_index=True).drop_duplicates(
        subset=['source','filename'])
    return orig, cir


def assert_original_supported(qc):
    """Reject input containing mid-circuit measurement, reset, or classical control."""
    from qiskit.circuit import Gate
    measured = set()
    for i,inst in enumerate(qc.data):
        op = inst.operation
        qids = {qc.find_bit(q).index for q in inst.qubits}
        if getattr(op, 'condition', None) is not None:
            raise ValueError(f'unsupported classical condition at operation {i}')
        if op.name == 'barrier':
            continue
        if op.name == 'measure':
            if len(qids)!=1 or len(inst.clbits)!=1:
                raise ValueError(f'unsupported measurement at operation {i}')
            measured.update(qids)
            continue
        if measured & qids:
            raise ValueError(f'mid-circuit measurement at operation {i}')
        if not isinstance(op, Gate):
            raise ValueError(f'unsupported nonunitary operation {op.name} at operation {i}')


def valid_chains(x):
    z = json.loads(x)
    return tuple(sorted(tuple(int(q) for q in chain) for chain in z))


def plan_id(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode('utf-8')).hexdigest()[:20]


def candidate_neighbours(chains, ck, *, per_operator=6, max_attempts_per_operator=300):
    from reuse_adversarial import canonical, OPERATORS
    seen = {canonical(chains)}
    output = []
    attempts = {}
    for op in ('suffix_swap','chain_transfer'):
        checked, accepted = 0, 0
        for p in OPERATORS[op](chains):
            if checked >= max_attempts_per_operator or accepted >= per_operator:
                break
            checked += 1
            if not p or not all(p):
                continue
            p = canonical(p)
            if p in seen:
                continue
            seen.add(p)
            if ck.validate(p, target_width=len(chains)).is_valid:
                output.append((op, p))
                accepted += 1
        attempts[op] = checked
    return output, attempts


def profile_one(mod, qc, profiles, timing, plan):
    started = time.perf_counter_ns()
    q, ew = mod.quotient_interaction_metrics(qc, plan)
    tq = time.perf_counter_ns()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        b = mod.aggregate_boundary_features(profiles, timing, plan)
    tb = time.perf_counter_ns()
    c = mod.edge_concentration_metrics(ew, len(plan))
    tc = time.perf_counter_ns()
    extras = {
        'extra_internal_2q_frac': q['internal_2q'] / q['total_2q'] if q['total_2q'] else 0.0,
        'extra_cross_2q': q['total_2q'] - q['internal_2q'],
    }
    features = {k:v for k,v in q.items() if k not in ('total_2q','internal_2q')}
    features.update(b); features.update(c); features.update(extras)
    if set(features) != set(FEATURES):
        raise RuntimeError('Feature list differs from Experiment-4D definitions.')
    return features, {
        'quotient_ms': (tq-started)/1e6,
        'boundary_ms': (tb-tq)/1e6,
        'concentration_ms': (tc-tb)/1e6,
        'total_ms': (tc-started)/1e6,
    }


def write_reports(rows, circuit_rows, out):
    out.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(rows)
    c = pd.DataFrame(circuit_rows)
    c.to_csv(out/'circuit_summary.csv', index=False)
    if d.empty:
        pd.DataFrame(columns=['feature','family','missing_fraction','circuits_variable_fraction','overall_std']).to_csv(out/'feature_summary.csv', index=False)
        return d
    d.to_csv(out/'feature_values.csv', index=False)
    frows = []
    for fam, fields in FAMILIES.items():
        for field in fields:
            v = pd.to_numeric(d[field], errors='coerce').replace([np.inf,-np.inf],np.nan)
            cv = []
            for _,g in d.groupby(['source','filename']):
                x = pd.to_numeric(g[field], errors='coerce').replace([np.inf,-np.inf],np.nan).dropna()
                if len(x) >= 2:
                    cv.append(x.max()-x.min()>1e-12)
            frows.append(dict(feature=field, family=fam,
                              missing_fraction=round(float(v.isna().mean()),4),
                              circuits_variable_fraction=round(float(np.mean(cv)),4) if cv else np.nan,
                              total_observed=int(v.notna().sum()),
                              overall_std=float(v.std()) if v.notna().sum()>=2 else np.nan,
                              **{fam+'_mean_ms':float(d[fam+'_ms'].mean()) if fam!='extras' else 0.0}))
    pd.DataFrame(frows).to_csv(out/'feature_summary.csv',index=False)
    report = []
    for a,b,relationship in REDUNDANT_PAIRS:
        maxerr, nchecked = 0.0, 0
        for (_,fn), g in d.groupby(['source','filename']):
            x = pd.to_numeric(g[a],errors='coerce').to_numpy(float)
            y = pd.to_numeric(g[b],errors='coerce').to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            if not ok.any():
                continue
            if a == 'normalized_edge_entropy':
                error = np.max(np.abs((1-x[ok])-y[ok]))
            elif a == 'quotient_unique_edges':
                w = g['active_width'].iloc[0]
                if w <= 1:
                    continue
                error = np.max(np.abs(x[ok]/(w*(w-1)/2)-y[ok]))
            else:
                # total number of 2q gates per source circuit is constant across plans;
                # infer total from any plan with internal fraction != 1.
                u = 1-x[ok]
                mask = np.abs(u)>1e-12
                if not mask.any():
                    continue
                scale = float(np.median(y[ok][mask]/u[mask]))
                error = np.max(np.abs(y[ok]-scale*u))
            maxerr = max(maxerr,float(error))
            nchecked += int(ok.sum())
        report.append(dict(first=a,second=b,mathematical_relation=relationship,
                           n_checked=nchecked,max_absolute_error=maxerr if nchecked else np.nan))
    pd.DataFrame(report).to_csv(out/'known_redundancies.csv',index=False)
    # Other correlations are only descriptive; not supervised relevance evidence.
    delta_rows=[]
    for (_,_,oid),g in d.groupby(['source','filename','original_plan_id']):
        base = g[g['kind']=='original']
        if len(base)!=1:
            continue
        for _,r in g[g['kind']=='neighbor'].iterrows():
            diffs={}
            for f in FEATURES:
                a,b=r[f],base.iloc[0][f]
                diffs[f]=float(a)-float(b) if pd.notna(a) and pd.notna(b) else np.nan
            delta_rows.append(diffs)
    pairs=[]
    if len(delta_rows)>2:
        delta=pd.DataFrame(delta_rows)
        corr=delta.corr(method='spearman', min_periods=3)
        for i,a in enumerate(FEATURES):
            for b in FEATURES[i+1:]:
                rho=corr.loc[a,b]
                if pd.notna(rho) and abs(float(rho))>=0.9:
                    pairs.append(dict(feature_a=a,feature_b=b,delta_spearman=float(rho)))
    pd.DataFrame(pairs, columns=['feature_a','feature_b','delta_spearman']).sort_values(
        by='delta_spearman', key=lambda x:x.abs(),ascending=False
    ).to_csv(out/'high_correlations_exploratory.csv',index=False)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audit-dirs', nargs='+', type=Path, required=True)
    ap.add_argument('--feature-code',type=Path,default=Path('exp4d_features.py'))
    ap.add_argument('--out-dir',type=Path,default=Path('feature_audit_out'))
    ap.add_argument('--max-circuits',type=int,default=0,help='0 means all circuits')
    ap.add_argument('--only-circuits',default='',help='comma-separated filenames for a short smoke run')
    ap.add_argument('--max-originals',type=int,default=2)
    ap.add_argument('--neighbors-per-op',type=int,default=6)
    ap.add_argument('--max-attempts-per-op',type=int,default=300)
    args=ap.parse_args()
    mod=load_module(args.feature_code)
    orig, cir = read_inputs(args.audit_dirs)
    try:
        from qiskit import QuantumCircuit
        from reuse_validity import ReuseValidityChecker
    except ImportError as exc:
        sys.exit(f'Activate the Qiskit 0.45.x env and run from the directory with reuse_validity.py: {exc}')
    entries = cir.sort_values(['source','filename']).to_dict('records')
    if args.only_circuits:
        wanted = {x.strip() for x in args.only_circuits.split(',')}
        entries = [x for x in entries if x['filename'] in wanted]
    if args.max_circuits:
        entries=entries[:args.max_circuits]
    rows=[];reports=[]
    for i,record in enumerate(entries,1):
        src,fn=record['source'],record['filename']
        info=dict(source=src,filename=fn,feature_status='skip',n_plans_profiled=0,
                  n_neighbors_profiled=0,feature_profile_ms=np.nan,
                  preparation_ms=np.nan,error='')
        g=orig[(orig.source==src)&(orig.filename==fn)]
        try:
            if str(record.get('status','')) in ('skip_parser','error'):
                info['error']='Original screening flagged as parser/error'
            elif len(g)==0:
                info['error']='No original plans recorded'
            elif not Path(str(record['qasm_path'])).is_file():
                info['error']='QASM path unavailable: '+str(record['qasm_path'])
            else:
                tpre=time.perf_counter_ns()
                qc0=QuantumCircuit.from_qasm_file(str(record['qasm_path']))
                assert_original_supported(qc0)
                qc,old_to_new=mod.prepare_gidnet_input(qc0)
                ck=ReuseValidityChecker(qc)
                profiles,timing=mod.build_interaction_profiles(qc)
                info['preparation_ms']=(time.perf_counter_ns()-tpre)/1e6
                local_rows=[];local_neigh=0;total_attempts=Counter()
                tested=0
                for r in g.itertuples(index=False):
                    if tested>=args.max_originals: break
                    p=valid_chains(r.chains_json)
                    if len(p)!=int(r.active_width) or not ck.validate(p,target_width=int(r.active_width)).is_valid:
                        raise ValueError(f'Original {r.plan_id}: invalid after compaction; cannot audit')
                    if len(p)==qc.num_qubits:
                        continue # no reuse; no candidate features to compare
                    tested +=1
                    ps=[('original',p,'none')]
                    ns,att=candidate_neighbours(p,ck,per_operator=args.neighbors_per_op,
                                               max_attempts_per_operator=args.max_attempts_per_op)
                    total_attempts.update(att)
                    ps.extend(('neighbor',z,operator) for operator,z in ns)
                    local_neigh+=len(ns)
                    for kind,pplan,operator in ps:
                        vals,times=profile_one(mod,qc,profiles,timing,pplan)
                        local_rows.append(dict(source=src,filename=fn,original_plan_id=str(r.plan_id),
                                               plan_signature=plan_id(pplan), kind=kind,operator=operator,
                                               active_width=len(pplan),qubits_active=qc.num_qubits,
                                               **vals,**times))
                rows.extend(local_rows)
                info['n_plans_profiled']=len(local_rows)
                info['n_neighbors_profiled']=local_neigh
                info['feature_profile_ms']=sum(x['total_ms'] for x in local_rows)
                info['mutation_attempts']=sum(total_attempts.values())
                info['feature_status']='ok' if local_rows else 'no_reuse_plan'
        except Exception as exc:
            info['feature_status']='error';info['error']=f'{type(exc).__name__}: {exc}'[:450]
        reports.append(info)
        print(f"[{i}/{len(entries)}] {src}/{fn}: {info['feature_status']}; plans={info['n_plans_profiled']} neighbors={info['n_neighbors_profiled']}",flush=True)
    values=write_reports(rows,reports,args.out_dir)
    metadata=dict(circuits_requested=len(entries),circuits_with_feature_profiles=sum(r['feature_status']=='ok' for r in reports),
                  profiled_plan_rows=len(values),feature_count=len(FEATURES),
                  note='No routing, no training. Only 25 plan_features attributes; logical_depth and norm_depth from 4C-1 are excluded. Low-candidate sampling is not full-search evaluation.',
                  dependencies=['Qiskit 0.45.x','exp4d_features.py','reuse_validity.py','reuse_adversarial.py'])
    (args.out_dir/'audit_readme.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('Saved:', args.out_dir.resolve(), '\n',json.dumps(metadata,indent=2))

if __name__=='__main__':
    main()
