# Audit-only compatibility extension for missing first-2q timing.
# All defined-delay outputs match the original exp4d_features.py.
# Missing timing is NaN (never zero). Do not silently use this code in an
# already frozen training protocol without redoing validation.
#!/usr/bin/env python3
"""Experiment 4D, step 1: Experiment-2 plan features for EVERY plan in the 4C-1 catalog.

The feature definitions are copied from experiment2_reuse_boundary_analysis.ipynb
(build_interaction_profiles, boundary_context, compute_boundary_diagnostics,
quotient_interaction_metrics, edge_concentration_metrics, aggregate_boundary_features),
so originals and neighbours are described on exactly the same scale.

Before trusting the neighbours, run with --validate experiment2_plan_features.csv : the
features recomputed for the 50 original plans must reproduce your Experiment-2 table
(max abs difference <= 1e-9).  The script exits with an error otherwise.

Run from the project root, same Python environment as Experiment 2:
  python exp4d_features.py --catalog exp4c1_out/catalog.csv --qasm-dir <benchmark dir> \
         --out exp4d_features.csv --validate experiment2_plan_features.csv
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

K = 5
E2_COLUMNS = [
    'quotient_unique_edges', 'quotient_density', 'max_degree', 'max_weighted_degree', 'max_edge_weight',
    'mean_raw_jaccard_incompat', 'max_raw_jaccard_incompat', 'top2_raw_jaccard_incompat',
    'mean_raw_weighted_incompat', 'max_raw_weighted_incompat',
    'mean_chain_jaccard_incompat', 'max_chain_jaccard_incompat', 'top2_chain_jaccard_incompat',
    'mean_chain_weighted_incompat', 'max_chain_weighted_incompat',
    'mean_first_2q_local_delay', 'mean_first_2q_layer_delay',
    'edge_weight_cv', 'max_edge_share', 'edge_weight_hhi', 'normalized_edge_entropy',
    'edge_weight_concentration', 'weighted_degree_cv', 'normalized_logical_depth',
]


# ---------------------------------------------------------------- circuit preparation
def prepare_gidnet_input(qc):
    """Notebook cell 'Step 3': compact circuit without idle qubits.
    Terminal measure/barrier ops are dropped (they do not exist in the Experiment-2 inputs)."""
    from qiskit import QuantumCircuit
    kept = []
    for item in qc.data:
        op = item.operation
        if op.name in ('measure', 'barrier'):
            continue
        if len(item.clbits) > 0 or getattr(op, 'condition', None) is not None:
            raise ValueError(f'classical operand/condition in {op.name}')
        kept.append(item)
    active = sorted({qc.find_bit(q).index for item in kept for q in item.qubits})
    old_to_new = {o: n for n, o in enumerate(active)}
    compact = QuantumCircuit(len(active))
    for item in kept:
        compact.append(item.operation.copy(), [old_to_new[qc.find_bit(q).index] for q in item.qubits], [])
    return compact, old_to_new


# ---------------------------------------------------------------- notebook step 3b
def build_interaction_profiles(qc):
    profiles = defaultdict(list)
    first_activity_layer, first_2q_local_delay, first_2q_layer = {}, {}, {}
    local_op_count = defaultdict(int)
    last_layer = defaultdict(lambda: -1)
    for op_index, item in enumerate(qc.data):
        op = item.operation
        if op.name == 'barrier':
            continue
        qs = [qc.find_bit(q).index for q in item.qubits]
        if not qs:
            continue
        layer = max(last_layer[q] for q in qs) + 1
        for q in qs:
            last_layer[q] = layer
            first_activity_layer.setdefault(q, layer)
        if len(qs) == 2:
            qa, qb = qs
            for q in (qa, qb):
                if q not in first_2q_local_delay:
                    first_2q_local_delay[q] = local_op_count[q]
                    first_2q_layer[q] = layer
            profiles[qa].append({'partner': qb, 'layer': layer})
            profiles[qb].append({'partner': qa, 'layer': layer})
        for q in qs:
            local_op_count[q] += 1
    timing = {}
    for q in range(qc.num_qubits):
        timing[q] = {
            'local': first_2q_local_delay.get(q),
            'layer': (first_2q_layer[q] - first_activity_layer[q]) if q in first_2q_layer else None,
        }
    return profiles, timing


def boundary_context(profiles, timing, pred, succ, k=K):
    return ([e['partner'] for e in profiles[pred][-k:]], [e['partner'] for e in profiles[succ][:k]],
            timing[succ]['local'], timing[succ]['layer'])


# ---------------------------------------------------------------- notebook step 4a
def jaccard_similarity(a, b):
    a, b = set(a), set(b)
    u = a | b
    return 1.0 if not u else len(a & b) / len(u)


def weighted_jaccard(sa, sb):
    ca, cb = Counter(sa), Counter(sb)
    keys = set(ca) | set(cb)
    if not keys:
        return 1.0
    return sum(min(ca[k], cb[k]) for k in keys) / sum(max(ca[k], cb[k]) for k in keys)


def top2_mean(values):
    values = sorted((x for x in values if pd.notna(x)), reverse=True)
    return float(np.mean(values[:2])) if values else np.nan


def aggregate_boundary_features(profiles, timing, chains):
    chain_map = {q: i for i, c in enumerate(chains) for q in c}
    rj, rw, cj, cw, ld, ly = [], [], [], [], [], []
    for chain in chains:
        for a, b in zip(chain, chain[1:]):
            tail, head, local, layer = boundary_context(profiles, timing, a, b)
            ct, ch = [chain_map[q] for q in tail], [chain_map[q] for q in head]
            rj.append(1 - jaccard_similarity(tail, head)); rw.append(1 - weighted_jaccard(tail, head))
            cj.append(1 - jaccard_similarity(ct, ch)); cw.append(1 - weighted_jaccard(ct, ch))
            # A logical qubit can have no 2-qubit interaction. In that case
        # the 'delay until first 2q gate' is UNDEFINED, not zero.
        ld.append(np.nan if local is None else float(local))
        ly.append(np.nan if layer is None else float(layer))
    return {
        'mean_raw_jaccard_incompat': np.mean(rj), 'max_raw_jaccard_incompat': np.max(rj),
        'top2_raw_jaccard_incompat': top2_mean(rj),
        'mean_raw_weighted_incompat': np.mean(rw), 'max_raw_weighted_incompat': np.max(rw),
        'mean_chain_jaccard_incompat': np.mean(cj), 'max_chain_jaccard_incompat': np.max(cj),
        'top2_chain_jaccard_incompat': top2_mean(cj),
        'mean_chain_weighted_incompat': np.mean(cw), 'max_chain_weighted_incompat': np.max(cw),
        # Pool only defined delays; if all are undefined, preserve missingness.
        'mean_first_2q_local_delay': float(np.nanmean(ld)) if np.isfinite(ld).any() else np.nan,
        'mean_first_2q_layer_delay': float(np.nanmean(ly)) if np.isfinite(ly).any() else np.nan,
    }


# ---------------------------------------------------------------- notebook steps 4c / 5
def quotient_interaction_metrics(qc, chains):
    chain_map = {q: i for i, c in enumerate(chains) for q in c}
    width = len(chains)
    edge_weights, internal, total = Counter(), 0, 0
    for item in qc.data:
        qa = [qc.find_bit(q).index for q in item.qubits]
        if len(qa) != 2:
            continue
        total += 1
        c1, c2 = chain_map[qa[0]], chain_map[qa[1]]
        if c1 == c2:
            internal += 1
            continue
        edge_weights[tuple(sorted((c1, c2)))] += 1
    adj = {i: set() for i in range(width)}
    wdeg = np.zeros(width)
    for (u, v), w in edge_weights.items():
        adj[u].add(v); adj[v].add(u); wdeg[u] += w; wdeg[v] += w
    degrees = np.array([len(adj[i]) for i in range(width)], float)
    weights = np.array(list(edge_weights.values()), float)
    ue = len(edge_weights)
    pe = width * (width - 1) / 2
    return dict(quotient_unique_edges=ue, quotient_density=ue / pe if pe > 0 else 0,
                max_degree=degrees.max(), max_weighted_degree=wdeg.max(),
                max_edge_weight=weights.max() if len(weights) else 0,
                total_2q=total, internal_2q=internal), dict(edge_weights)


def edge_concentration_metrics(edge_weights, width):
    w = np.array(list(edge_weights.values()), float)
    if len(w) == 0:
        return dict(edge_weight_cv=np.nan, max_edge_share=np.nan, edge_weight_hhi=np.nan,
                    normalized_edge_entropy=np.nan, edge_weight_concentration=np.nan, weighted_degree_cv=np.nan)
    m = w.mean()
    p = w / w.sum()
    ent = -float(np.sum(p * np.log(p))) / np.log(len(p)) if len(p) > 1 else 0.0
    wd = np.zeros(width)
    for (u, v), x in edge_weights.items():
        wd[u] += x; wd[v] += x
    return dict(edge_weight_cv=w.std() / m if m > 0 else np.nan, max_edge_share=float(p.max()),
                edge_weight_hhi=float(np.sum(p ** 2)), normalized_edge_entropy=ent,
                edge_weight_concentration=1.0 - ent,
                weighted_degree_cv=wd.std() / wd.mean() if wd.mean() > 0 else np.nan)


def plan_features(qc, profiles, timing, chains):
    q, ew = quotient_interaction_metrics(qc, chains)
    out = {k: v for k, v in q.items() if k not in ('total_2q', 'internal_2q')}
    out.update(aggregate_boundary_features(profiles, timing, chains))
    out.update(edge_concentration_metrics(ew, len(chains)))
    # extras not in Experiment 2
    out['extra_internal_2q_frac'] = q['internal_2q'] / q['total_2q'] if q['total_2q'] else 0.0
    out['extra_cross_2q'] = q['total_2q'] - q['internal_2q']
    return out


# ---------------------------------------------------------------- driver
def canon(ch):
    return tuple(sorted(tuple(int(q) for q in c) for c in ch))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--catalog', required=True)
    ap.add_argument('--qasm-dir', required=True)
    ap.add_argument('--out', default='exp4d_features.csv')
    ap.add_argument('--validate', default='')
    ap.add_argument('--tol', type=float, default=1e-9)
    args = ap.parse_args()

    from qiskit import QuantumCircuit
    cat = pd.read_csv(args.catalog)
    prepared = {}
    rows = []
    for r in cat.itertuples():
        if r.circuit not in prepared:
            qc0 = QuantumCircuit.from_qasm_file(str(Path(args.qasm_dir) / r.circuit))
            qc, o2n = prepare_gidnet_input(qc0)
            if any(o != n for o, n in o2n.items()):
                sys.exit(f'{r.circuit}: compaction is not the identity map ({o2n}); plan indices differ '
                         f'between the 4C-1 catalog and Experiment 2 - fix before continuing.')
            prepared[r.circuit] = (qc, *build_interaction_profiles(qc))
        qc, profiles, timing = prepared[r.circuit]
        chains = canon(json.loads(r.plan_json))
        f = plan_features(qc, profiles, timing, chains)
        rows.append(dict(circuit=r.circuit, sig=r.sig, **f))
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f'wrote {len(out)} plans x {out.shape[1] - 2} features -> {args.out}')

    if args.validate:
        e2 = pd.read_csv(args.validate)
        e2['key'] = e2.reuse_chains.apply(lambda s: canon(ast.literal_eval(s)))
        cat['key'] = cat.plan_json.apply(lambda s: canon(json.loads(s)))
        m = e2.merge(cat[['circuit', 'sig', 'key']], on=['circuit', 'key']).merge(out, on=['circuit', 'sig'],
                                                                                   suffixes=('_e2', '_new'))
        print(f'validation: {len(m)} of {len(e2)} Experiment-2 plans matched in the catalog')
        bad = []
        for col in E2_COLUMNS:
            if col == 'normalized_logical_depth':
                continue          # comes from the GidNET run, not from this script
            d = (m[col + '_e2'] - m[col + '_new']).abs().max()
            print(f'  {col:34s} max|diff| = {d:.3g}')
            if not d <= args.tol:
                bad.append(col)
        if len(m) != len(e2) or bad:
            sys.exit(f'VALIDATION FAILED (matched {len(m)}/{len(e2)}; mismatching columns: {bad}). '
                     f'Do not use the neighbour features until this is resolved.')
        print('VALIDATION PASSED: neighbour features are on the same scale as Experiment 2.')


if __name__ == '__main__':
    main()
