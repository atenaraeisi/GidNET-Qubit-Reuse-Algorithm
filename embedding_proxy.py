#!/usr/bin/env python3
"""Experiment 8 (2): a topology-aware, pre-routing cost proxy for a reuse plan.

Idea: `rule_hotspot` (max_weighted_degree of the quotient interaction graph) is BLIND to the
target coupling map -- it would rank plans identically on hh7, grid and line.  But
topology_sweep.py showed the quality ranking of plans only correlates ~0.3-0.7 (Spearman)
across topologies, so a hardware-blind feature cannot have a high ceiling.

This script builds a CHEAP, topology-AWARE proxy instead:
  1. the quotient interaction graph of the plan (same construction as exp4d_features.py's
     quotient_interaction_metrics, so max_weighted_degree stays comparable);
  2. a fast greedy embedding of that graph's chains onto the target coupling graph
     (no SABRE, no iterative SWAP search -- just nearest-free-neighbour placement);
  3. embedding_cost = sum over quotient edges of weight * hardware_distance(pos[u], pos[v]).
     embedding_cost_excess subtracts 1 hop (the "free" adjacent step) per edge, which is closer
     to an actual SWAP-count proxy.

Stages
  compute  : for every (circuit, sig) in a 4C-1 catalog.csv and every topology, place the plan
             and write embedding_features.csv (circuit, sig, topology, embedding_cost,
             embedding_cost_excess, embedding_cost_mean, placement_seconds).
  evaluate : correlate the proxy (and, if given, the existing hotspot feature) with the REAL
             routed cost in topo_out/<topology>/cost_table.csv, per circuit and pooled -- same
             format as exp4d's feature_signal.csv, so the two are directly comparable.

Usage
  python embedding_proxy.py compute  --catalog exp4c1_all/catalog.csv --qasm-dir benchmarks_compact \
         --topologies hh7,grid,line --out embedding_features.csv
  python embedding_proxy.py evaluate --features embedding_features.csv --topo-dir topo_out \
         --topologies hh7,grid,line --hotspot-features exp4d_results_all/exp4d_features_all.csv \
         --out embedding_signal.csv
"""
import argparse, ast, json, os, sys, time
from collections import deque

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from router_topo import coupling                       # same topology definitions used for routing
from exp4d_features import quotient_interaction_metrics  # same quotient graph as the hotspot feature


# --------------------------------------------------------------------- hardware graph
def hw_adjacency(cm):
    n = cm.size()
    adj = [set() for _ in range(n)]
    for a, b in cm.get_edges():
        adj[a].add(b); adj[b].add(a)
    return adj


def all_pairs_bfs(adj):
    n = len(adj)
    dist = np.full((n, n), -1, dtype=np.int32)
    for s in range(n):
        dist[s, s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if dist[s, v] == -1:
                    dist[s, v] = dist[s, u] + 1
                    dq.append(v)
    if (dist < 0).any():
        raise RuntimeError('hardware graph is disconnected')
    return dist


# --------------------------------------------------------------------- greedy embedding
def greedy_embed(width, edge_weights, dist):
    """Place `width` quotient nodes on the hardware graph, minimising sum(weight*distance).
    Pure nearest-neighbour greedy construction (no local search) -- O(width^2 * hw_size)."""
    n_hw = dist.shape[0]
    nb_weight = [{} for _ in range(width)]
    for (u, v), w in edge_weights.items():
        nb_weight[u][v] = w; nb_weight[v][u] = w
    wdeg = np.array([sum(nb_weight[i].values()) for i in range(width)], float)
    order = list(np.argsort(-wdeg))                     # heaviest chains placed first
    pos = {}
    used_hw = np.zeros(n_hw, dtype=bool)
    # anchor: heaviest chain -> most-connected hardware node (a reasonable, cheap seed)
    hw_degree = (dist == 1).sum(axis=1)
    first = order[0]
    anchor_hw = int(np.argmax(hw_degree))
    pos[first] = anchor_hw; used_hw[anchor_hw] = True
    for q in order[1:]:
        placed_nb = [(p, nb_weight[q][p]) for p in nb_weight[q] if p in pos]
        free = np.where(~used_hw)[0]
        if placed_nb:
            cost = np.zeros(len(free))
            for p, w in placed_nb:
                cost += w * dist[pos[p], free]
        else:                                             # no placed neighbour yet: stay close to the anchor
            cost = dist[anchor_hw, free].astype(float)
        best = free[int(np.argmin(cost))]
        pos[q] = int(best); used_hw[best] = True
    return pos


def embedding_cost(width, edge_weights, dist):
    pos = greedy_embed(width, edge_weights, dist)
    tot = tot_excess = tot_w = 0.0
    for (u, v), w in edge_weights.items():
        d = dist[pos[u], pos[v]]
        tot += w * d
        tot_excess += w * max(d - 1, 0)
        tot_w += w
    n_edges = len(edge_weights)
    return dict(embedding_cost=tot, embedding_cost_excess=tot_excess,
                embedding_cost_mean=(tot / tot_w if tot_w else 0.0),
                embedding_cost_excess_mean=(tot_excess / tot_w if tot_w else 0.0),
                n_quotient_edges=n_edges, sum_weight=tot_w)


# --------------------------------------------------------------------- IO helpers
def parse_chains(x):
    v = json.loads(x) if isinstance(x, str) else x
    return [[int(q) for q in c] for c in v]


def load_qc(path):
    return QuantumCircuit.from_qasm_file(path)


# --------------------------------------------------------------------- stages
def cmd_compute(a):
    cat = pd.read_csv(a.catalog)
    topos = a.topologies.split(',')
    dists = {}
    for t in topos:
        cm = coupling(t)
        t0 = time.time()
        dists[t] = all_pairs_bfs(hw_adjacency(cm))
        print(f'[{t}] hardware graph {cm.size()} nodes, all-pairs BFS in {time.time()-t0:.1f}s')
    rows = []
    qc_cache = {}
    for i, r in enumerate(cat.itertuples(), 1):
        if r.circuit not in qc_cache:
            qc_cache[r.circuit] = load_qc(os.path.join(a.qasm_dir, r.circuit))
        qc = qc_cache[r.circuit]
        chains = parse_chains(r.plan_json)
        width = len(chains)
        _, edge_weights = quotient_interaction_metrics(qc, chains)
        for t in topos:
            if width > dists[t].shape[0]:
                continue
            t0 = time.time()
            feat = embedding_cost(width, edge_weights, dists[t])
            feat['placement_seconds'] = time.time() - t0
            rows.append(dict(circuit=r.circuit, sig=r.sig, topology=t, width=width, **feat))
        if i % 200 == 0:
            print(f'[compute] {i}/{len(cat)}', flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(f'wrote {len(out)} rows -> {a.out}')


def mean_costs(cost_path):
    c = pd.read_csv(cost_path)
    c = c[c.status == 'ok']
    return c.groupby(['circuit', 'sig'])[['swap_count', 'routed_depth']].mean().reset_index()


def spearman_rows(df, feat_col, objectives, label):
    rows = []
    for o in objectives:
        sub = df[['circuit', feat_col, o]].dropna()
        pooled = sub[feat_col].corr(sub[o], method='spearman')
        per_c = sub.groupby('circuit').apply(
            lambda g: g[feat_col].corr(g[o], method='spearman') if g[feat_col].nunique() > 1 and g[o].nunique() > 1 else np.nan)
        rows.append(dict(objective=o, feature=label, pooled_spearman=pooled,
                          mean_circuit_spearman=per_c.mean(), circuits_negative_sign=int((per_c < 0).sum()),
                          circuits_positive_sign=int((per_c > 0).sum()), n_circuits=int(per_c.notna().sum()),
                          n_pairs=len(sub)))
    return rows


def cmd_evaluate(a):
    emb = pd.read_csv(a.features)
    hot = None
    if a.hotspot_features:
        h = pd.read_csv(a.hotspot_features)
        hot = h[['circuit', 'sig', 'max_weighted_degree']].drop_duplicates(['circuit', 'sig'])
    topos = a.topologies.split(',')
    all_rows = []
    for t in topos:
        cp = os.path.join(a.topo_dir, t, 'cost_table.csv')
        if not os.path.exists(cp):
            print('skip', t, '(no cost table)'); continue
        mc = mean_costs(cp)
        e = emb[emb.topology == t].merge(mc, on=['circuit', 'sig'], how='inner')
        for col, label in [('embedding_cost', 'embedding_cost'), ('embedding_cost_excess', 'embedding_cost_excess'),
                            ('embedding_cost_mean', 'embedding_cost_mean')]:
            for row in spearman_rows(e, col, ['swap_count', 'routed_depth'], label):
                row['topology'] = t; all_rows.append(row)
        if hot is not None:
            eh = mc.merge(hot, on=['circuit', 'sig'], how='inner')
            for row in spearman_rows(eh, 'max_weighted_degree', ['swap_count', 'routed_depth'], 'max_weighted_degree'):
                row['topology'] = t; all_rows.append(row)
    out = pd.DataFrame(all_rows)
    out.to_csv(a.out, index=False)
    pd.set_option('display.width', 220, 'display.max_columns', 30)
    print(out[['topology', 'objective', 'feature', 'pooled_spearman', 'mean_circuit_spearman',
               'circuits_negative_sign', 'circuits_positive_sign', 'n_circuits']]
          .sort_values(['topology', 'objective', 'feature']).round(3).to_string(index=False))
    print(f'\nwrote {len(out)} rows -> {a.out}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='stage', required=True)

    c = sub.add_parser('compute')
    c.add_argument('--catalog', required=True)
    c.add_argument('--qasm-dir', required=True)
    c.add_argument('--topologies', default='hh7,grid,line')
    c.add_argument('--out', default='embedding_features.csv')

    e = sub.add_parser('evaluate')
    e.add_argument('--features', required=True, help='output of the compute stage')
    e.add_argument('--topo-dir', required=True, help='--out-dir of topology_sweep.py')
    e.add_argument('--topologies', default='hh7,grid,line')
    e.add_argument('--hotspot-features', default='', help='exp4d_features*.csv, for a side-by-side comparison')
    e.add_argument('--out', default='embedding_signal.csv')

    a = ap.parse_args()
    {'compute': cmd_compute, 'evaluate': cmd_evaluate}[a.stage](a)


if __name__ == '__main__':
    main()
