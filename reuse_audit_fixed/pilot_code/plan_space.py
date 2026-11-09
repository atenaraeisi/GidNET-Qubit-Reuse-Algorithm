#!/usr/bin/env python3
"""Experiment 7b: the COMPLETE space of valid fixed-width reuse plans (matching view).

A reuse plan of ACTIVE width w over n active qubits is a set E of K = n - w reuse edges a->b
(terminal of a feeds input of b) such that
  (1) every edge is individually legal (the checker's `allowed[a][b]`, a function of the pair only),
  (2) each qubit has at most one successor and one predecessor   (=> E is a bipartite MATCHING; chains = paths),
  (3) E plus the original dependency DAG is acyclic              (=> the ~3 % `dependency_cycle` rejections).
So "all equivalent plans" = all size-K matchings of the allowed-pair graph that pass (3).  This module
  * builds `allowed` and the event-level reachability from reuse_validity.ReuseValidityChecker (same source of truth),
  * counts size-K matchings exactly by DP (upper bound on the number of valid plans: ignores (3)),
  * enumerates the valid plans exactly by DFS with bit-parallel cycle detection (cap --max-plans),
    or, when the space is bigger than the cap, returns a random sample + the exchange neighbourhood of the sources,
  * measures the EXCHANGE DISTANCE r = |E_source \\ E_plan| (number of reuse edges that change) between plans.
    The four operators of reuse_adversarial.py are exchanges with r <= 3 (suffix_swap/node_swap: r=2,
    chain_transfer: r<=3, segment_reverse: r>=2) - this is verified on the known catalog.
  * writes exp5-compatible files, so `exp5_multistep.py route` / `analyze` give the ceiling over the WHOLE space.

Stages (run from the project root; Qiskit 0.45.3 env; reuse_validity.py + reuse_adversarial.py + exp4c1_build_table.py on path):
  python plan_space.py enumerate --plans plans_all.csv --qasm-dir benchmarks_compact --c1-dir exp4c1_all --out-dir space_out
  python exp5_multistep.py route   --c1-dir exp4c1_all --out-dir space_out --qasm-dir benchmarks_compact --router router_template:route --workers 4
  python exp5_multistep.py analyze --c1-dir exp4c1_all --out-dir space_out --max-depth 3     # D = exchange distance
  (--max-depth 99 = the whole space)
`plans_all.csv` = experiment2_plan_features.csv (old circuits) concatenated with plans_out/plans_new.csv (columns circuit, plan_id, reuse_chains).
Outputs: space_report.csv, catalog5.csv, reach.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())


def sig_of(plan) -> str:                       # identical to exp4c1_build_table.sig_of
    return hashlib.sha1(json.dumps([list(c) for c in plan]).encode()).hexdigest()[:24]


def canonical(chains):
    return tuple(sorted(tuple(int(q) for q in c) for c in chains))


def edges_of(chains):
    return frozenset((int(a), int(b)) for c in chains for a, b in zip(c, c[1:]))


def chains_from_edges(active, edges):
    succ = {a: b for a, b in edges}
    has_pred = {b for _, b in edges}
    chains = []
    for q in active:
        if q in has_pred:
            continue
        c, cur = [q], q
        while cur in succ:
            cur = succ[cur]
            c.append(cur)
        chains.append(tuple(c))
    return canonical(chains)


class PlanSpace:
    """Matching view of the fixed-width plan space of one circuit."""

    def __init__(self, circuit, checker=None):
        import networkx as nx
        if checker is None:
            from reuse_validity import ReuseValidityChecker
            checker = ReuseValidityChecker(circuit)
        self.ck = checker
        self.active = sorted(checker.active_qubits)
        self.n = len(self.active)
        self.pos = {q: i for i, q in enumerate(self.active)}
        n = self.n
        # event nodes: start_i = 2i, finish_i = 2i+1
        ev = []
        for q in self.active:
            ev += [checker.input_nodes[q], checker.output_nodes[q]]
        eidx = {node: k for k, node in enumerate(ev)}
        self.reach0 = []
        for node in ev:
            m = 0
            for d in nx.descendants(checker.original_graph, node):
                k = eidx.get(d)
                if k is not None:
                    m |= 1 << k
            self.reach0.append(m)
        self.allowed = [[False] * n for _ in range(n)]
        for a in self.active:
            for b in self.active:
                if a != b and (a, b) not in checker._reachable_to_output:
                    self.allowed[self.pos[a]][self.pos[b]] = True
        self.allowed_mask = [sum(1 << j for j in range(n) if self.allowed[i][j]) for i in range(n)]

    # ---------- acyclicity with bit-parallel closure over the 2n event nodes
    def _add_edge(self, reach, a, b):
        """Return the new closure after adding finish_a -> start_b, or None if that creates a cycle."""
        u, v = 2 * a + 1, 2 * b
        if (reach[v] >> u) & 1 or u == v:
            return None
        new = reach[v] | (1 << v)
        out = list(reach)
        bu = 1 << u
        for x in range(len(out)):
            if x == u or out[x] & bu:
                out[x] |= new
        return out

    def closure_of(self, edge_pos):
        reach = list(self.reach0)
        for a, b in edge_pos:
            reach = self._add_edge(reach, a, b)
            if reach is None:
                return None
        return reach

    def is_acyclic(self, edges):
        return self.closure_of([(self.pos[a], self.pos[b]) for a, b in edges]) is not None

    # ---------- counting (upper bound: ignores global acyclicity)
    def matching_counts(self):
        """counts[k] = number of size-k matchings of the allowed graph (exact big ints)."""
        n = self.n
        dp = {0: 1}
        for a in range(n):
            nd = dict(dp)
            for mask, c in dp.items():
                m = self.allowed_mask[a] & ~mask
                while m:
                    low = m & -m
                    nd[mask | low] = nd.get(mask | low, 0) + c
                    m ^= low
            dp = nd
        counts = [0] * (n + 1)
        for mask, c in dp.items():
            counts[bin(mask).count('1')] += c
        return counts

    # ---------- enumeration
    def enumerate_plans(self, K, max_plans=50000, rng=None, first_only=False):
        """DFS over qubits in order; each picks a successor or none.  Returns (list of edge frozensets, complete?)."""
        n = self.n
        out, st = [], {'trunc': False}
        order = list(range(n))
        if rng is not None:
            order = list(rng.permutation(n))
        am = self.allowed_mask

        def rec(i, cnt, edges, used_in, reach):
            if st['trunc']:
                return
            if cnt == K:
                out.append(frozenset((self.active[a], self.active[b]) for a, b in edges))
                if first_only or len(out) >= max_plans:
                    st['trunc'] = len(out) >= max_plans or first_only
                return
            if i == n:
                return
            # bound: remaining outs that still have a free allowed successor
            poss = sum(1 for k in range(i, n) if am[order[k]] & ~used_in)
            if cnt + poss < K:
                return
            a = order[i]
            cand = [b for b in range(n) if (am[a] >> b) & 1 and not (used_in >> b) & 1]
            if rng is not None:
                cand = [cand[j] for j in rng.permutation(len(cand))]
            for b in cand:
                nr = self._add_edge(reach, a, b)
                if nr is None:
                    continue
                edges.append((a, b))
                rec(i + 1, cnt + 1, edges, used_in | (1 << b), nr)
                edges.pop()
                if st['trunc']:
                    return
            rec(i + 1, cnt, edges, used_in, reach)

        sys.setrecursionlimit(10000)
        rec(0, 0, [], 0, list(self.reach0))
        return out, not st['trunc']

    def sample_plans(self, K, n_samples, seed=0, tries_factor=20):
        rng = np.random.default_rng(seed)
        seen, res = set(), []
        for _ in range(n_samples * tries_factor):
            plans, _ = self.enumerate_plans(K, rng=rng, first_only=True)
            if plans and plans[0] not in seen:
                seen.add(plans[0]); res.append(plans[0])
                if len(res) >= n_samples:
                    break
        return res

    # ---------- exchange neighbourhood (equal-size matchings differing in <= r_max edges)
    def exchange_neighbours(self, edges, r_max=2, cap=200000):
        E = sorted(edges)
        n, am = self.n, self.allowed_mask
        found = set()
        pos = lambda e: (self.pos[e[0]], self.pos[e[1]])
        for r in range(1, r_max + 1):
            for removed in itertools.combinations(E, r):
                rest = [e for e in E if e not in removed]
                reach = self.closure_of([pos(e) for e in rest])
                if reach is None:
                    continue
                used_in = 0
                used_out = 0
                for a, b in rest:
                    used_out |= 1 << self.pos[a]
                    used_in |= 1 << self.pos[b]

                def add(k, reach, used_in, used_out, added, start_a):
                    if k == 0:
                        cand = frozenset(rest) | frozenset(added)
                        if cand != edges:
                            found.add(cand)
                        return
                    for a in range(start_a, n):
                        if (used_out >> a) & 1:
                            continue
                        m = am[a] & ~used_in
                        while m:
                            low = m & -m
                            b = low.bit_length() - 1
                            m ^= low
                            nr = self._add_edge(reach, a, b)
                            if nr is None:
                                continue
                            add(k - 1, nr, used_in | low, used_out | (1 << a),
                                added + [(self.active[a], self.active[b])], a + 1)
                add(r, reach, used_in, used_out, [], 0)
                if len(found) > cap:
                    return found
        return found


def exchange_distance(e1, e2):
    return len(e1 - e2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['enumerate'])
    ap.add_argument('--plans', required=True, help='CSV with circuit, plan_id, reuse_chains (all circuits)')
    ap.add_argument('--chains-col', default='reuse_chains')
    ap.add_argument('--qasm-dir', default='benchmarks_compact')
    ap.add_argument('--c1-dir', default='exp4c1_all', help='existing catalog (for the sanity check; optional)')
    ap.add_argument('--out-dir', default='space_out')
    ap.add_argument('--max-plans', type=int, default=20000)
    ap.add_argument('--sample', type=int, default=3000, help='random sample size when the space exceeds --max-plans')
    ap.add_argument('--local-r', type=int, default=2, help='exchange radius added around each source when sampling')
    ap.add_argument('--circuits', default='')
    ap.add_argument('--verify', type=int, default=300, help='plans per circuit re-checked with the independent oracle')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    import exp4c1_build_table as b
    from reuse_adversarial import reference_static_feasible
    os.makedirs(a.out_dir, exist_ok=True)
    plans = pd.read_csv(a.plans)
    if a.circuits:
        plans = plans[plans.circuit.isin(a.circuits.split(','))]
    known = {}
    cp = os.path.join(a.c1_dir, 'catalog.csv')
    if os.path.exists(cp):
        for r in pd.read_csv(cp).itertuples():
            known.setdefault(r.circuit, {})[r.sig] = b.parse_chains(r.plan_json)

    cat_rows, reach_rows, rep_rows = {}, [], []
    t00 = time.time()
    for circuit, g in plans.groupby('circuit', sort=False):
        t0 = time.time()
        qc = b.load_qc(os.path.join(a.qasm_dir, circuit))
        ps = PlanSpace(qc)
        srcs = []
        for r in g.itertuples():
            ch = canonical(b.parse_chains(getattr(r, a.chains_col)))
            srcs.append((str(r.plan_id), ch, edges_of(ch)))
        Ks = {sum(len(c) - 1 for c in ch) for _, ch, _ in srcs}
        if len(Ks) != 1:
            print(f'[warn] {circuit}: source plans have different widths {Ks}; using the most common')
        K = max(Ks, key=lambda k: sum(1 for _, ch, _ in srcs if sum(len(c) - 1 for c in ch) == k))
        srcs = [s for s in srcs if sum(len(c) - 1 for c in s[1]) == K]
        counts = ps.matching_counts()
        k_ub = max(k for k, c in enumerate(counts) if c)
        space, complete = ps.enumerate_plans(K, max_plans=a.max_plans)
        mode = 'exact'
        if not complete:
            mode = 'sample'
            space = list(ps.sample_plans(K, a.sample, seed=a.seed))
            for _, _, e in srcs:
                space += list(ps.exchange_neighbours(e, r_max=a.local_r))
        space = list({frozenset(e) for e in space} | {e for _, _, e in srcs})
        # --- correctness: checker + independent oracle
        rng = np.random.default_rng(a.seed)
        width = ps.n - K
        bad_ck = 0
        for e in space:
            if not ps.ck.validate(chains_from_edges(ps.active, e), target_width=width).is_valid:
                bad_ck += 1
        sub = [space[i] for i in rng.permutation(len(space))[:a.verify]]
        bad_ref = sum(0 if reference_static_feasible(qc, [list(c) for c in chains_from_edges(ps.active, e)], width) else 1
                      for e in sub)
        # --- the known catalog must be inside the exact space
        kn = known.get(circuit, {})
        kn_edges = {s: edges_of(p) for s, p in kn.items()
                    if len(p) == width and set(q for c in p for q in c) == set(ps.active)}
        space_set = set(space)
        missing_known = sum(1 for e in kn_edges.values() if e not in space_set) if mode == 'exact' else None
        # --- exchange distance of known 1-step neighbours from their sources
        rs = []
        for _, _, e in srcs:
            for s, ek in kn_edges.items():
                d = exchange_distance(e, ek)
                if 0 < d <= 3:
                    rs.append(d)
        # --- write
        sigs = {}
        for e in space:
            ch = chains_from_edges(ps.active, e)
            sigs[e] = (sig_of(ch), ch)
        orig_sigs = {sig_of(ch) for _, ch, _ in srcs}
        for e, (s, ch) in sigs.items():
            cat_rows[(circuit, s)] = dict(circuit=circuit, sig=s, plan_json=json.dumps([list(c) for c in ch]),
                                          is_original=s in orig_sigs)
        for pid, ch, es in srcs:
            for e, (s, _) in sigs.items():
                reach_rows.append(dict(circuit=circuit, source_plan_id=pid, source_sig=sig_of(ch), sig=s,
                                       dist=exchange_distance(es, e)))
        dist_hist = pd.Series([exchange_distance(es, e) for _, _, es in srcs for e in space]).value_counts().sort_index().to_dict()
        rep_rows.append(dict(circuit=circuit, active=ps.n, width=width, K=K, K_max_matching=k_ub,
                             allowed_pairs=int(sum(map(sum, ps.allowed))), n_sources=len(srcs),
                             matchings_of_size_K_upper_bound=counts[K], valid_plans=len(space), mode=mode,
                             checker_rejects=bad_ck, oracle_disagreements=bad_ref, oracle_checked=len(sub),
                             known_catalog_plans=len(kn_edges), known_missing_from_space=missing_known,
                             r_hist_known_neighbours=json.dumps({int(k): int(v) for k, v in pd.Series(rs).value_counts().sort_index().items()}),
                             dist_hist_space=json.dumps({int(k): int(v) for k, v in dist_hist.items()}),
                             seconds=round(time.time() - t0, 1)))
        r = rep_rows[-1]
        print(f"[{circuit}] n={r['active']} w={r['width']} K={K} (max matching {k_ub}) allowed={r['allowed_pairs']} "
              f"UB={r['matchings_of_size_K_upper_bound']} valid={r['valid_plans']} ({mode}) "
              f"ck_rej={bad_ck} oracle_dis={bad_ref}/{len(sub)} missing_known={missing_known} {r['seconds']}s", flush=True)

    pd.DataFrame(list(cat_rows.values())).to_csv(os.path.join(a.out_dir, 'catalog5.csv'), index=False)
    pd.DataFrame(reach_rows).to_csv(os.path.join(a.out_dir, 'reach.csv'), index=False)
    rep = pd.DataFrame(rep_rows)
    rep.to_csv(os.path.join(a.out_dir, 'space_report.csv'), index=False)
    ok = (rep.checker_rejects.sum() == 0) and (rep.oracle_disagreements.sum() == 0) and \
         (rep.known_missing_from_space.fillna(0).sum() == 0)
    print(f'\n{len(rep)} circuits in {time.time() - t00:.0f}s; SELF-CHECKS {"PASSED" if ok else "FAILED - inspect space_report.csv"}')


if __name__ == '__main__':
    main()
