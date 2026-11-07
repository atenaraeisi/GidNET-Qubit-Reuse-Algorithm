#!/usr/bin/env python3
"""Qiskit-free deterministic tests; real Qiskit integration requires the project env."""
import json
import tempfile
from pathlib import Path
import networkx as nx
import numpy as np

from topology_embedding_v1 import TopologyScorer,canonical,graph_for_topology,read_twoq_qasm2_compact
from topology_guided_repair_v1 import (plan_sig,sampled_mutations,search_core,struct_score)


class DummyValidation:
    def __init__(self,yes):self.is_valid=yes


class FakeChecker:
    def __init__(self,allqs,pairs):self.allqs=set(allqs);self.pairs=pairs
    def validate(self,plan,target_width=None):
        ok=len(plan)==target_width and len({q for c in plan for q in c})==len(self.allqs)
        ok=ok and {q for c in plan for q in c}==self.allqs
        owner={q:i for i,c in enumerate(plan) for q in c}
        ok=ok and all(owner[a]!=owner[b] for a,b in self.pairs)
        return DummyValidation(ok)


def test_basic_score():
    pairs=[(0,1),(0,2),(0,3)]
    plan=((0,),(1,),(2,),(3,))
    a=TopologyScorer(pairs,nx.star_graph(3),starts=4,local_steps=2)
    b=TopologyScorer(pairs,nx.path_graph(4),starts=4,local_steps=2)
    assert a.score(plan).score==0.0
    assert b.score(plan).score>=1.0
    assert a.score(reversed(plan)).score==0.0
    assert len(a.score(plan).placement)==len(set(a.score(plan).placement))
    print('[PASS] topology-dependent score, canonical invariance, injective placement')


def test_qasm_compaction():
    qasm='''OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
h q[4];
cx q[4],q[2];
measure q[2] -> c[0];'''
    with tempfile.NamedTemporaryFile(mode='w',suffix='.qasm',delete=False) as f:
        f.write(qasm);path=f.name
    try:
        pairs,n=read_twoq_qasm2_compact(path)
        assert n==2 and pairs==[(1,0)]
    finally:Path(path).unlink()
    print('[PASS] QASM2 qreg index compaction')


def test_search():
    pairs=[(0,2),(0,3),(1,2)]
    original=canonical(((0,1),(2,3)))
    ck=FakeChecker(range(4),pairs)
    assert ck.validate(original,target_width=2).is_valid
    scorer=TopologyScorer(pairs,nx.path_graph(5),starts=4,local_steps=1)
    sig=plan_sig(original)
    ans={}
    for policy in ('topology_beam','random_beam','structural_beam','structural_final'):
        a=search_core(original,pairs,scorer,ck,policy=policy,seed=42,budget=12,
                      per_op_sample=30,beam_width=3,max_depth=4)
        b=search_core(original,pairs,scorer,ck,policy=policy,seed=42,budget=12,
                      per_op_sample=30,beam_width=3,max_depth=4)
        assert a['pick_sig']==b['pick_sig'] and a['n_valid_scored']==b['n_valid_scored']
        assert ck.validate(a['pick_plan'],target_width=2).is_valid
        assert a['n_valid_scored']<=12
        assert a['selected_topology_score']<=a['original_topology_score']+1e-9 or policy=='structural_final'
        ans[policy]=a
    print('[PASS] fixed-width valid search, seeded determinism, budget, no-repair guard, controls')


def test_reservoir_proposal():
    original=canonical(((0,1),(2,3)))
    pairs=[(0,2),(0,3),(1,2)]
    ck=FakeChecker(range(4),pairs)
    p,st=sampled_mutations(original,'suffix_swap',777,3,ck)
    assert st['validation_calls']<=3 and st['raw_possible']>=st['validation_calls']
    assert all(ck.validate(x,target_width=2).is_valid for _,x in p)
    # With a sample budget beyond the raw move count, reproduce the existing generators exactly.
    from reuse_adversarial import unique_mutations
    for operator in ('suffix_swap','chain_transfer'):
        sample,stats=sampled_mutations(original,operator,777,1000,ck)
        legacy={canonical(p) for p in unique_mutations(original,operator) if ck.validate(p,target_width=2).is_valid}
        obtained={canonical(p) for _,p in sample}
        assert obtained==legacy, (operator,obtained,legacy)
    print('[PASS] parameter-sampled operator exactly matches the existing operator neighborhood')


if __name__=='__main__':
    test_basic_score();test_qasm_compaction();test_search();test_reservoir_proposal()
    print('[PASS] ALL PURE TESTS. Real Qiskit/Compiler parity NOT tested in this environment.')
