"""Qiskit-free synthetic dependency-DAG regression tests (run with pytest)."""
from __future__ import annotations
import itertools
import random
import sys
from pathlib import Path
import networkx as nx
sys.path.insert(0, str(Path(__file__).parent))
from reuse_validity import ReuseValidityChecker
from reuse_adversarial import OPERATORS, canonical
from plan_space import PlanSpace, chains_from_edges
from validity_aware_neighbors import SUPPORTED_OPERATORS, ValidityAwareEngine, iter_mutation_deltas


def make_checker(n, seed):
    rng=random.Random(seed)
    g=nx.DiGraph()
    ins={i:('in',i) for i in range(n)}
    outs={i:('out',i) for i in range(n)}
    last={i:ins[i] for i in range(n)}
    for i in range(n): g.add_node(ins[i]);g.add_node(outs[i])
    for t in range(rng.randrange(0,13)):
        wires=rng.sample(range(n),rng.choice([1,2] if n>=2 else [1]))
        gate=('gate',t)
        for q in wires: g.add_edge(last[q],gate);last[q]=gate
    for q in range(n): g.add_edge(last[q],outs[q])
    return ReuseValidityChecker.from_graphs(num_qubits=n,active_qubits=range(n),
        full_graph=g,full_inputs=ins,full_outputs=outs)


def test_exhaustive_operator_delta_and_checker_parity():
    nplans=ntests=0
    for n in (3,4,5,6):
        for seed in range(9):
            ck=make_checker(n,100*n+seed)
            eng=ValidityAwareEngine(ck)
            ps=eng.ps
            rng=random.Random(seed)
            all_edges=set()
            for width in range(2,n):
                edges,complete=ps.enumerate_plans(n-width, max_plans=20000)
                assert complete
                all_edges.update(edges)
            assert all(ck.validate(chains_from_edges(ps.active, e), target_width=n-len(e)).is_valid for e in all_edges)
            plans=[chains_from_edges(ps.active,e) for e in sorted(all_edges,key=lambda e:sorted(e))]
            rng.shuffle(plans)
            for plan in plans[:25]:
                prep=eng.prepare(plan)
                nplans+=1
                prev_seen=set()
                for op in SUPPORTED_OPERATORS:
                    old=[]
                    for raw in OPERATORS[op](plan):
                        if not raw or not all(raw):continue
                        raw=canonical(raw)
                        if raw == plan or raw in prev_seen: continue
                        prev_seen.add(raw)
                        if ck.validate(raw,target_width=len(plan)).is_valid: old.append(raw)
                    new,stats=prep.scan(op,verify_survivors=True)
                    assert old==new, (n,seed,op,plan,old,new,stats)
                    ntests+=1
            # Check plan-space cycle closure agrees with checker for ALL enumerated plans.
            # Validate candidate matching patterns even when they are invalid.
            if n<=5:
                for perm in itertools.permutations(range(n)):
                    for split in range(1,n):
                        plan=canonical((perm[:split],perm[split:]))
                        if len(plan)!=2:continue
                        valid=ck.validate(plan,target_width=2).is_valid
                        edges={(a,b) for chain in plan for a,b in zip(chain,chain[1:])}
                        legal=all(ps.allowed[ps.pos[a]][ps.pos[b]] for a,b in edges)
                        fast=legal and ps.is_acyclic(edges)
                        assert fast==valid, (n,seed,plan,valid,fast)
    print('synthetic parity: source plans',nplans,'operator comparisons',ntests)


def test_pilot_sampling_and_bfs_exact_deterministic_parity():
    from build_multistep_pilot_validity_aware import legal_neighbors, discover
    for n in (4,5,6):
        for seed in range(4):
            ck=make_checker(n,900+n*10+seed)
            eng=ValidityAwareEngine(ck)
            ps=eng.ps
            candidates=[]
            for width in range(2,n):
                plans,_=ps.enumerate_plans(n-width,max_plans=20000)
                candidates.extend(chains_from_edges(ps.active,e) for e in plans)
            if not candidates:continue
            plan=candidates[0]
            for cap in (1,4,8,180):
                l,st_l=legal_neighbors(plan,ck,random.Random(42),
                     max_attempts_per_op=cap,per_op_keep=4)
                f,st_f=legal_neighbors(plan,ck,random.Random(42),
                     max_attempts_per_op=cap,per_op_keep=4,engine=eng,verify_fast_survivors=True)
                assert l==f,(n,seed,cap,l,f)
                assert st_l['attempted']==st_f['attempted']
                assert st_l['valid']==st_f['valid']
            old,co=discover(plan,ck,seed=707,max_depth=3,max_frontier=8,
                max_attempts_per_op=180,per_op_keep=4,max_discovered=100)
            new,cn=discover(plan,ck,seed=707,max_depth=3,max_frontier=8,
                max_attempts_per_op=180,per_op_keep=4,max_discovered=100,
                engine=eng,verify_fast_survivors=True)
            assert old==new
            for key in ('attempted','valid','found_at_1','found_at_2','found_at_3'):
                assert co[key]==cn[key]

if __name__ == '__main__':
    test_exhaustive_operator_delta_and_checker_parity()
    test_pilot_sampling_and_bfs_exact_deterministic_parity()
    print('PASS: synthetic checker/operator parity and deterministic pilot discovery')
