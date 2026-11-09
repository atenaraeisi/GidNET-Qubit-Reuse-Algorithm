"""Quick no-Qiskit unit checks for search logic and strict TRAIN-only access."""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from structural_beam_equal_budget_v1 import (
    signature,structural_score,search_core,get_training_sources,stable_seed, POLICIES)


def test_score():
    plan=((0,2),(1,3))
    assert structural_score(plan,[(0,1),(2,3),(0,1)])==(1,-1.0,1)
    assert structural_score(plan,[(0,2),(1,3)])==(0,-0.0,0)
    assert structural_score(((0,),(1,),(2,),(3,)),[(0,1),(2,3),(0,1)])==(2,-round(5/9,12),1)


def test_search():
    # Node signatures are plan hashes. This test ensures a temporarily bad bridge
    # is not rejected, and No Repair is possible with no structural improvement.
    A=((0,),(1,),(2,));B=((0,1),(2,));C=((0,2),(1,));D=((0,),(1,2))
    graph={signature(A):[B,D],signature(B):[C],signature(C):[],signature(D):[]}
    values={signature(A):(3,-.2,2),signature(B):(4,-.2,3),
            signature(C):(1,-.8,1),signature(D):(3,-.2,2)}
    def propose(plan, node, op, seed, remaining_raw):
        if op=="suffix_swap":return [(signature(p),p) for p in graph[node]],2
        return [],0
    score=lambda p:values[signature(p)]
    v=search_core(A,propose,score,policy="structural_beam",seed=35,beam_width=2,max_depth=3,budget=5)
    assert v["best_sig"]==signature(C)
    assert v["n_evaluated"]==3
    assert v["n_expanded_nodes"]>=2
    v2=search_core(A,propose,lambda p:(0,0,0),policy="structural_beam",seed=35,beam_width=2,max_depth=3,budget=5)
    assert v2["best_sig"]==signature(A)
    v3=search_core(A,propose,score,policy="random_beam",seed=35,beam_width=2,max_depth=3,budget=2)
    assert v3["n_evaluated"]==2
    v4=search_core(A,propose,score,policy="random_beam",seed=35,beam_width=2,max_depth=3,budget=2)
    assert v3["best_sig"]==v4["best_sig"]


def write_rows(path, rows):
    with open(path,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def test_train_guard():
    with tempfile.TemporaryDirectory() as d:
        path=Path(d); tr=path/'train';tr.mkdir()
        plan=((0,),(1,));s=signature(plan)
        write_rows(path/'split.csv',[dict(circuit_id='CaQR::demo.qasm',split='train',family_group='demo'),
                                     dict(circuit_id='CaQR::test.qasm',split='final_test',family_group='t')])
        write_rows(tr/'selected_candidates.csv',[dict(circuit_id='CaQR::demo.qasm',source='CaQR',filename='demo.qasm',original_sig=s,sig=s,is_original=1)])
        write_rows(tr/'plan_catalog.csv',[dict(circuit_id='CaQR::demo.qasm',sig=s,plan_json=json.dumps(plan))])
        (tr/'manifest.json').write_text(json.dumps(dict(qasm_paths={'CaQR::demo.qasm':'demo.qasm'})))
        src=get_training_sources(tr,path/'split.csv',set())
        assert len(src)==1 and src[0]['plan']==plan
        # Must refuse non-training rows instead of quietly evaluating them.
        with open(tr/'selected_candidates.csv','a',newline='') as f:
            f.write(f'CaQR::test.qasm,CaQR,test.qasm,{s},{s},1\n')
        try:get_training_sources(tr,path/'split.csv',set())
        except ValueError as e:assert 'Non-train' in str(e)
        else:raise AssertionError('Test data access not rejected')


if __name__=='__main__':
    test_score();print('[PASS] topology-agnostic structural score')
    test_search();print('[PASS] equal-budget beam core, non-monotone bridge and No Repair')
    test_train_guard();print('[PASS] frozen split train-only guard')


def test_search_stage_and_analyze():
    """Integration smoke without Qiskit; route values are FIXTURE ONLY, not science."""
    import sys, types
    from argparse import Namespace
    import structural_beam_equal_budget_v1 as m
    A=((0,1),(2,),(3,4));B=((0,2),(1,),(3,4));C=((0,3),(1,),(2,4))
    class Validity:
        is_valid=True
    class Checker:
        def validate(self,plan,target_width):return Validity()
    class Prepared:
        def __init__(self,plan):self.plan=plan
        def scan(self,op,max_attempts=180,verify_survivors=False):
            if op=='suffix_swap' and self.plan==A:return [B,C],{'attempted':min(max_attempts,2)}
            return [],{'attempted':min(max_attempts,1)}
    class Engine:
        def prepare(self,plan):return Prepared(plan)
    old_acquire=m.acquire_circuit
    mock_qiskit=types.ModuleType('qiskit');mock_qiskit.__version__='0.45.3'
    mock_reuse=types.ModuleType('reuse_adversarial');mock_reuse.canonical=lambda p:tuple(sorted(tuple(x) for x in p))
    old_qiskit=sys.modules.get('qiskit')
    old_reuse=sys.modules.get('reuse_adversarial')
    try:
        sys.modules['qiskit']=mock_qiskit;sys.modules['reuse_adversarial']=mock_reuse
        m.acquire_circuit=lambda p:(None,Checker(),Engine(),[(0,1),(0,2)],lambda *_:True)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);train=p/'train';train.mkdir()
            s=signature(A)
            write_rows(p/'split.csv',[dict(circuit_id='CaQR::demo.qasm',split='train',family_group='demo')])
            write_rows(train/'selected_candidates.csv',[dict(circuit_id='CaQR::demo.qasm',source='CaQR',filename='demo.qasm',original_sig=s,sig=s,is_original=1)])
            write_rows(train/'plan_catalog.csv',[dict(circuit_id='CaQR::demo.qasm',sig=s,plan_json=json.dumps(A))])
            (train/'manifest.json').write_text(json.dumps(dict(qasm_paths={'CaQR::demo.qasm':'demo.qasm'})))
            a=Namespace(out_dir=p/'run',train_dir=train,split_ledger=p/'split.csv',circuits='',max_originals=0,
                  max_circuits=0,seed=2026,repetitions=2,beam_width=2,max_depth=2,budget=10,raw_budget=100,
                  per_op_keep=10,max_attempts_per_op=180,verify_fast_survivors=False)
            m.stage_search(a)
            decisions=m.read_csv(p/'run'/'search_decisions.csv')
            assert len(decisions)==6 and len(m.read_csv(p/'run'/'selected_plan_catalog.csv'))>=1
            assert json.loads((p/'run'/'search_diagnostic_summary.json').read_text())['n_paired_runs']==2
            # Fixture-only routing rows for all selected plans on two independent seeds.
            j=m.jwrite
            j(p/'run'/'routing_protocol_lock.json', dict(seed_set=[300,301]))
            for r in m.read_csv(p/'run'/'selected_plan_catalog.csv'):
                for seed in (300,301):
                    # Simulated physical costs ONLY exercise file logic.
                    val=10.0 if r['sig']==s else 8.0
                    m.append_csv(p/'run'/'new_seed_routing_results.csv',dict(circuit_id=r['circuit_id'],
                           source=r['source'],filename=r['filename'],sig=r['sig'],seed=seed,
                           swap_count=val,routed_depth=3*val,routing_time_sec=.01,status='ok',error=''),m.ROUTE_FIELDS)
            m.stage_analyze(Namespace(out_dir=p/'run'))
            assert (p/'run'/'method_comparison.csv').is_file()
            assert json.loads((p/'run'/'analysis_summary.json').read_text())['n_paired_runs']==2
    finally:
        m.acquire_circuit=old_acquire
        if old_qiskit is None:sys.modules.pop('qiskit',None)
        else:sys.modules['qiskit']=old_qiskit
        if old_reuse is None:sys.modules.pop('reuse_adversarial',None)
        else:sys.modules['reuse_adversarial']=old_reuse


if __name__=='__main__':
    test_search_stage_and_analyze()
    print('[PASS] search + result files + post-route analysis integration (synthetic fixture)')
