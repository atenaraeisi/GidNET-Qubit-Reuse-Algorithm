#!/usr/bin/env python3
"""Topology-guided fixed-width REPAIR v1 (SWAP objective, NO model training).

- Existing, unmodified suffix_swap and chain_transfer mutation operators.
- Static quotient interaction graph, multi-start greedy placement against an
  input physical CouplingMap, small bounded placement-local-search improvement.
- Same score for final choice in topology_beam / random_beam / structural_beam.
- No routing whatsoever when searching or deciding whether to repair.
- Static validity checked by existing independent ReuseValidityChecker.
- Physical Qiskit/SABRE routing only in explicit 'route' stage, after search.
- All decisions and budgets logged. Outputs are not claims of generalization.

Run in user's original Qiskit 0.45.3 project with included code renamed to:
 exp4d_features.py, reuse_adversarial.py, reuse_validity.py, router_topo.py.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
import time
from collections import Counter,defaultdict
from pathlib import Path

from topology_embedding_v1 import TopologyScorer,graph_for_topology,canonical

SCHEMA='topology_guided_repair_v1_2026_11_06'
POLICIES=('topology_beam','random_beam','structural_beam','structural_final')
OPS=('suffix_swap','chain_transfer')


def stable_seed(*args):return int.from_bytes(hashlib.sha256('|'.join(map(str,args)).encode()).digest()[:8],'big')


def plan_sig(plan):
    val=json.dumps([list(c) for c in canonical(plan)])
    return hashlib.sha1(val.encode()).hexdigest()[:24]


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def write_csv(path,items,keys):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');wr.writeheader();wr.writerows(items)


def save_json(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def struct_score(plan,pairs):
    owner={q:i for i,c in enumerate(plan) for q in c}
    weights=Counter()
    for a,b in pairs:
        x,y=owner[a],owner[b]
        if x==y:raise ValueError('Invalid inter-chain 2q interaction')
        weights[tuple(sorted((x,y)))]+=1
    degrees=[set() for _ in plan]
    for a,b in weights:degrees[a].add(b);degrees[b].add(a)
    n=sum(weights.values())
    hhi=sum(x*x for x in weights.values())/(n*n) if n else 0
    return (len(weights),-round(hhi,12),max((len(x) for x in degrees),default=0))


def sampled_mutations(plan,op,seed,max_checked,checker, *, verify_reference=None):
    """Uniformly sample raw mutation parameters *without enumerating* all moves.

    This matches the two existing mutation definitions exactly, but directly
    samples their parameter space. O(number of sampled tuples + chain-pair
    setup), instead of materializing all potentially huge raw neighbors.
    Raw configurations that create identical / no-op canonical plans are
    deduplicated and do not count against the validation-call budget.
    """
    import bisect
    if op not in OPS:raise ValueError(op)
    if max_checked<1:raise ValueError('max_checked must be positive')
    plan=tuple(tuple(c) for c in plan)
    chunks=[];ends=[];total=0
    if op=='suffix_swap':
        for i in range(len(plan)):
            for j in range(i+1,len(plan)):
                count=len(plan[i])*len(plan[j])
                if count:
                    total+=count;chunks.append((i,j));ends.append(total)
    else:
        for i,src in enumerate(plan):
            if len(src)<=1:continue
            for j,dst in enumerate(plan):
                if j==i:continue
                count=len(src)*(len(dst)+1)
                total+=count;chunks.append((i,j));ends.append(total)
    if total==0:
        return [],dict(raw_possible=0,raw_sampled=0,validation_calls=0,valid_from_sample=0)
    rng=random.Random(seed)
    # Draw at most 4x the requested number to replace repeated / identity moves.
    raw_slots=rng.sample(range(total),min(total,4*max_checked))
    original=canonical(plan);unique=set();selected=[]
    for global_idx in raw_slots:
        chunk_idx=bisect.bisect_right(ends,global_idx)
        offset=global_idx-(ends[chunk_idx-1] if chunk_idx else 0)
        i,j=chunks[chunk_idx]
        if op=='suffix_swap':
            a=offset//len(plan[j])+1
            b=offset%len(plan[j])+1
            proposal=list(plan)
            proposal[i]=plan[i][:a]+plan[j][b:]
            proposal[j]=plan[j][:b]+plan[i][a:]
        else:
            pos=offset//(len(plan[j])+1)
            insertion=offset%(len(plan[j])+1)
            q=plan[i][pos]
            proposal=list(plan)
            proposal[i]=plan[i][:pos]+plan[i][pos+1:]
            proposal[j]=plan[j][:insertion]+(q,)+plan[j][insertion:]
        if not proposal or any(not c for c in proposal):continue
        cand=canonical(proposal)
        if cand==original or cand in unique:continue
        unique.add(cand)
        selected.append(cand)
        if len(selected)>=max_checked:break
    accepted=[]
    for candidate in selected:
        if checker.validate(candidate,target_width=len(plan)).is_valid:
            if verify_reference is not None and not verify_reference(candidate):
                raise AssertionError('Independent reference disagrees with checker')
            accepted.append((plan_sig(candidate),candidate))
    return accepted,dict(raw_possible=total,raw_sampled=len(raw_slots),
                         validation_calls=len(selected),valid_from_sample=len(accepted))


def search_core(start,pairs,scorer,checker,*,policy,seed,beam_width=5,max_depth=5,
                budget=120,per_op_sample=24,verify_reference=None):
    """Deterministic reproducible beam comparison under capped score/validation counts.

    All three guided/random policies use TOPOLOGY score for final selection.
    structural_final uses the old structural score for expansion+final selection.
    No policy sees routed SWAP labels.
    """
    if policy not in POLICIES:raise ValueError(policy)
    if min(beam_width,max_depth,budget,per_op_sample)<1:raise ValueError('Positive search budgets required')
    original=canonical(start);s0=plan_sig(original)
    base_score=scorer.relative_score(original)
    base_struct=struct_score(original,pairs)
    seen={s0:(original,0,base_score,base_struct)}
    # Counter does not materialize missing keys when converted with **counters.
    # An original with zero NEW valid plans has no topology scoring after its
    # initial score, so score_seconds must be emitted explicitly as 0.0.
    counters=Counter({
        'expanded_parents': 0, 'mutation_seconds': 0.0,
        'raw_possible': 0, 'raw_sampled': 0,
        'validation_calls': 0, 'valid_found': 0,
        'score_seconds': 0.0,
    })
    frontier=[s0];trace=[];stop='depth_limit';rng=random.Random(stable_seed(seed,policy))
    for depth in range(1,max_depth+1):
        layer=[]
        for parent_sig in frontier:
            parent=seen[parent_sig][0]
            counters['expanded_parents']+=1
            for op in OPS:
                t=time.perf_counter()
                found,stats=sampled_mutations(parent,op,
                     stable_seed(seed,'proposal',parent_sig,op),per_op_sample,checker,
                     verify_reference=verify_reference)
                counters['mutation_seconds']+=time.perf_counter()-t
                counters['raw_possible']+=stats['raw_possible']
                counters['raw_sampled']+=stats['raw_sampled']
                counters['validation_calls']+=stats['validation_calls']
                counters['valid_found']+=stats['valid_from_sample']
                for sig,plan in found:
                    if sig in seen:continue
                    if len(seen)-1>=budget:
                        stop='valid_score_budget';break
                    t=time.perf_counter()
                    score=scorer.relative_score(plan)
                    ss=struct_score(plan,pairs)
                    counters['score_seconds']+=time.perf_counter()-t
                    seen[sig]=(plan,depth,score,ss)
                    layer.append(sig)
                    trace.append(dict(sig=sig,parent_sig=parent_sig,operator=op,depth=depth,
                        topology_score=score,struct_score=json.dumps(ss)))
                if len(seen)-1>=budget:break
            if len(seen)-1>=budget:break
        if not layer:
            stop='no_new_valid_plans_in_sample';break
        if policy=='topology_beam':frontier=sorted(layer,key=lambda s:(seen[s][2],s))[:beam_width]
        elif policy in ('structural_beam','structural_final'):
            frontier=sorted(layer,key=lambda s:(seen[s][3],s))[:beam_width]
        else:
            rng.shuffle(layer);frontier=layer[:beam_width]
        if len(seen)-1>=budget:stop='valid_score_budget';break
    key_index=3 if policy=='structural_final' else 2
    selected_sig=min(seen,key=lambda s:(seen[s][key_index],s))
    if not(seen[selected_sig][key_index]<seen[s0][key_index]):selected_sig=s0
    sel=seen[selected_sig]
    return dict(original_sig=s0,pick_sig=selected_sig,pick_plan=sel[0],
        path_length=sel[1],selected_topology_score=sel[2],original_topology_score=base_score,
        selected_structural_score=sel[3],original_structural_score=base_struct,
        seen=seen,trace=trace,n_valid_scored=len(seen)-1,stop_reason=stop,
        **counters)


def qiskit_circuit(path):
    import qiskit
    from qiskit import QuantumCircuit
    from exp4d_features import prepare_gidnet_input
    if qiskit.__version__!='0.45.3':
        raise RuntimeError(f'For comparison parity use Qiskit 0.45.3, got {qiskit.__version__}')
    qc,old_to_new=prepare_gidnet_input(QuantumCircuit.from_qasm_file(str(path)))
    pairs=[]
    for item in qc.data:
        qubits=[qc.find_bit(q).index for q in item.qubits]
        if len(qubits)==2:pairs.append(tuple(qubits))
    return qc,pairs


def source_rows(catalog,chosen):
    res=[]
    for r in read_csv(catalog):
        if r.get('is_original','').lower() not in ('true','1','yes'):continue
        if chosen and r['circuit'] not in chosen:continue
        plan=canonical(json.loads(r['plan_json']))
        if plan_sig(plan)!=r['sig']:raise AssertionError(f'Catalog plan signature mismatch {r["circuit"]}')
        res.append(dict(circuit=r['circuit'],original_sig=r['sig'],plan=plan))
    res.sort(key=lambda x:(x['circuit'],x['original_sig']))
    if not res:raise ValueError('No original plans in selected catalog')
    return res


def stage_search(args):
    from reuse_validity import ReuseValidityChecker
    out=Path(args.out_dir)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Choose a NEW --out-dir; do not overwrite results')
    selected={x.strip() for x in args.circuits.split(',') if x.strip()}
    originals=source_rows(args.catalog,selected)
    if args.max_originals: originals=originals[:args.max_originals]
    out.mkdir(parents=True,exist_ok=True)
    all_decisions=[];all_plans={};trace=[];prepared={}
    topo=graph_for_topology(args.topology)
    for idx,row in enumerate(originals,1):
        circuit=row['circuit'];start=row['plan']
        if circuit not in prepared:
            qc,pairs=qiskit_circuit(Path(args.qasm_dir)/circuit)
            ck=ReuseValidityChecker(qc)
            prepared[circuit]=(qc,pairs,ck,TopologyScorer(pairs,topo,starts=args.starts,local_steps=args.local_steps))
        qc,pairs,ck,scorer=prepared[circuit]
        if plan_sig(start)!=row['original_sig']:raise AssertionError('Original hash mismatch')
        if not ck.validate(start,target_width=len(start)).is_valid:raise ValueError('Invalid Original')
        if len(start)>topo.number_of_nodes():raise ValueError('Hardware too small for plan')
        verify=None
        if args.verify_reference:
            from reuse_adversarial import reference_static_feasible
            verify=lambda p:reference_static_feasible(qc,p,len(start))
        basekey=(circuit,row['original_sig'])
        for rep in range(args.repetitions):
            for policy in POLICIES:
                seed=stable_seed(args.seed,circuit,row['original_sig'],rep)
                result=search_core(start,pairs,scorer,ck,policy=policy,seed=seed,
                     beam_width=args.beam_width,max_depth=args.max_depth,
                     budget=args.budget,per_op_sample=args.per_op_sample,verify_reference=verify)
                p=result['pick_plan']
                if not ck.validate(p,target_width=len(start)).is_valid:raise AssertionError('Selected invalid plan')
                if verify is not None and not verify(p):raise AssertionError('Selected independent feasibility failed')
                for sig,pl in ((row['original_sig'],start),(result['pick_sig'],p)):
                    all_plans[(circuit,sig)]=dict(circuit=circuit,sig=sig,plan_json=json.dumps([list(c) for c in pl]))
                all_decisions.append(dict(circuit=circuit,original_sig=row['original_sig'],
                    pick_sig=result['pick_sig'],repetition=rep,policy=policy,
                    topology=args.topology,path_length=result['path_length'],
                    original_topology_score=result['original_topology_score'],
                    selected_topology_score=result['selected_topology_score'],
                    original_structural_score=json.dumps(result['original_structural_score']),
                    selected_structural_score=json.dumps(result['selected_structural_score']),
                    n_valid_scored=result['n_valid_scored'],raw_possible=result['raw_possible'],raw_sampled=result['raw_sampled'],
                    validation_calls=result['validation_calls'],valid_found=result['valid_found'],
                    expanded_parents=result['expanded_parents'],
                    score_seconds=result.get('score_seconds',0.0),mutation_seconds=result.get('mutation_seconds',0.0),
                    stop_reason=result['stop_reason']))
                for t in result['trace']:
                    trace.append(dict(circuit=circuit,original_sig=row['original_sig'],
                         repetition=rep,policy=policy,**t))
        print(f'[search {idx}/{len(originals)}] {circuit} {row["original_sig"]}',flush=True)
    write_csv(out/'search_decisions.csv',all_decisions,list(all_decisions[0]))
    write_csv(out/'selected_plan_catalog.csv',list(all_plans.values()),['circuit','sig','plan_json'])
    write_csv(out/'candidate_trace.csv',trace,
              ['circuit','original_sig','repetition','policy','sig','parent_sig','operator','depth','topology_score','struct_score'])
    save_json(out/'search_manifest.json',dict(schema=SCHEMA,topology=args.topology,
       catalog=str(Path(args.catalog).resolve()),qasm_dir=str(Path(args.qasm_dir).resolve()),
       scorer=dict(starts=args.starts,local_steps=args.local_steps),
       budgets=dict(beam_width=args.beam_width,max_depth=args.max_depth,budget=args.budget,per_op_sample=args.per_op_sample),
       n_originals=len(originals),n_decisions=len(all_decisions),n_unique_selected_plans=len(all_plans),
       note='SEARCH ONLY. NO physical routing or physical labels used. Prior catalog originals are exploratory; do not claim fresh heldout.'))
    pairs=defaultdict(dict)
    for d in all_decisions:pairs[d['circuit'],d['original_sig'],d['repetition']][d['policy']]=d['pick_sig']
    save_json(out/'search_diagnostic.json',dict(n_originals=len(originals),n_paired_repetitions=len(pairs),
       same_topology_and_random=sum(x['topology_beam']==x['random_beam'] for x in pairs.values()),
       same_topology_and_structural=sum(x['topology_beam']==x['structural_beam'] for x in pairs.values()),
       n_routing_jobs_if_fresh_5seeds=len(all_plans)*5))
    print(f'[PASS] SEARCH ONLY: {len(all_decisions)} decisions; {len(all_plans)} unique Original/final plans. NO ROUTING.')


ROUTE_FIELDS=['circuit','sig','seed','swap_count','routed_depth','routing_seconds','status','error']


def stage_route(args):
    """Explicitly routes ONLY Original and selected final plans on new seeds."""
    import qiskit
    from qiskit import QuantumCircuit
    from experiment3_pipeline_adapter import build_dynamic_circuit,normalize_for_routing
    from reuse_validity import ReuseValidityChecker
    from reuse_adversarial import reference_static_feasible
    from router_topo import _route
    if qiskit.__version__!='0.45.3':raise RuntimeError('Qiskit 0.45.3 required')
    out=Path(args.out_dir)
    metadata=json.loads((out/'search_manifest.json').read_text(encoding='utf-8'))
    if metadata['schema']!=SCHEMA:raise RuntimeError('Wrong search manifest')
    seeds=tuple(int(x) for x in args.seeds.split(','))
    if len(set(seeds))!=len(seeds) or any(x<400 for x in seeds):
        raise ValueError('Use new seeds >=400, never reuse training or prior diagnostic seeds')
    topo=metadata['topology']
    lock=dict(topology=topo,seeds=seeds,router='router_topo._route; sabre layout+routing, opt1',
              qiskit=qiskit.__version__,adapter='experiment3_pipeline_adapter')
    lockp=out/'routing_lock.json'
    if lockp.exists() and json.loads(lockp.read_text(encoding='utf8'))!=lock:raise RuntimeError('Routing lock mismatch')
    if not lockp.exists():save_json(lockp,lock)
    plans=read_csv(out/'selected_plan_catalog.csv')
    old=read_csv(out/'routed_results.csv') if (out/'routed_results.csv').exists() else []
    existing={(r['circuit'],r['sig'],int(r['seed'])):r for r in old}
    if len(existing)!=len(old):raise RuntimeError('Duplicate routed rows')
    jobs={(r['circuit'],r['sig'],seed) for r in plans for seed in seeds}
    if not set(existing)<=jobs:raise RuntimeError('Unexpected old routing jobs')
    by_key={(r['circuit'],r['sig']):r for r in plans}
    cached={};compiled={};count=0
    for circ,sig,seed in sorted(jobs-set(existing)):
        if args.max_new_jobs and count>=args.max_new_jobs:break
        if circ not in cached:
            qc,_=qiskit_circuit(Path(metadata['qasm_dir'])/circ)
            cached[circ]=(qc,ReuseValidityChecker(qc))
        qc,ck=cached[circ]
        if (circ,sig) not in compiled:
            plan=canonical(json.loads(by_key[circ,sig]['plan_json']))
            if sig!=plan_sig(plan):raise AssertionError('Corrupt plan signature')
            if not ck.validate(plan,target_width=len(plan)).is_valid or not reference_static_feasible(qc,plan,len(plan)):
                raise AssertionError('Invalid selected Plan')
            dyn=build_dynamic_circuit(qc,plan)
            norm=normalize_for_routing(dyn)
            if dyn.num_qubits!=len(plan) or norm.num_qubits!=len(plan):raise AssertionError('Unexpected dynamic circuit width')
            compiled[circ,sig]=norm
        t=time.perf_counter()
        try:
            value=_route(compiled[circ,sig],seed,topo)
            record=dict(circuit=circ,sig=sig,seed=seed,swap_count=value['swap_count'],
                       routed_depth=value['routed_depth'],routing_seconds=time.perf_counter()-t,status='ok',error='')
        except Exception as exc:
            record=dict(circuit=circ,sig=sig,seed=seed,swap_count='',routed_depth='',
                       routing_seconds=time.perf_counter()-t,status='error',error=f'{type(exc).__name__}: {exc}')
        isnew=not (out/'routed_results.csv').exists()
        with (out/'routed_results.csv').open('a',encoding='utf-8-sig' if isnew else 'utf-8',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=ROUTE_FIELDS)
            if isnew:wr.writeheader()
            wr.writerow(record)
        count+=1
        if count%20==0:print(f'[route] {count} new jobs',flush=True)
    print(f'[route] {count} new jobs, total {len(jobs)} expected; rerun to resume')


def stage_analyze(args):
    out=Path(args.out_dir)
    meta=json.loads((out/'search_manifest.json').read_text(encoding='utf8'))
    lock=json.loads((out/'routing_lock.json').read_text(encoding='utf8'))
    plans=read_csv(out/'selected_plan_catalog.csv');raw=read_csv(out/'routed_results.csv')
    jobs={(p['circuit'],p['sig'],int(s)) for p in plans for s in lock['seeds']}
    result={(r['circuit'],r['sig'],int(r['seed'])):r for r in raw}
    if len(result)!=len(raw) or set(result)!=jobs:raise RuntimeError(f'Incomplete/duplicate routing; missing {len(jobs-set(result))}')
    if any(x['status']!='ok' for x in raw):raise RuntimeError('Routing errors need fixing; no silent omission')
    costs={}
    for p in plans:
        vals=[result[p['circuit'],p['sig'],int(s)] for s in lock['seeds']]
        costs[p['circuit'],p['sig']]=(
            sum(float(v['swap_count']) for v in vals)/len(vals),
            sum(float(v['routed_depth']) for v in vals)/len(vals))
    decisions=read_csv(out/'search_decisions.csv')
    evaluated=[]
    for x in decisions:
        a=costs[x['circuit'],x['original_sig']];b=costs[x['circuit'],x['pick_sig']]
        evaluated.append({**x,'orig_swap':a[0],'selected_swap':b[0],
                'swap_gain_abs':a[0]-b[0],
                'swap_gain_pct':(100*(a[0]-b[0])/a[0] if a[0]>0 else ''),
                'routed_depth_change':b[1]-a[1]})
    write_csv(out/'evaluated_decisions.csv',evaluated,list(evaluated[0]))
    summary=[]
    for method in POLICIES:
        rows=[d for d in evaluated if d['policy']==method]
        by_circuit=defaultdict(list)
        for d in rows:by_circuit[d['circuit']].append(d)
        avg=lambda vals:sum(vals)/len(vals) if vals else None
        summary.append(dict(method=method,n_orig_rep=len(rows),n_circuits=len(by_circuit),
            n_repair=sum(d['pick_sig']!=d['original_sig'] for d in rows),
            n_harm=sum(d['swap_gain_abs'] < -1e-9 for d in rows),
            mean_gain_abs=avg([d['swap_gain_abs'] for d in rows]),
            mean_circuit_gain_abs=avg([avg([x['swap_gain_abs'] for x in v]) for v in by_circuit.values()]),
            mean_circuit_gain_pct=avg([avg([x['swap_gain_pct'] for x in v if x['swap_gain_pct']!='']) for v in by_circuit.values() if any(x['swap_gain_pct']!='' for x in v)]),
            mean_valid_scored=avg([int(d['n_valid_scored']) for d in rows]),
            mean_raw_sampled=avg([int(d['raw_sampled']) for d in rows]),
            mean_search_sec=avg([float(d['score_seconds'])+float(d['mutation_seconds']) for d in rows])))
    write_csv(out/'method_comparison.csv',summary,list(summary[0]))
    save_json(out/'analysis_manifest.json',dict(schema=SCHEMA,topology=meta['topology'],
             routing_seeds=lock['seeds'], n_decisions=len(evaluated),status='routed_final_plans_only',
             note='Old benchmark family; exploratory until independent new circuits and family-aware uncertainty evaluated.'))
    for d in summary:print(d)
    print('[PASS] physical evaluation of ONLY final selected and original plans')


def main():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,description=__doc__)
    p.add_argument('stage',choices=('search','route','analyze'))
    p.add_argument('--catalog',default='topo_out/hh7/catalog.csv')
    p.add_argument('--qasm-dir',default='benchmarks_compact')
    p.add_argument('--out-dir',default='topo_guided_v1_out')
    p.add_argument('--topology',choices=('hh7','grid','line'),default='hh7')
    p.add_argument('--circuits',default='')
    p.add_argument('--max-originals',type=int,default=0)
    p.add_argument('--starts',type=int,default=4)
    p.add_argument('--local-steps',type=int,default=1)
    p.add_argument('--beam-width',type=int,default=5)
    p.add_argument('--max-depth',type=int,default=5)
    p.add_argument('--budget',type=int,default=120)
    p.add_argument('--per-op-sample',type=int,default=24)
    p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--seed',type=int,default=20261106)
    p.add_argument('--verify-reference',action='store_true')
    p.add_argument('--seeds',default='400,401,402,403,404')
    p.add_argument('--max-new-jobs',type=int,default=0)
    a=p.parse_args()
    if a.stage=='search':stage_search(a)
    if a.stage=='route':stage_route(a)
    if a.stage=='analyze':stage_analyze(a)

if __name__=='__main__':main()
