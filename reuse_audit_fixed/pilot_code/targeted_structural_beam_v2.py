"""Targeted physical verification for existing structural-beam search; no new search.

Prepare copies *only* the plan catalogs of structural-vs-random disagreement
pairs into an isolated routed output directory. Analyze reports all original
pairs, using exact zero for decisions that select the same plan. Never makes
any decision based on physical labels.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from shutil import copyfile
from statistics import mean

MAIN = ('structural_beam', 'random_beam')
KEY = ('circuit_id', 'original_sig', 'repetition')
CATKEY = ('circuit_id','sig')

def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fields):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='raise')
        w.writeheader(); w.writerows(rows)

def jwrite(path, data):
    with open(path,'w',encoding='utf-8') as f:json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False)

def insist(cond,msg):
    if not cond:raise RuntimeError(msg)

def pair_index(rows):
    result=defaultdict(dict)
    for r in rows:
        if r['policy'] not in MAIN:continue
        k=tuple(r[c] for c in KEY)
        insist(r['policy'] not in result[k],f'Duplicate policy {k}/{r["policy"]}')
        result[k][r['policy']]=r
    insist(result,'No structural_beam/random_beam pairs')
    for k,v in result.items():insist(set(v)==set(MAIN),f'Incomplete pair: {k}')
    return result

def prepare(searchdir:Path,outdir:Path):
    insist(searchdir.resolve()!=outdir.resolve(),'Output directory must differ from search directory')
    insist(not outdir.exists() or not any(outdir.iterdir()),'Output directory is not empty: '+str(outdir))
    decisions=read_csv(searchdir/'search_decisions.csv')
    pairs=pair_index(decisions)
    catalog=read_csv(searchdir/'selected_plan_catalog.csv')
    meta=json.loads((searchdir/'search_manifest.json').read_text(encoding='utf-8'))
    insist(meta.get('status')=='search_only_no_routing','Original search manifest is not search-only')
    insist('structural_beam_control' in str(meta.get('schema')),'Unexpected search schema')
    byplan={}
    for r in catalog:
        k=tuple(r[c] for c in CATKEY)
        insist(k not in byplan or r['plan_json']==byplan[k]['plan_json'],f'Plan hash collision: {k}')
        byplan[k]=r
    diff=[]; diag=[]; missing=set(); matched_raw=True; raw_equal_count=0; valid_equal_count=0
    for k in sorted(pairs):
        s=pairs[k]['structural_beam'];r=pairs[k]['random_beam']
        insist(s['family_group']==r['family_group'],'Family mismatch')
        insist(s['original_sig']==r['original_sig'],'Baseline mismatch')
        same=s['pick_sig']==r['pick_sig']
        raw_equal=s['raw_mutation_attempts']==r['raw_mutation_attempts']
        valid_equal=s['n_valid_scored']==r['n_valid_scored']
        raw_equal_count+=raw_equal;valid_equal_count+=valid_equal
        diag.append(dict(circuit_id=k[0],original_sig=k[1],repetition=k[2],family_group=s['family_group'],
            structural_sig=s['pick_sig'],random_sig=r['pick_sig'],same_selected_plan=int(same),
            same_raw_attempts=int(raw_equal),same_valid_scored=int(valid_equal),
            structural_raw_attempts=s['raw_mutation_attempts'],random_raw_attempts=r['raw_mutation_attempts'],
            structural_valid_scored=s['n_valid_scored'],random_valid_scored=r['n_valid_scored']))
        if not same:
            matched_raw &= raw_equal
            diff.extend([s,r])
            missing.update(((k[0],k[1]),(k[0],s['pick_sig']),(k[0],r['pick_sig'])))
    insist(diff,'All final plans are identical, no targeted routing needed')
    absent=missing-set(byplan)
    insist(not absent,f'Plans missing from catalog: {sorted(absent)[:10]}')
    training_circuits={d['circuit_id'] for d in diff}
    insist(training_circuits <= set(meta['qasm_paths']),'Manifest is missing some QASM paths')
    # A family split was frozen before search; verify the search is training-only through the original manifest.
    insist('ml_train_prepare' in meta.get('train_dir',''),'Unexpected source: not the locked Training preparation')
    outdir.mkdir(parents=True,exist_ok=True)
    selected=[byplan[k] for k in sorted(missing)]
    write_csv(outdir/'selected_plan_catalog.csv',selected,list(catalog[0]))
    write_csv(outdir/'search_decisions.csv',diff,list(decisions[0]))
    write_csv(outdir/'all_pair_diagnostics.csv',diag,list(diag[0]))
    meta['status']='search_only_no_routing'
    meta['n_decisions']=len(diff)
    meta['n_unique_routing_plans']=len(selected)
    meta['targeted_routing_subset_of']=str(searchdir.resolve())
    meta['targeted_pair_design']='strictly differing structural and random final plans; same picks are exact zero delta'
    jwrite(outdir/'search_manifest.json',meta)
    info=dict(schema='targeted_disagreement_routing_v2',is_train_only=True,
        original_search_dir=str(searchdir.resolve()),n_all_pairs=len(pairs),n_same_picks=len(pairs)-len(diff)//2,
        n_disagreement_pairs=len(diff)//2,n_routed_plans=len(selected),n_disagreement_circuits=len(training_circuits),
        n_disagreement_families=len({r['family_group'] for r in diff}),
        n_all_equal_raw_attempt_pairs=raw_equal_count,n_all_equal_valid_scored_pairs=valid_equal_count,
        disagreement_equal_raw_attempts=matched_raw,
        expected_routing_jobs_for_five_seeds=5*len(selected),
        note='Search/policies never rerun. Same-plan pairs are exact paired zero. Random-uniform not evaluated.')
    jwrite(outdir/'targeted_subset_manifest.json',info)
    print(f'[PASS] prepared {len(diff)//2} disagreements from {len(pairs)} paired runs, '
          f'{len(selected)} unique (circuit,plan) jobs per seed; {len(training_circuits)} circuits')
    print(f'[INFO] route 5 independent seeds: {len(selected)*5} jobs, versus {len(catalog)*5} if all catalog routed')
    print('[INFO] paired disagreements same raw attempts:', matched_raw)

def analyze(outdir:Path):
    info=json.loads((outdir/'targeted_subset_manifest.json').read_text(encoding='utf-8'))
    insist(info['is_train_only'],'Not a Training-only subset')
    pairs=pair_index(read_csv(outdir/'search_decisions.csv'))
    diag=read_csv(outdir/'all_pair_diagnostics.csv')
    catalog=read_csv(outdir/'selected_plan_catalog.csv')
    lock=json.loads((outdir/'routing_protocol_lock.json').read_text(encoding='utf-8'))
    seeds=lock['seed_set']
    insist(len(seeds)>=2 and len(seeds)==len(set(seeds)) and min(seeds)>=300,'Seeds must be fresh >=300')
    expected={(r['circuit_id'],r['sig'],str(seed)) for r in catalog for seed in seeds}
    raw=read_csv(outdir/'new_seed_routing_results.csv')
    rr={}
    for r in raw:
        k=(r['circuit_id'],r['sig'],r['seed'])
        insist(k not in rr,'Duplicate Routing result '+str(k))
        rr[k]=r
    insist(set(rr)==expected,f'Routing jobs are incomplete: missing={len(expected-set(rr))}, extra={len(set(rr)-expected)}')
    insist(all(r['status']=='ok' for r in raw),'Routing failures: cannot analyze partial data')
    byplan=defaultdict(list)
    for r in raw:
        try:a,b=float(r['swap_count']),float(r['routed_depth'])
        except (TypeError, ValueError):raise RuntimeError('Invalid physical values')
        insist(math.isfinite(a) and math.isfinite(b) and a>=0 and b>=0,'Bad physical value')
        byplan[(r['circuit_id'],r['sig'])].append((a,b))
    costs={k:(mean(x[0] for x in vals),mean(x[1] for x in vals)) for k,vals in byplan.items()}
    measured={}
    drows=[]
    for k,v in sorted(pairs.items()):
        s=v['structural_beam'];r=v['random_beam']; cid=k[0]
        insist(s['pick_sig']!=r['pick_sig'],'Subset contains same-selected-plan pair')
        base=costs[(cid,k[1])];cs=costs[(cid,s['pick_sig'])];cr=costs[(cid,r['pick_sig'])]
        # Positive means Structural has fewer SWAPs / less depth than Random.
        measured[k]=(cr[0]-cs[0],cr[1]-cs[1])
        drows.append(dict(circuit_id=cid,original_sig=k[1],repetition=k[2],family_group=s['family_group'],
            structural_sig=s['pick_sig'],random_sig=r['pick_sig'],
            structural_swap_mean=cs[0],random_swap_mean=cr[0],original_swap_mean=base[0],
            structural_depth_mean=cs[1],random_depth_mean=cr[1],original_depth_mean=base[1],
            structural_swap_gain_abs=base[0]-cs[0],random_swap_gain_abs=base[0]-cr[0],
            structural_depth_gain_abs=base[1]-cs[1],random_depth_gain_abs=base[1]-cr[1],
            structural_minus_random_swap_gain=cr[0]-cs[0],
            structural_minus_random_depth_gain=cr[1]-cs[1],
            same_raw_attempts=int(s['raw_mutation_attempts']==r['raw_mutation_attempts']),
            same_valid_scored=int(s['n_valid_scored']==r['n_valid_scored']),
            structural_valid_scored=s['n_valid_scored'],random_valid_scored=r['n_valid_scored']))
    allpairs=[]
    for r in diag:
        k=tuple(r[c] for c in KEY)
        is_same=int(r['same_selected_plan'])==1
        insist((k not in measured)==is_same,'Mismatch original pair / disagreement subset')
        d=(0.0,0.0) if is_same else measured[k]
        allpairs.append(dict(**r,structural_minus_random_swap_gain=d[0],
            structural_minus_random_depth_gain=d[1],physical_costs_measured=int(not is_same)))
    insist(len(allpairs)==info['n_all_pairs'],'Some original pairs missing')
    family=defaultdict(list)
    for r in allpairs:family[r['family_group']].append(r)
    famrows=[]
    for f,rs in sorted(family.items()):
        deltas=[r['structural_minus_random_swap_gain'] for r in rs]
        famrows.append(dict(family_group=f,n_pairs=len(rs),n_disagreement_pairs=sum(r['physical_costs_measured'] for r in rs),
            mean_paired_swap_gain_difference=mean(deltas),
            mean_paired_depth_gain_difference=mean(r['structural_minus_random_depth_gain'] for r in rs),
            wins=sum(d>1e-9 for d in deltas),losses=sum(d < -1e-9 for d in deltas),ties=sum(abs(d)<=1e-9 for d in deltas)))
    alldelta=[r['structural_minus_random_swap_gain'] for r in allpairs]
    active=[r['structural_minus_random_swap_gain'] for r in allpairs if r['physical_costs_measured']]
    diagdir=outdir
    write_csv(diagdir/'disagreement_physical_pairs.csv',drows,list(drows[0]))
    write_csv(diagdir/'full_paired_method_comparison.csv',allpairs,list(allpairs[0]))
    write_csv(diagdir/'family_paired_summary.csv',famrows,list(famrows[0]))
    summary=dict(schema='targeted_disagreement_routing_v2',training_only=True,routing_seeds=seeds,
        n_all_paired_runs=len(allpairs),n_measured_disagreements=len(active),n_exact_zero_same_picks=len(allpairs)-len(active),
        n_families=len(family),n_disagreement_families=sum(r['n_disagreement_pairs']>0 for r in famrows),
        mean_paired_swap_difference_all=mean(alldelta),
        mean_paired_swap_difference_disagreement_only=mean(active),
        mean_of_family_mean_paired_swap_difference=mean(r['mean_paired_swap_gain_difference'] for r in famrows),
        mean_paired_depth_difference_all=mean(r['structural_minus_random_depth_gain'] for r in allpairs),
        structural_wins=sum(d>1e-9 for d in alldelta),random_wins=sum(d < -1e-9 for d in alldelta),ties=sum(abs(d)<=1e-9 for d in alldelta),
        mean_structural_swap_gain_on_disagreements=mean(r['structural_swap_gain_abs'] for r in drows),
        mean_random_swap_gain_on_disagreements=mean(r['random_swap_gain_abs'] for r in drows),
        n_same_raw_attempts_in_disagreements=sum(r['same_raw_attempts'] for r in drows),
        n_same_valid_count_in_disagreements=sum(r['same_valid_scored'] for r in drows),
        note='TRAIN exploratory, not independent held-out circuit evaluation; 3 reps per Original and 5 physical seeds are correlated. Same selected plan => EXACT zero relative difference and requires no routing. To claim improvement vs initial plans over full corpus, route same-selected nonoriginal plans and originals too.')
    jwrite(diagdir/'disagreement_routing_summary.json',summary)
    print('[PASS] '+str(len(active))+' disagreements physically measured, '+str(len(allpairs))+' total paired comparisons')
    print(f"[RESULT] Structural − Random SWAP gain: all={summary['mean_paired_swap_difference_all']:+.5f}, "
          f"disagreements={summary['mean_paired_swap_difference_disagreement_only']:+.5f}")
    print(f"[RESULT] wins={summary['structural_wins']} random_wins={summary['random_wins']} ties={summary['ties']}")

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    sp=ap.add_subparsers(dest='stage',required=True)
    p=sp.add_parser('prepare');p.add_argument('--search-dir',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    p=sp.add_parser('analyze');p.add_argument('--out-dir',type=Path,required=True)
    a=ap.parse_args()
    if a.stage=='prepare':prepare(a.search_dir,a.out_dir)
    else:analyze(a.out_dir)
