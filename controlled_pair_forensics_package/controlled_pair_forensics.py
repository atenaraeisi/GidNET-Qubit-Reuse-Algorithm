#!/usr/bin/env python3
"""Controlled pair reuse-plan forensics, STAGE A (no routing, no plan generation).

Goal: compare pre-routing structure for matched-width plans of the SAME
original circuit (default system9:P5/P4 and sym9_146:P8/P4). CSV-only
results still written when the exact original QASM is missing. Fine-grained
operation analysis REQUIRES a matched original QASM and is checked against
Experiment 2's quotient interaction graph features to prevent misalignment.

Uses structural_forensics.py in this directory for original validated parser,
CSV consistency checks, and dependency graph reconstruction.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import t as student_t

import structural_forensics as sf

DEFAULT_PAIRS = [('system9.qasm', 'P5', 'P4'), ('sym9_146.qasm', 'P8', 'P4')]


def pyval(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, Path): return str(v)
    raise TypeError(type(v).__name__)


def write_csv(rows, path, columns=None):
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding='utf-8-sig')


def edges_from_chains(chains):
    return {(a, b) for chain in chains for a, b in zip(chain, chain[1:])}


def aligned_chains(a, b):
    """Match unlabeled reused wires by maximum-overlap assignment."""
    if len(a) != len(b):
        raise ValueError('Fixed width must match')
    overlap = np.array([[len(set(x) & set(y)) for y in b] for x in a], dtype=int)
    ia, ib = linear_sum_assignment(-overlap)
    amap = {int(y): int(x) for x, y in zip(ia, ib)}
    assigned = [(int(x), int(y), int(overlap[x, y])) for x, y in zip(ia, ib)]
    if set(amap.values()) != set(range(len(a))):
        raise ValueError('Chain matching not a permutation')
    return amap, assigned


def all_operation_dependencies(ops, chains):
    """Per original logical wire + reuse completion → successor-start edges.

    Gate duration is one unit. This is NOT a hardware scheduling model.
    """
    n = len(ops)
    pred = [set() for _ in range(n)]
    succ = [set() for _ in range(n)]
    last = {}
    first = {}
    reuse_links = []
    for op in ops:
        for q in op.qubits:
            if q not in first: first[q] = op.idx
            if q in last:
                pred[op.idx].add(last[q]); succ[last[q]].add(op.idx)
            last[q] = op.idx
    for chain in chains:
        for a,b in zip(chain,chain[1:]):
            u,v=last[a],first[b]
            pred[v].add(u);succ[u].add(v)
            reuse_links.append((a,b,u,v))
    degree = [len(x) for x in pred]
    todo = deque(i for i,d in enumerate(degree) if d==0)
    order=[]
    while todo:
        u=todo.popleft();order.append(u)
        for v in sorted(succ[u]):
            degree[v]-=1
            if degree[v]==0:todo.append(v)
    if len(order)!=n:
        raise ValueError('Invalid plan: dependency cycle in operation graph')
    level=[1]*n
    for u in order:
        for v in succ[u]:level[v]=max(level[v], level[u]+1)
    depth=max(level)
    tail=[1]*n
    for u in reversed(order):
        for v in succ[u]:tail[u]=max(tail[u],tail[v]+1)
    slack=[depth-(level[u]+tail[u]-1) for u in range(n)]
    return {'pred':pred,'succ':succ,'level':level,'tail':tail,'slack':slack,
            'critical':{i for i,x in enumerate(slack) if x==0},'depth':depth,
            'reuse_links':reuse_links}


def numeric_agreement(ops,chains,reference,circuit,plan):
    active={q for op in ops for q in op.qubits}
    where=sf.validate_plan_membership(chains,active,len(chains))
    p, w, deg, own, n_two=sf.stats_quotient(ops,chains,where)
    actual={'max_weighted_degree':max(w),'weighted_degree_cv':float(np.std(w)/np.mean(w)) if np.mean(w) else 0.,
            'quotient_unique_edges':len(p)}
    for k, value in actual.items():
        if not np.isclose(value, float(reference[k]), atol=1e-7):
            raise AssertionError(f'QASM / stored plan mismatch {circuit}:{plan} {k} original={reference[k]} derived={value}')
    return where,p,w,deg,own


def paired_stats(raw,circuit,better,worse,metric):
    sub=raw[raw.circuit==circuit]
    a=sub[sub.plan_id==better][['routing_seed',metric]].rename(columns={metric:'a'})
    b=sub[sub.plan_id==worse][['routing_seed',metric]].rename(columns={metric:'b'})
    if a.routing_seed.duplicated().any() or b.routing_seed.duplicated().any():raise ValueError('Duplicate routing seeds')
    if len(a)!=len(b):raise ValueError('Mismatch routing trials between plans')
    d=a.merge(b,on='routing_seed',validate='one_to_one').sort_values('routing_seed')
    if len(d)!=len(a):raise ValueError('Routing seeds differ between plans')
    diff=(d.b-d.a).to_numpy(dtype=float)
    std=float(np.std(diff,ddof=1)) if len(diff)>1 else 0.
    ci=float(student_t.ppf(.975,len(diff)-1)*std/np.sqrt(len(diff))) if len(diff)>1 else float('nan')
    return {'n':len(diff),'mean_worse_minus_better':float(np.mean(diff)),
            'sd_matched_diff':std,'descriptive_95ci_low':float(np.mean(diff)-ci),
            'descriptive_95ci_high':float(np.mean(diff)+ci),
            'worse_greater':int(sum(diff>0)),'same':int(sum(diff==0)),
            'worse_less':int(sum(diff<0))}


def fixed_summary(circuit,aid,bid,d):
    df=d['fresh']; rows=df[(df.circuit==circuit)&(df.plan_id.isin([aid,bid]))]
    if len(rows)!=2:raise ValueError(f'Could not find both {circuit}:{aid}:{bid} in fresh results')
    a=rows.set_index('plan_id').loc[aid]
    b=rows.set_index('plan_id').loc[bid]
    if int(a.width)!=int(b.width):raise ValueError('Width mismatch')
    features=d['features'].set_index(['circuit','plan_id'])
    assert (circuit,aid) in features.index and (circuit,bid) in features.index
    af,bf=features.loc[(circuit,aid)],features.loc[(circuit,bid)]
    ca,cb=sf.parse_chains(a.plan_signature),sf.parse_chains(b.plan_signature)
    if set(q for x in ca for q in x)!=set(q for x in cb for q in x):
        raise ValueError('Active qubit memberships differ')
    emap,alignment=aligned_chains(ca,cb)
    wa={q:i for i,c in enumerate(ca) for q in c}
    wb={q:emap[j] for j,c in enumerate(cb) for q in c}
    moved=[q for q in sorted(wa) if wa[q]!=wb[q]]
    edge_a,edge_b=edges_from_chains(ca),edges_from_chains(cb)
    summary={'circuit':circuit,'plan_A':aid,'plan_B':bid,
             'plan_A_routed_depth':float(a.fresh_mean_routed_depth),
             'plan_B_routed_depth':float(b.fresh_mean_routed_depth),
             'B_minus_A_depth':float(b.fresh_mean_routed_depth-a.fresh_mean_routed_depth),
             'plan_A_swaps':float(a.fresh_mean_swaps),'plan_B_swaps':float(b.fresh_mean_swaps),
             'B_minus_A_swaps':float(b.fresh_mean_swaps-a.fresh_mean_swaps),
             'active_width':int(a.width),
             'normalized_logical_depth_A':int(a.normalized_logical_depth),
             'normalized_logical_depth_B':int(b.normalized_logical_depth),
             'max_weighted_degree_A':float(af.max_weighted_degree),
             'max_weighted_degree_B':float(bf.max_weighted_degree),
             'weighted_degree_cv_A':float(af.weighted_degree_cv),
             'weighted_degree_cv_B':float(bf.weighted_degree_cv),
             'chain_alignment_max_overlap':int(sum(z for _,_,z in alignment)),
             'n_logical_qubits_different_chain_after_alignment':len(moved),
             'moved_logical_qubits_after_alignment':str(moved),
             'reuse_edges_only_A':len(edge_a-edge_b),'reuse_edges_only_B':len(edge_b-edge_a),
             'matched_depth':paired_stats(d['raw'],circuit,aid,bid,'routed_depth'),
             'matched_swaps':paired_stats(d['raw'],circuit,aid,bid,'swap_count'),
             'plan_A_chains':str(ca),'plan_B_chains':str(cb)}
    changed=[]
    for edge in sorted(edge_a|edge_b):
        if (edge in edge_a) != (edge in edge_b):
            changed.append({'circuit':circuit,'plan_A':aid,'plan_B':bid,'reuse_predecessor':edge[0],
                            'reuse_successor':edge[1], 'A_has_edge':edge in edge_a,'B_has_edge':edge in edge_b})
    chain_rows=[]
    for i,j,n_inter in alignment:
        chain_rows.append({'circuit':circuit,'plan_A':aid,'plan_B':bid,'aligned_chain_id':i,
                           'chain_idx_A':i,'chain_idx_B':j,'A_chain':str(ca[i]),'B_chain':str(cb[j]),
                           'overlap_qubits':n_inter,
                           'removed_logical_qubits':str(sorted(set(ca[i])-set(cb[j]))),
                           'added_logical_qubits':str(sorted(set(cb[j])-set(ca[i]))),
                           'same_order':ca[i]==cb[j]})
    return summary,ca,cb,emap,wa,wb,moved,changed,chain_rows


def focus_logic(ops,moved):
    out=[]
    involved=set(moved)
    for q in sorted(involved):
        with_q=[op for op in ops if q in op.qubits]
        two=[op for op in with_q if len(op.qubits)==2]
        partners=Counter(next(x for x in op.qubits if x!=q) for op in two)
        out.append({'logical_qubit':q,'n_total_gates':len(with_q),'n_2q_gates':len(two),
                    'first_qasm_gate_index':with_q[0].idx if with_q else None,
                    'last_qasm_gate_index':with_q[-1].idx if with_q else None,
                    'partner_qubits_by_frequency':str(partners.most_common()),
                    '2q_event_indices':str(tuple(op.idx for op in two))})
    return out


def plot_asap_shift(ops,ga,gb,out):
    shift=np.array(gb['level'])-np.array(ga['level'])
    fig,ax=plt.subplots(figsize=(11,4.2))
    ax.scatter(range(len(ops)),shift,s=13,alpha=0.65)
    ax.axhline(0,linewidth=1)
    ax.set(xlabel='Original QASM gate index',ylabel='ASAP layer B − A (unit gate time)',
           title='Pre-routing operation-level schedule shift; NOT a hardware timing estimate')
    fig.tight_layout();fig.savefig(out,dpi=170);plt.close(fig)


def plot_interaction_changes(events, aid,bid,out):
    subset=[e for e in events if e['chain_pair_changed_after_alignment']]
    fig,ax=plt.subplots(figsize=(11,4.5))
    if subset:
        x=[e['qasm_gate_index'] for e in subset]
        a=[e['asap_layer_A'] for e in subset];b=[e['asap_layer_B'] for e in subset]
        ax.scatter(x,a,marker='o',label=f'{aid}: ASAP layer')
        ax.scatter(x,b,marker='x',label=f'{bid}: ASAP layer')
        for e in subset:
            ax.plot([e['qasm_gate_index']]*2,[e['asap_layer_A'],e['asap_layer_B']],alpha=.22)
        ax.legend()
    else:
        ax.text(.5,.5,'No 2q events with changed chain-pair after alignment',ha='center',transform=ax.transAxes)
    ax.set(xlabel='Original QASM 2q gate index (shared identity)',ylabel='ASAP gate layer',
           title='Changed interaction endpoints on reused wires (pre-routing only)')
    fig.tight_layout();fig.savefig(out,dpi=170);plt.close(fig)


def plot_chain_windows(window_rows,aid,bid,out):
    """Compare 2q incident event distribution across fixed original-index windows.
    Counts reflect original operation order, NOT real concurrent time slots.
    """
    a=[x for x in window_rows if x['plan']==aid]
    b=[x for x in window_rows if x['plan']==bid]
    # across chain readout may be dense: heatmaps with explicit normalization are less misleading.
    k=max((x['aligned_chain_id'] for x in window_rows),default=-1)+1
    bins=max((x['original_gate_index_bin'] for x in window_rows),default=-1)+1
    h1=np.zeros((k,bins),dtype=int);h2=np.zeros_like(h1)
    for z in a:h1[z['aligned_chain_id'],z['original_gate_index_bin']]=z['n_2q_incident_events']
    for z in b:h2[z['aligned_chain_id'],z['original_gate_index_bin']]=z['n_2q_incident_events']
    fig,ax=plt.subplots(figsize=(10,4.7))
    im=ax.imshow(h2-h1,aspect='auto',cmap='coolwarm')
    ax.set_xticks(range(bins));ax.set_yticks(range(k));ax.set_yticklabels([f'Chain {i}' for i in range(k)])
    ax.set(xlabel='Original QASM gate-index bin (identical boundaries)',title=f'2q incidences: {bid} − {aid} per aligned chain')
    fig.colorbar(im,ax=ax,label='Count difference')
    fig.tight_layout();fig.savefig(out,dpi=170);plt.close(fig)


def analyze_gate_level(circuit,aid,bid,qasm,ca,cb,emap,wa,wb,moved,d,output,summary,n_windows):
    ndecl,ops=sf.parse_qasm(qasm)
    features=d['features'].set_index(['circuit','plan_id'])
    amap,paira,wdega,dega,own=numeric_agreement(ops,ca,features.loc[(circuit,aid)],circuit,aid)
    bmap,pairb,wdegb,degb,_=numeric_agreement(ops,cb,features.loc[(circuit,bid)],circuit,bid)
    ga=all_operation_dependencies(ops,ca);gb=all_operation_dependencies(ops,cb)
    sum_2q=sum(len(op.qubits)==2 for op in ops)
    summary.update({'qasm_status':'VERIFIED_FULL_GATE_ANALYSIS','qasm_path_used':str(qasm),
                    'declared_original_qubits':ndecl,'n_original_gates':len(ops),'n_2q_gates':sum_2q,
                    'asap_depth_A':ga['depth'],'asap_depth_B':gb['depth'],
                    'n_operations_changed_asap_layer':sum(x!=y for x,y in zip(ga['level'],gb['level'])),
                    'n_operations_earlier_in_B':sum(y<x for x,y in zip(ga['level'],gb['level'])),
                    'n_operations_later_in_B':sum(y>x for x,y in zip(ga['level'],gb['level'])),
                    'n_critical_ops_A':len(ga['critical']),'n_critical_ops_B':len(gb['critical']),
                    'n_ops_changed_critical_membership':len(ga['critical']^gb['critical'])})
    ops_shift=[]
    for op in ops:
        shift=gb['level'][op.idx]-ga['level'][op.idx]
        ca_c=op.idx in ga['critical'];cb_c=op.idx in gb['critical']
        entry={'circuit':circuit,'plan_A':aid,'plan_B':bid,'qasm_gate_index':op.idx,
               'gate_name':op.name,'logical_qubits':str(op.qubits),'n_qubits':len(op.qubits),
               'asap_layer_A':ga['level'][op.idx],'asap_layer_B':gb['level'][op.idx],
               'B_minus_A_asap_layer':shift,'critical_A':ca_c,'critical_B':cb_c,
               'slack_A':ga['slack'][op.idx],'slack_B':gb['slack'][op.idx],
               'involves_reassigned_logical_qubit':any(q in moved for q in op.qubits)}
        if shift or ca_c!=cb_c:
            ops_shift.append(entry)
    # Cross-plan links annotated with original gate index; gate semantics stay identical.
    boundaries=[]
    for p,chains,g in [(aid,ca,ga),(bid,cb,gb)]:
        for a,b,u,v in g['reuse_links']:
            boundaries.append({'circuit':circuit,'plan':p,'from_logical':a,'to_logical':b,
                               'last_gate_from':u,'first_gate_to':v,
                               'on_some_critical_path':bool(u in g['critical'] and v in g['critical'] and g['level'][v] == g['level'][u] + 1),
                               'asap_separation':g['level'][v]-g['level'][u]})
    # Re-identify all interactions by original QASM op index, never by post-routing gate index.
    events=[];changed_pair_count=0
    incident_per_q=Counter()
    logical_partner=defaultdict(Counter)
    for op in ops:
        if len(op.qubits)!=2:continue
        q,r=op.qubits
        A=tuple(sorted((wa[q],wa[r])))
        B=tuple(sorted((wb[q],wb[r])))
        changed=A!=B
        changed_pair_count+=int(changed)
        incident_per_q[q]+=1;incident_per_q[r]+=1
        logical_partner[q][r]+=1;logical_partner[r][q]+=1
        events.append({'circuit':circuit,'plan_A':aid,'plan_B':bid,'qasm_gate_index':op.idx,'gate_name':op.name,
                       'logical_q0':q,'logical_q1':r,'physical_chain_pair_A':str(A),
                       'physical_chain_pair_B_aligned_to_A':str(B),
                       'chain_pair_changed_after_alignment':changed,
                       'involves_reassigned_logical_qubit':q in moved or r in moved,
                       'asap_layer_A':ga['level'][op.idx],'asap_layer_B':gb['level'][op.idx],
                       'B_minus_A_asap_layer':gb['level'][op.idx]-ga['level'][op.idx],
                       'critical_A':op.idx in ga['critical'],'critical_B':op.idx in gb['critical']})
    summary.update({'n_2q_events_changed_chain_pair':changed_pair_count,
                    'fraction_2q_events_changed_chain_pair':changed_pair_count/max(1,sum_2q),
                    'n_2q_events_involving_reassigned_logical_qubits':sum(e['involves_reassigned_logical_qubit'] for e in events)})
    partner_delta=[]
    newpairb=Counter()
    for (i,j),count in pairb.items():
        a,b=sorted((emap[i],emap[j]))
        newpairb[(a,b)]+=count
    for pair in sorted(set(paira)|set(newpairb)):
        m,n=pair
        partner_delta.append({'circuit':circuit,'plan_A':aid,'plan_B':bid,'aligned_chain_i':m,
                              'aligned_chain_j':n,'n_2q_A':paira[pair],
                              'n_2q_B':newpairb[pair], 'B_minus_A':newpairb[pair]-paira[pair]})
    chain_d=[]
    qmap_a={q:i for i,c in enumerate(ca) for q in c}
    qmap_b={q:emap[i] for i,c in enumerate(cb) for q in c}
    def ordered_events_for_chain(chains,idx,where):
        events_on_logical=defaultdict(list)
        for op in ops:
            if len(op.qubits)==2:
                q,r=op.qubits
                events_on_logical[q].append((op.idx,where[r],r))
                events_on_logical[r].append((op.idx,where[q],q))
        # This concatenates chronological per-logical sequences IN CHAIN ORDER;
        # NOT actual global execution time for reused wires.
        return [entry for q in chains[idx] for entry in events_on_logical[q]]
    for i in range(len(ca)):
        jb=next(j for j,x in emap.items() if x==i)
        ae=ordered_events_for_chain(ca,i,qmap_a)
        be=ordered_events_for_chain(cb,jb,qmap_b)
        sa=[p for _,p,_ in ae]; sb=[p for _,p,_ in be]
        aa=sum(x!=y for x,y in zip(sa,sa[1:])); bb=sum(x!=y for x,y in zip(sb,sb[1:]))
        chain_d.append({'circuit':circuit,'plan_A':aid,'plan_B':bid,'aligned_chain_id':i,
                        'logical_chain_A':str(ca[i]),'logical_chain_B':str(cb[jb]),
                        'incident_2q_A':len(sa),'incident_2q_B':len(sb),
                        'B_minus_A_incident_2q':len(sb)-len(sa),
                        'partner_switches_A':aa,'partner_switches_B':bb,
                        'B_minus_A_partner_switches':bb-aa,
                        'partner_sequence_A':str(tuple(sa)),
                        'partner_sequence_B_aligned':str(tuple(sb)),
                        'original_2q_gate_indices_A':str(tuple(x for x,_,_ in ae)),
                        'original_2q_gate_indices_B':str(tuple(x for x,_,_ in be))})
    moved_rows=[]
    for qrow in focus_logic(ops,moved):
        q=qrow['logical_qubit']
        moved_rows.append({'circuit':circuit,'plan_A':aid,'plan_B':bid,
                           'chain_A':wa[q],'chain_B_aligned':wb[q],**qrow})
    # FIXED original gate-index windows, so windows remain same between plans.
    bins=[]
    for plan,chains,where in [(aid,ca,wa),(bid,cb,wb)]:
        count=Counter()
        for op in ops:
            if len(op.qubits)!=2:continue
            t=min(n_windows-1,int(op.idx*n_windows/len(ops)))
            for q in op.qubits:count[(where[q],t)]+=1
        for k in range(len(ca)):
            for t in range(n_windows):
                bins.append({'circuit':circuit,'plan':plan,'aligned_chain_id':k,
                             'original_gate_index_bin':t,'n_2q_incident_events':count[(k,t)]})
    # Cross-plan differences include changed gates throughout downstream DAG, not only moved qubits.
    def p(name):return output/name
    stem=circuit.replace('.qasm','')+f'__{aid}_vs_{bid}'
    write_csv(ops_shift,p(stem+'__operation_asap_or_critical_changes.csv'))
    write_csv(events,p(stem+'__all_two_qubit_gate_events.csv'))
    write_csv([x for x in events if x['chain_pair_changed_after_alignment']],p(stem+'__changed_two_qubit_chain_pairs.csv'))
    write_csv(partner_delta,p(stem+'__quotient_pair_load_differences.csv'))
    write_csv(chain_d,p(stem+'__per_chain_sequences.csv'))
    write_csv(boundaries,p(stem+'__reuse_boundary_gate_links.csv'))
    write_csv(moved_rows,p(stem+'__moved_logical_qubits.csv'))
    write_csv(bins,p(stem+'__original_index_window_load.csv'))
    figpath=output/'figures';figpath.mkdir(exist_ok=True)
    plot_asap_shift(ops,ga,gb,figpath/(stem+'__asap_level_shift.png'))
    plot_interaction_changes(events,aid,bid,figpath/(stem+'__changed_interaction_events.png'))
    plot_chain_windows(bins,aid,bid,figpath/(stem+'__original_index_window_load_delta.png'))
    return summary


def run(args):
    args.out_dir.mkdir(parents=True,exist_ok=True)
    d=sf.read_inputs(args.data_dir)
    sf.audit_data(d)
    qdir=args.qasm_dir or args.data_dir
    allpaths=defaultdict(list)
    for f in qdir.rglob('*.qasm'): allpaths[f.name].append(f)
    summaries=[];changed=[];aligned=[]
    for circuit,aid,bid in args.pair:
        x,ca,cb,emap,wa,wb,moved,edge_changes,chain_rows=fixed_summary(circuit,aid,bid,d)
        rawmatches=allpaths.get(circuit,[])
        if len(rawmatches)==1:
            x=analyze_gate_level(circuit,aid,bid,rawmatches[0],ca,cb,emap,wa,wb,moved,d,args.out_dir,x,args.windows)
            print(f'{circuit} {aid}/{bid}: full gate forensics finished, QASM validated vs Experiment 2')
        elif not rawmatches:
            x['qasm_status']='MISSING_QASM__SUMMARY_ONLY'
            print(f'{circuit} {aid}/{bid}: missing QASM, only plan/old feature/30-seed comparisons')
        else:
            raise FileExistsError(f'Ambiguous QASM for {circuit}: {rawmatches}')
        summaries.append(x);changed.extend(edge_changes);aligned.extend(chain_rows)
    write_csv([{k:v for k,v in x.items() if not isinstance(v,dict)} for x in summaries],args.out_dir/'pair_summary.csv')
    write_csv(changed,args.out_dir/'changed_reuse_boundaries.csv')
    write_csv(aligned,args.out_dir/'aligned_reuse_chains.csv')
    report=['# Controlled pair structural forensics — exploratory (Stage A)','',
            'No GidNET generation or SABRE routing was performed. These are descriptive differences, NOT causal explanations.',
            'Plans are fixed-width from same original circuit. 30-seed outcomes come from previously saved routing results.',
            'Gate-level reconstruction is shown ONLY when corresponding original QASM is present and cross-checked against Experiment 2.','']
    for x in summaries:
        report += [f"## {x['circuit']}: {x['plan_A']} versus {x['plan_B']}",'',
                   f"- Routed depth (30 seeds): {x['plan_A_routed_depth']:.3f} → {x['plan_B_routed_depth']:.3f}, B−A = {x['B_minus_A_depth']:+.3f}.",
                   f"- SWAP count: {x['plan_A_swaps']:.3f} → {x['plan_B_swaps']:.3f}, B−A = {x['B_minus_A_swaps']:+.3f}.",
                   f"- Matched seed depth comparison: {x['matched_depth']['worse_greater']} seeds B > A, {x['matched_depth']['same']} ties, {x['matched_depth']['worse_less']} seeds B < A; n={x['matched_depth']['n']}.",
                   f"- Descriptive paired-depth 95% t interval: [{x['matched_depth']['descriptive_95ci_low']:.3f}, {x['matched_depth']['descriptive_95ci_high']:.3f}]. Selected after inspecting routing outcomes: not a confirmatory interval.",
                   f"- Normalized logical depth: {x['normalized_logical_depth_A']} / {x['normalized_logical_depth_B']}; Max weighted degree: {x['max_weighted_degree_A']:.0f} / {x['max_weighted_degree_B']:.0f}.",
                   f"- Reuse edges present only in A / B: {x['reuse_edges_only_A']} / {x['reuse_edges_only_B']}; reassigned logical qubits after chain alignment: {x['moved_logical_qubits_after_alignment']}.",
                   f"- QASM status: **{x['qasm_status']}**."]
        if x['qasm_status'].startswith('VERIFIED'):
            report += [f"- Original QASM: {x['n_original_gates']} gates ({x['n_2q_gates']} 2q). ASAP depth A / B: {x['asap_depth_A']} / {x['asap_depth_B']}.",
                       f"- Operations with changed ASAP level: {x['n_operations_changed_asap_layer']}; earlier in B: {x['n_operations_earlier_in_B']}; later in B: {x['n_operations_later_in_B']}.",
                       f"- Two-qubit gates with changed reused-wire chain pair (after alignment): {x['n_2q_events_changed_chain_pair']} ({x['fraction_2q_events_changed_chain_pair']:.1%}).",
                       f"- Gates entering/leaving the unit-gate-time critical-path set: {x['n_ops_changed_critical_membership']}.",
                       '- Detailed outputs include moved-qubit interaction history, changed 2q gate endpoints, reuse boundary gate links, pre-routing ASAP shifts, aligned chain partner sequences, and fixed-original-order load bins.']
        else:
            report += ['- **No gate-level QASM analysis was attempted**. Re-run locally with `--qasm-dir` pointing to the original benchmarks directory.']
        report.append('')
    report+=['## What these results can and cannot say','',
             '1. A plan-level delta in routed depth is not necessarily caused by the specific changed reuse edge. Hardware layout and SABRE decisions also matter.',
             '2. Reused-wire chain IDs are arbitrary. Chain alignment maximizes logical-qubit overlap; ambiguous symmetric matchings may have multiple equally good alignments.',
             '3. ASAP layers assume each original gate lasts one logical time unit. These are **not** routed depth, device duration, or gate wait time.',
             '4. Equal global logical depth / MaxWD can hide local temporal shifts and changed pairwise interactions.',
             '5. A paired 30-seed gap quantifies conditional route variability, but the selected pairs were chosen after looking at the same data. Confirm findings on other plans/circuits.',
             '6. Use Stage B (targeted routed-circuit inspection) only if Stage A provides a focused structural hypothesis; no routing performed here.',
             '', '## Suggested next inspection',
             '', 'Inspect `*__moved_logical_qubits.csv`, `*__changed_two_qubit_chain_pairs.csv`, `*__operation_asap_or_critical_changes.csv`, and the figures in `figures/`. Look for a change that precedes/aligns with depth pressure, then test it on more fixed-width plans.',
             '']
    (args.out_dir/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    (args.out_dir/'audit.json').write_text(json.dumps({'num_pairs':len(summaries),'qasm_complete_pairs':sum(x['qasm_status'].startswith('VERIFIED') for x in summaries),
                                              'qasm_missing_pairs':[x['circuit'] for x in summaries if not x['qasm_status'].startswith('VERIFIED')],
                                              'no_new_routing':True,'no_new_plan_generation':True,
                                              'pair_summary':summaries},indent=2,ensure_ascii=False,default=pyval),encoding='utf-8')
    print('Outputs:',args.out_dir.resolve())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--qasm-dir',type=Path,default=None)
    p.add_argument('--out-dir',type=Path,default=Path('controlled_pair_results'))
    p.add_argument('--pair',nargs=3,action='append',metavar=('CIRCUIT_QASM','PLAN_A','PLAN_B'),
                   help='Repeatable; default system9 P5 P4 and sym9_146 P8 P4')
    p.add_argument('--windows',type=int,default=8,help='Number of fixed original QASM gate-index bins, default 8')
    args=p.parse_args()
    if args.windows<2 or args.windows>100:raise ValueError('windows must be 2..100')
    if args.pair is None: args.pair=DEFAULT_PAIRS
    run(args)

if __name__=='__main__':main()
