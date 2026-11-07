#!/usr/bin/env python3
"""Exploratory fixed-width reuse-plan *pre-routing* structural forensics.

Input: Experiment 1 fresh-30 / raw CSVs, Experiment 2 feature CSV, matching
original OpenQASM 2 files. No new plan generation, no routing, no Qiskit.

Intended for GidNET's static original QASM benchmark subset. Not a semantic
validity checker: use independent checker/semantic suite separately.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict, deque, Counter
from dataclasses import dataclass
from pathlib import Path
import json
import math
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as t_distribution
from scipy.optimize import linear_sum_assignment

QUBIT_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]')
REG_RE = re.compile(r'^(qreg|creg)\s+([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]$', re.I)
GATE_RE = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)(?:\([^;]*\))?\s+(.+)$')


@dataclass(frozen=True)
class Operation:
    idx: int
    name: str
    qubits: tuple[int, ...]


def parse_qasm(qasm_path: Path) -> tuple[int, list[Operation]]:
    """Parse *supported static QASM2 gate subset*, preserving original ordering.

    Abort on unsupported control-flow/non-gate constructs; never silently skip
    transformations that would corrupt the dependency analysis.
    """
    raw = qasm_path.read_text(encoding='utf-8-sig')
    raw = re.sub(r'//[^\n]*', '', raw)
    raw = re.sub(r'/\*.*?\*/', '', raw, flags=re.S)
    parts = [s.strip() for s in raw.split(';') if s.strip()]
    qregs: dict[str,int] = {}
    ops: list[Operation] = []
    for s in parts:
        if s.startswith(('OPENQASM ', 'include ')):
            continue
        m = REG_RE.fullmatch(s)
        if m:
            typ,name,n = m.group(1).lower(),m.group(2),int(m.group(3))
            if typ == 'qreg':
                qregs[name] = n
            continue
        if s.startswith(('gate ', 'opaque ', 'if ', 'while ', 'for ')) or '{' in s or '}' in s:
            raise ValueError(f'Unsupported QASM2 definition/control-flow in {qasm_path.name}: {s[:85]}')
        if s.startswith(('measure ', 'reset ', 'barrier ')):
            # A non-unitary original circuit needs a fuller semantic-aware parser.
            # For measure/barrier, reject rather than silently omitting dependencies.
            raise ValueError(f'Original QASM has measurement/reset/barrier; this pilot gate parser intentionally fails closed: {s[:85]}')
        m=GATE_RE.fullmatch(s)
        if not m:
            raise ValueError(f'Cannot parse QASM statement in {qasm_path.name}: {s[:85]}')
        name,args = m.group(1).lower(),m.group(2)
        refs=QUBIT_RE.findall(args)
        if not refs or re.sub(QUBIT_RE, '', args).replace(',','').strip():
            raise ValueError(f'Unsupported gate operands: {s[:85]}')
        if any(reg not in qregs or int(i) >= qregs[reg] for reg,i in refs):
            raise ValueError(f'Qubit reference outside declared qreg: {s[:85]}')
        if len(qregs)!=1:
            raise ValueError('Pilot script supports one QASM quantum register; do not flatten multiple registers silently.')
        qubits=tuple(int(i) for reg,i in refs)
        if len(set(qubits))!=len(qubits):
            raise ValueError(f'Gate has repeated qubit argument: {s[:85]}')
        ops.append(Operation(len(ops),name,qubits))
    if len(qregs)!=1 or not ops:
        raise ValueError(f'QASM must have one qreg and nonempty gates: {qasm_path}')
    return next(iter(qregs.values())),ops


def parse_chains(s: str) -> tuple[tuple[int,...],...]:
    x=ast.literal_eval(str(s))
    if not isinstance(x,(tuple,list)) or not x or not all(isinstance(c,(tuple,list)) and c for c in x):
        raise ValueError(f'Bad plan signature: {str(s)[:100]}')
    if any(not isinstance(q,int) or isinstance(q,bool) for c in x for q in c):
        raise ValueError('Non-integer chain element')
    return tuple(tuple(c) for c in x)


def validate_plan_membership(chains,active,width):
    flat=[q for c in chains for q in c]
    if len(flat)!=len(set(flat)) or set(flat)!=active or len(chains)!=int(width):
        raise ValueError('Plan membership/target-width mismatch')
    return {q:i for i,c in enumerate(chains) for q in c}


def stats_quotient(ops,chains,where):
    k=len(chains)
    pair_counts=Counter()
    incident=[0]*k
    own_degree=Counter()
    gate_two=0
    for op in ops:
        if len(op.qubits)!=2: continue
        a,b=op.qubits
        own_degree[a]+=1;own_degree[b]+=1;gate_two+=1
        ca,cb=where[a],where[b]
        if ca==cb:
            raise ValueError(f'2q gate within same reuse chain: {op.name} on ({a},{b}). Wrong plan or qubit indexing?')
        pair_counts[tuple(sorted((ca,cb)))]+=1
        incident[ca]+=1;incident[cb]+=1
    weighted_degrees=[sum(w for (x,y),w in pair_counts.items() if x==c or y==c) for c in range(k)]
    quotient_degrees=[sum(1 for x,y in pair_counts if x==c or y==c) for c in range(k)]
    return pair_counts,weighted_degrees,quotient_degrees,own_degree,gate_two


def operation_dag(ops,chains):
    """QASM per-logical-wire order plus one edge per reuse boundary."""
    pred=[set() for _ in ops]; succ=[set() for _ in ops]
    def edge(i,j):
        if i==j: raise ValueError('Self-loop in operation DAG')
        succ[i].add(j);pred[j].add(i)
    previous:dict[int,int]={}
    first:dict[int,int]={}
    for op in ops:
        for q in op.qubits:
            if q not in first: first[q]=op.idx
            if q in previous: edge(previous[q],op.idx)
            previous[q]=op.idx
    added=[]
    for chain in chains:
        for a,b in zip(chain,chain[1:]):
            # Every plan-active qubit has an operation in this supported parser.
            if a not in previous or b not in first:
                raise ValueError('Reuse boundary references a qubit with no gate')
            edge(previous[a],first[b]);added.append((previous[a],first[b],a,b))
    indeg=[len(x) for x in pred]
    dq=deque(i for i,x in enumerate(indeg) if x==0)
    order=[]
    while dq:
        u=dq.popleft();order.append(u)
        for v in succ[u]:
            indeg[v]-=1
            if indeg[v]==0: dq.append(v)
    if len(order)!=len(ops):
        raise ValueError('Dependency cycle in combined gate DAG; reject plan')
    earliest=[1]*len(ops)
    for u in order:
        for v in succ[u]: earliest[v]=max(earliest[v],earliest[u]+1)
    depth=max(earliest)
    backward=[1]*len(ops)
    for u in reversed(order):
        for v in succ[u]: backward[u]=max(backward[u],backward[v]+1)
    critical={i for i in range(len(ops)) if earliest[i]+backward[i]-1 == depth}
    critical_reuses=sum(earliest[u]+backward[v]==depth for u,v,_,_ in added)
    return earliest,depth,critical,critical_reuses


def chain_partner_sequences(ops,chains,where):
    """Order 2q partners using chronological logical-wire order within each reuse chain."""
    per_logical=defaultdict(list)
    for op in ops:
        if len(op.qubits)==2:
            a,b=op.qubits
            per_logical[a].append(where[b]);per_logical[b].append(where[a])
    seqs=[]
    for c in chains:
        seqs.append([p for q in c for p in per_logical[q]])
    return seqs


def temporal_features(ops,levels,chain_map,num_chains,depth,bins=8):
    # Bin schedule into equal fractions of its *own pre-routing DAG depth*.
    # Relative bins capture within-plan scheduling structure, not hardware travel time.
    h=np.zeros((bins,num_chains),dtype=int)
    for op in ops:
        if len(op.qubits)!=2:continue
        b=min(bins-1, int((levels[op.idx]-1)*bins/max(1,depth)))
        for q in op.qubits: h[b,chain_map[q]]+=1
    total=h.sum()
    chain_total=h.sum(axis=0)
    nonzero_chain=[]
    for c in range(num_chains):
        if chain_total[c]:
            nonzero_chain.append(float(h[:,c].max()/chain_total[c]))
    # Max and average burst fraction within a reused wire's relative scheduling window.
    return {
       'max_window_2q_chain_fraction':max(nonzero_chain,default=0.0),
       'mean_window_2q_chain_fraction':float(np.mean(nonzero_chain)) if nonzero_chain else 0.0,
       'max_2q_events_in_plan_bin':int(h.sum(axis=1).max())//2 if len(h) else 0,
       'temporal_bin_2q_cv':float(np.std(h.sum(axis=1))/np.mean(h.sum(axis=1))) if np.mean(h.sum(axis=1)) else 0.0,
       'time_bin_matrix':h
    }


def compute_features(ops,chains,num_qubits,cols_depth):
    active={q for op in ops for q in op.qubits}
    where=validate_plan_membership(chains,active,len(chains))
    k=len(chains)
    pairs,wdeg,deg,own_degree,n_2q=stats_quotient(ops,chains,where)
    levels,gate_depth,critical,critical_reuses=operation_dag(ops,chains)
    seqs=chain_partner_sequences(ops,chains,where)
    switches=[sum(x!=y for x,y in zip(s,s[1:])) for s in seqs]
    switch_den=sum(max(0,len(s)-1) for s in seqs)
    # Pair interleaving / fragmentation is a new structural candidate signal,
    # not necessarily causal; switch_rate measures how often a chain must
    # change the physical interaction partner as gates progress.
    total_switch=sum(switches)
    runs=[1+switches[i] if len(s)>0 else 0 for i,s in enumerate(seqs)]
    segs=sum(runs)
    two_critical=sum(len(ops[i].qubits)==2 for i in critical)
    t=temporal_features(ops,levels,where,k,gate_depth)
    clen=np.array([len(c) for c in chains],dtype=float)
    f={
        'active_width':k,
        'full_declared_width_estimate':k+num_qubits-len(active),
        'n_active_logical_qubits':len(active),
        'n_operations':len(ops),
        'n_2q_gates':n_2q,
        'n_reuse_edges':sum(len(c)-1 for c in chains),
        'max_chain_length':int(max(clen)),
        'chain_length_cv':float(np.std(clen)/np.mean(clen)),
        'n_multi_logical_chains':sum(len(c)>1 for c in chains),
        'quotient_unique_edges_rebuilt':len(pairs),
        'max_weighted_degree_rebuilt':max(wdeg),
        'weighted_degree_cv_rebuilt':float(np.std(wdeg)/np.mean(wdeg)) if np.mean(wdeg) else 0.,
        'max_unique_partners':max(deg),
        'mean_unique_partners':float(np.mean(deg)),
        'max_pair_interaction_weight':max(pairs.values()) if pairs else 0,
        # New: partner switching/fragmentation and scheduling pressure.
        'partner_switch_rate':total_switch/switch_den if switch_den else 0.0,
        'partner_switches_total':total_switch,
        'max_chain_partner_switches':max(switches),
        'mean_pair_run_length':(2*n_2q)/segs if segs else 0.0,
        'critical_2q_count':two_critical,
        'critical_2q_fraction':two_critical/n_2q if n_2q else 0.,
        'critical_reuse_boundaries':critical_reuses,
        'critical_reuse_fraction':critical_reuses/max(1,sum(len(c)-1 for c in chains)),
        'pre_routing_asap_gate_depth':gate_depth,
        'max_window_2q_chain_fraction':t['max_window_2q_chain_fraction'],
        'mean_window_2q_chain_fraction':t['mean_window_2q_chain_fraction'],
        'max_2q_events_in_plan_bin':t['max_2q_events_in_plan_bin'],
        'temporal_bin_2q_cv':t['temporal_bin_2q_cv'],
    }
    per_chain=[]
    for i,chain in enumerate(chains):
        s=seqs[i]
        per_chain.append({
           'chain_index':i,'chain':str(chain),'n_logical_qubits':len(chain),
           'two_qubit_incident_gates':wdeg[i],'n_partner_chains':deg[i],
           'partner_switches':switches[i],
           'partner_switch_rate':switches[i]/(len(s)-1) if len(s)>1 else 0.,
           'logical_2q_loads':str(tuple((q,own_degree[q]) for q in chain)),
           'partner_sequence':str(tuple(s)),
        })
    return f,per_chain,pairs



def compare_chain_structures(best, worst, per_chain_rows):
    """Align unlabeled chain IDs by maximum logical-qubit overlap before comparison.

    Chain index itself has no inherent physical meaning, so comparing index i
    between two plans without relabeling would be misleading.
    """
    a=parse_chains(best.plan_signature); b=parse_chains(worst.plan_signature)
    if len(a)!=len(b): raise ValueError('Chain-width mismatch')
    overlaps=np.array([[len(set(x)&set(y)) for y in b] for x in a],dtype=int)
    ra,cb=linear_sum_assignment(-overlaps)
    comparison=[]
    for i,j in zip(ra,cb):
        ca=next(x for x in per_chain_rows if x['plan_id']==best.plan_id and x['chain_index']==i)
        cw=next(x for x in per_chain_rows if x['plan_id']==worst.plan_id and x['chain_index']==j)
        comparison.append({
          'circuit':best.circuit,'best_plan':best.plan_id,'worst_plan':worst.plan_id,
          'best_chain_idx':i,'worst_aligned_chain_idx':j,'qubit_overlap':int(overlaps[i,j]),
          'best_chain':str(a[i]),'worst_chain':str(b[j]),
          'best_chain_2q_load':ca['two_qubit_incident_gates'],
          'worst_chain_2q_load':cw['two_qubit_incident_gates'],
          'best_chain_partner_switches':ca['partner_switches'],
          'worst_chain_partner_switches':cw['partner_switches'],
        })
    ea={(x,y) for c in a for x,y in zip(c,c[1:])}
    eb={(x,y) for c in b for x,y in zip(c,c[1:])}
    edge_rows=[{'circuit':best.circuit,'best_plan':best.plan_id,'worst_plan':worst.plan_id,
                'reuse_edge':str(p),'in_best':p in ea,'in_worst':p in eb}
               for p in sorted(ea|eb)]
    return comparison,edge_rows


def existing_five_circuit_summary(d):
    """Existing Experiment 2 features permit a five-circuit comparison,
    even if only a subset of original QASMs is uploaded for deep forensics.
    """
    fresh=d['fresh'];features=d['features']
    cols=['max_weighted_degree','weighted_degree_cv','quotient_unique_edges',
          'max_edge_weight','edge_weight_cv','max_edge_share','normalized_edge_entropy']
    a=fresh.merge(features[['circuit','plan_id']+cols],on=['circuit','plan_id'],validate='one_to_one')
    rows=[]
    for circuit,g in a.groupby('circuit'):
        v=g.sort_values(['fresh_mean_routed_depth','plan_id'])
        best,worst=v.iloc[0],v.iloc[-1]
        row={'circuit':circuit,'n_plans':len(g),'width':int(best.width),'best_plan':best.plan_id,'worst_plan':worst.plan_id,
             'best_depth':float(best.fresh_mean_routed_depth),'worst_depth':float(worst.fresh_mean_routed_depth),
             'depth_gap':float(worst.fresh_mean_routed_depth-best.fresh_mean_routed_depth),
             'best_swaps':float(best.fresh_mean_swaps),'worst_swaps':float(worst.fresh_mean_swaps),
             'best_normalized_logical_depth':int(best.normalized_logical_depth),
             'worst_normalized_logical_depth':int(worst.normalized_logical_depth)}
        for col in cols:
            row[f'best_{col}']=float(best[col]);row[f'worst_{col}']=float(worst[col])
            row[f'worst_minus_best_{col}']=float(worst[col]-best[col])
        rows.append(row)
    return pd.DataFrame(rows)


def controlled_comparisons(per_plan,raw):
    rows=[]
    from itertools import combinations
    for cir,g in per_plan.groupby('circuit'):
        for a,b in combinations(g.to_dict('records'),2):
            # Same actual width automatically, but also match normalized logical depth.
            if int(a['normalized_logical_depth'])!=int(b['normalized_logical_depth']):continue
            best,worst=sorted([a,b],key=lambda z:z['fresh_mean_routed_depth'])
            pair=paired_seed_diff(raw,cir,best['plan_id'],worst['plan_id'],'routed_depth')
            rows.append({'circuit':cir,'equal_normalized_logical_depth':True,
                    'same_max_weighted_degree':np.isclose(best['max_weighted_degree_rebuilt'],worst['max_weighted_degree_rebuilt']),
                    'best_id':best['plan_id'],'worst_id':worst['plan_id'],
                    'logical_depth':int(best['normalized_logical_depth']),
                    'hotspot_best':best['max_weighted_degree_rebuilt'],'hotspot_worst':worst['max_weighted_degree_rebuilt'],
                    'weighted_degree_cv_best':best['weighted_degree_cv_rebuilt'],
                    'weighted_degree_cv_worst':worst['weighted_degree_cv_rebuilt'],
                    'depth_best':best['fresh_mean_routed_depth'],'depth_worst':worst['fresh_mean_routed_depth'],
                    'mean_depth_gap':worst['fresh_mean_routed_depth']-best['fresh_mean_routed_depth'],
                    'matched_seed_gap_95ci_low':pair['approx_95pct_CI_low'],
                    'matched_seed_gap_95ci_high':pair['approx_95pct_CI_high'],
                    'partner_switch_rate_best':best['partner_switch_rate'],
                    'partner_switch_rate_worst':worst['partner_switch_rate'],
                    'mean_window_2q_chain_fraction_best':best['mean_window_2q_chain_fraction'],
                    'mean_window_2q_chain_fraction_worst':worst['mean_window_2q_chain_fraction']})
    return pd.DataFrame(rows).sort_values('mean_depth_gap',ascending=False) if rows else pd.DataFrame()

def read_inputs(data_dir):
    wanted={
        'fresh':'experiment1_fresh30_plan_summary.csv',
        'raw':'experiment1_fresh30_raw.csv',
        'old':'experiment1_plan_summary.csv',
        'features':'experiment2_plan_features.csv'
    }
    out={}
    for k,name in wanted.items():
        matches=list(data_dir.rglob(name))
        if len(matches)!=1:raise FileNotFoundError(f'Expected exactly one {name} under {data_dir}; found {len(matches)}')
        out[k]=pd.read_csv(matches[0])
    return out


def audit_data(d):
    keys=['circuit','plan_id'];ident=['gidnet_seed','plan_signature','width','normalized_logical_depth']
    for name,df in d.items():
        for k in keys+ident:
            if k not in df.columns:raise ValueError(f'Missing {k} in {name}')
        if name!='raw' and df.duplicated(keys).any(): raise ValueError(f'Duplicated plan ids in {name}')
    fresh=d['fresh'];old=d['old'];features=d['features'];raw=d['raw']
    if len(fresh)!=len(old) or len(fresh)!=len(features):raise ValueError('Plan counts differ')
    for name,df in [('old',old),('features',features)]:
        joined=fresh[keys+ident].merge(df[keys+ident],on=keys,suffixes=('_a','_b'),validate='one_to_one')
        if len(joined)!=len(fresh):raise ValueError(f'Missing plan rows in {name}')
        for k in ident:
            if not (joined[f'{k}_a'].astype(str)==joined[f'{k}_b'].astype(str)).all():
                raise ValueError(f'Plan identity mismatch {name}: {k}')
    joined=raw[keys+ident+['routing_seed','routed_depth','swap_count']].merge(fresh[keys+ident+['fresh_mean_routed_depth','fresh_mean_swaps']],on=keys,suffixes=('_a','_b'),validate='many_to_one')
    if len(joined)!=len(raw):raise ValueError('Routing data missing plans')
    for k in ident:
        if not (joined[f'{k}_a'].astype(str)==joined[f'{k}_b'].astype(str)).all():
            raise ValueError(f'Routing raw identity mismatch: {k}')
    agg=raw.groupby(keys,as_index=False).agg(mean_depth=('routed_depth','mean'),mean_swap=('swap_count','mean'))
    check=agg.merge(fresh[keys+['fresh_mean_routed_depth','fresh_mean_swaps']],on=keys,validate='one_to_one')
    if not np.allclose(check.mean_depth,check.fresh_mean_routed_depth,atol=1e-8):raise ValueError('Fresh depth mean mismatch')
    if not np.allclose(check.mean_swap,check.fresh_mean_swaps,atol=1e-8):raise ValueError('Fresh SWAP mean mismatch')


def normalized_z(x):
    v=np.asarray(x,dtype=float)
    return np.zeros_like(v) if len(v)<2 or np.std(v)==0 else (v-np.mean(v))/np.std(v)


def make_figures(out,df,circuit,all_feature_names):
    fig_dir=out/'figures';fig_dir.mkdir(exist_ok=True)
    # Each figure contains exactly one independent plot.
    ranked=df.sort_values('fresh_mean_routed_depth')
    fig,ax=plt.subplots(figsize=(9,5))
    ax.bar(ranked['plan_id'],ranked['fresh_mean_routed_depth'])
    ax.set(title=f'{circuit} — independent 30-seed routed depth',xlabel='Reuse plan (best → worst)',ylabel='Mean routed depth')
    ax.tick_params(axis='x',rotation=45)
    fig.tight_layout();fig.savefig(fig_dir/(circuit.replace('.qasm','')+'_depth_ranking.png'),dpi=170);plt.close(fig)
    for fname in ['partner_switch_rate','critical_reuse_boundaries','mean_window_2q_chain_fraction','max_weighted_degree_rebuilt']:
        if fname not in df:continue
        fig,ax=plt.subplots(figsize=(6.8,5))
        ax.scatter(df[fname],df['fresh_mean_routed_depth'])
        for r in df.itertuples(index=False):
            ax.annotate(r.plan_id,(getattr(r,fname),r.fresh_mean_routed_depth),xytext=(4,3),textcoords='offset points',fontsize=8)
        ax.set(xlabel=fname.replace('_',' '),ylabel='30-seed mean routed depth',title=f'{circuit}: pre-routing feature vs physical depth (exploratory)')
        fig.tight_layout();fig.savefig(fig_dir/(circuit.replace('.qasm','')+'_'+fname+'.png'),dpi=170);plt.close(fig)
    xfeatures=[x for x in all_feature_names if x in df and df[x].nunique()>1]
    if xfeatures:
        z=np.vstack([normalized_z(df[f].to_numpy()) for f in xfeatures]).T
        order=np.argsort(df['fresh_mean_routed_depth'].to_numpy())
        fig,ax=plt.subplots(figsize=(max(8, len(xfeatures)*0.75), max(4.5,len(df)*.42)))
        im=ax.imshow(z[order,:],aspect='auto',cmap='coolwarm',vmin=-2,vmax=2)
        ax.set_xticks(range(len(xfeatures)),[f.replace('_','\n') for f in xfeatures],rotation=75,ha='right',fontsize=8)
        ax.set_yticks(range(len(df)),df.iloc[order].plan_id.tolist())
        ax.set(title=f'{circuit}: feature z-scores across same-width plans (not causal)',ylabel='Plans ranked by routed depth')
        fig.colorbar(im,ax=ax,label='Within-circuit feature z-score',fraction=0.03)
        fig.tight_layout();fig.savefig(fig_dir/(circuit.replace('.qasm','')+'_feature_heatmap.png'),dpi=160);plt.close(fig)


def paired_seed_diff(raw,circuit,best_id,worst_id,metric):
    a=raw[(raw.circuit==circuit)&(raw.plan_id==best_id)][['routing_seed',metric]].rename(columns={metric:'best'})
    b=raw[(raw.circuit==circuit)&(raw.plan_id==worst_id)][['routing_seed',metric]].rename(columns={metric:'worst'})
    z=a.merge(b,on='routing_seed',validate='one_to_one')
    d=(z.worst-z.best).to_numpy(dtype=float)
    if not len(d):raise ValueError('No common routing seeds')
    se=np.std(d,ddof=1)/np.sqrt(len(d)) if len(d)>1 else float('nan')
    ci=float(t_distribution.ppf(.975,len(d)-1)*se) if len(d)>1 else float('nan')
    return {'matched_seeds':len(d),'mean_worst_minus_best':float(np.mean(d)),'sample_sd':float(np.std(d,ddof=1)),
            'approx_95pct_CI_low':float(np.mean(d)-ci),'approx_95pct_CI_high':float(np.mean(d)+ci),
            'seed_wise_worst_greater_count':int(np.count_nonzero(d>0))}


def write_report(out,comp,feature_diffs,corrs,warnings):
    s=['# Fixed-width reuse-plan structural forensics — exploratory pilot','',
       'No new routing and no new plan generation were performed. A comparison here is **descriptive and hypothesis-generating**, not proof that a feature causes physical cost.', '']
    for row in comp.itertuples(index=False):
        s += [f'## {row.circuit}',
              f'- Matching reuse plans: {row.n_plans}; active width: {row.width}.',
              f'- Best by 30-seed routed depth: **{row.best_plan_id}** ({row.best_mean_depth:.3f}); worst: **{row.worst_plan_id}** ({row.worst_mean_depth:.3f}).',
              f'- Worst − best: **{row.depth_gap:.3f}** depth, **{row.swaps_gap:.3f}** SWAP. SWAP-best/worst may be different plan IDs.',
              f'- Matched routing-seed depth gap (approx. 95% t interval, descriptive): [{row.paired_depth_ci_low:.3f}, {row.paired_depth_ci_high:.3f}].',
              '- Differences from feature data and newly derived pre-routing features are in `feature_differences.csv` and `per_plan_features.csv`.', '']
        ds=feature_diffs[(feature_diffs.circuit==row.circuit)&feature_diffs.feature.isin([
              'max_weighted_degree_rebuilt','weighted_degree_cv_rebuilt','partner_switch_rate',
              'critical_reuse_boundaries','pre_routing_asap_gate_depth','mean_window_2q_chain_fraction'])]
        s+=['| Pre-routing feature | Best | Worst | Worst − Best |','|---|---:|---:|---:|']
        for d in ds.itertuples(index=False):
            s.append(f'| {d.feature} | {d.best_value:.4g} | {d.worst_value:.4g} | {d.worst_minus_best:+.4g} |')
        s+=['', '### Exploratory across-plan signals',
            'The feature-depth correlations below are within this circuit, across currently available plans only. They are *not independently validated*, and multiple features were inspected.',
            '']
        c=corrs[corrs.circuit==row.circuit].sort_values('spearman_rho',ascending=False)
        s+=['| Feature | Spearman with routed depth |', '|---|---:|']
        for x in c.itertuples(index=False):s.append(f'| {x.feature} | {x.spearman_rho:+.3f} |')
        s+=['']
    s+=['## Interpretation rules','',
        '1. Existing metrics (hotspot, weighted degree CV) are used as sanity checks, not claimed to be novel.',
        '2. New candidates include partner switching/interaction run fragmentation, critical reuse boundaries, and temporal bin pressure. A strong association would be a *hypothesis*, not causality.',
        '3. Selecting best and worst on routing outcomes creates selection bias; require confirmation on other circuits or held-out plans/seeds.',
        '4. Plan-dependent DAG layering assumes one gate-time unit per operation and ignores physical layout/routing; it is a deliberately cheap *pre-routing proxy*, not a simulation of real execution time.',
        '5. The QASM2 parser supports static gate-only single-qreg circuits. It stops rather than silently interpreting reset/measurement/classical conditions. Plan semantics and reset correctness must be tested elsewhere.',
        '6. Absence of other matching original QASMs is a real data limit; do not claim cross-circuit structural validation when only system9 is matched.', '']
    if warnings: s += ['## Skipped QASMs / warnings','']+['- '+w for w in warnings]+['']
    (out/'REPORT.md').write_text('\n'.join(s),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,required=True,help='Folder containing provided CSV files (recursive lookup).')
    ap.add_argument('--qasm-dir',type=Path,default=None,help='Folder containing original QASM files (recursive lookup). Default: data-dir')
    ap.add_argument('--out-dir',type=Path,default=Path('structural_forensics_results'))
    ap.add_argument('--circuits',nargs='*',default=[],help='Optional filenames; default: all matched QASM circuits.')
    args=ap.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    d=read_inputs(args.data_dir);audit_data(d)
    available_qasms={p.name:p for p in (args.qasm_dir or args.data_dir).rglob('*.qasm')}
    target=set(d['fresh'].circuit.unique())
    if args.circuits:target &= set(args.circuits)
    matched=sorted(target & set(available_qasms))
    missing=sorted(target-set(available_qasms))
    print('Matched circuits:',matched)
    print('Missing QASM for:',missing)
    if not matched:raise ValueError('No original QASM matches the circuit names in plan CSVs. Cannot reconstruct plan structure.')
    rows=[];chain_rows=[];all_pairs=[];warnings=[]; comps=[]; diffs=[];correlations=[];aligned_chains=[];edge_diffs=[]
    for cir in matched:
        qasm=available_qasms[cir]
        try:n_declared,ops=parse_qasm(qasm)
        except Exception as e:
            warnings.append(f'{cir}: {e}');continue
        f=d['fresh'][d['fresh'].circuit==cir].copy()
        fe=d['features'][d['features'].circuit==cir].set_index('plan_id')
        if f.width.nunique()!=1:raise ValueError(f'Mixed widths in same circuit {cir}')
        for row in f.itertuples(index=False):
            chains=parse_chains(row.plan_signature)
            if not set(q for c in chains for q in c)==set(q for op in ops for q in op.qubits):
                raise ValueError(f'{cir}/{row.plan_id}: active gate-qubit set differs from plan signature')
            fvals,per_chain,pairs=compute_features(ops,chains,n_declared,row.normalized_logical_depth)
            # Hard sanity check: reconstructed quotient graph must reproduce
            # independently computed Experiment 2 features.
            ref=fe.loc[row.plan_id]
            for local,remote,tol in [('max_weighted_degree_rebuilt','max_weighted_degree',1e-7),
                         ('weighted_degree_cv_rebuilt','weighted_degree_cv',1e-7),
                         ('quotient_unique_edges_rebuilt','quotient_unique_edges',1e-7)]:
                if not np.isclose(fvals[local],float(ref[remote]),atol=tol):
                    raise AssertionError(f'{cir} {row.plan_id}: mismatch {local}={fvals[local]} vs Exp2 {remote}={ref[remote]}')
            r={'circuit':cir,'plan_id':row.plan_id,'plan_signature':row.plan_signature,'gidnet_seed':row.gidnet_seed,
               'width':row.width,'normalized_logical_depth':row.normalized_logical_depth,
               'fresh_mean_routed_depth':row.fresh_mean_routed_depth,
               'fresh_mean_swaps':row.fresh_mean_swaps,**fvals}
            rows.append(r)
            for c in per_chain:chain_rows.append({'circuit':cir,'plan_id':row.plan_id,**c})
            for (a,b),w in pairs.items():all_pairs.append({'circuit':cir,'plan_id':row.plan_id,'chain_a':a,'chain_b':b,'n_2q_gates':w})
        joined=pd.DataFrame([r for r in rows if r['circuit']==cir])
        by=joined.sort_values(['fresh_mean_routed_depth','plan_id'])
        best=by.iloc[0];worst=by.iloc[-1]
        # Seed matching checks conditional stability of the observed two plans.
        aligned, edge_info=compare_chain_structures(best,worst,[x for x in chain_rows if x['circuit']==cir])
        aligned_chains.extend(aligned);edge_diffs.extend(edge_info)
        depthpair=paired_seed_diff(d['raw'],cir,best.plan_id,worst.plan_id,'routed_depth')
        swappair=paired_seed_diff(d['raw'],cir,best.plan_id,worst.plan_id,'swap_count')
        comps.append({'circuit':cir,'n_plans':len(joined),'width':int(best.width),
             'best_plan_id':best.plan_id,'worst_plan_id':worst.plan_id,
             'best_plan_signature':best.plan_signature,'worst_plan_signature':worst.plan_signature,
             'best_mean_depth':best.fresh_mean_routed_depth,'worst_mean_depth':worst.fresh_mean_routed_depth,
             'depth_gap':worst.fresh_mean_routed_depth-best.fresh_mean_routed_depth,
             'best_mean_swaps':best.fresh_mean_swaps,'worst_mean_swaps':worst.fresh_mean_swaps,
             'swaps_gap':worst.fresh_mean_swaps-best.fresh_mean_swaps,
             'min_swap_plan_id':joined.sort_values(['fresh_mean_swaps','plan_id']).iloc[0].plan_id,
             'max_swap_plan_id':joined.sort_values(['fresh_mean_swaps','plan_id']).iloc[-1].plan_id,
             'paired_n_seeds':depthpair['matched_seeds'],
             'paired_depth_ci_low':depthpair['approx_95pct_CI_low'],
             'paired_depth_ci_high':depthpair['approx_95pct_CI_high'],
             'paired_swaps_ci_low':swappair['approx_95pct_CI_low'],
             'paired_swaps_ci_high':swappair['approx_95pct_CI_high'],
             'depth_seed_worst_greater_count':depthpair['seed_wise_worst_greater_count']})
        names=[k for k in fvals if k not in ['active_width','full_declared_width_estimate','n_operations','n_active_logical_qubits','n_2q_gates','n_reuse_edges']]
        for name in names:
            v1=float(best[name]);v2=float(worst[name]);
            diffs.append({'circuit':cir,'feature':name,'best_plan_id':best.plan_id,'worst_plan_id':worst.plan_id,
                          'best_value':v1,'worst_value':v2,'worst_minus_best':v2-v1,'feature_range_across_plans':float(joined[name].max()-joined[name].min())})
            if joined[name].nunique()>1 and len(joined)>3:
                rho,pval=spearmanr(joined[name].to_numpy(dtype=float),joined['fresh_mean_routed_depth'].to_numpy(dtype=float))
                correlations.append({'circuit':cir,'feature':name,'n_plans':len(joined),'spearman_rho':float(rho),'exploratory_p_unadjusted':float(pval)})
        make_figures(args.out_dir,joined,cir,['max_weighted_degree_rebuilt','weighted_degree_cv_rebuilt',
                     'partner_switch_rate','mean_pair_run_length','critical_reuse_boundaries',
                     'critical_2q_fraction','mean_window_2q_chain_fraction','pre_routing_asap_gate_depth'])
        print(f'{cir}: best {best.plan_id} mean depth {best.fresh_mean_routed_depth:.2f}; worst {worst.plan_id} {worst.fresh_mean_routed_depth:.2f}')
    if not rows:raise RuntimeError('No analysable QASM circuits')
    out=args.out_dir
    df=pd.DataFrame(rows); comp=pd.DataFrame(comps); df_d=pd.DataFrame(diffs);df_c=pd.DataFrame(correlations)
    df.sort_values(['circuit','fresh_mean_routed_depth']).to_csv(out/'per_plan_features.csv',index=False)
    pd.DataFrame(chain_rows).to_csv(out/'per_chain_diagnostics.csv',index=False)
    pd.DataFrame(all_pairs).to_csv(out/'quotient_interaction_edges.csv',index=False)
    pd.DataFrame(aligned_chains).to_csv(out/'best_worst_aligned_chains.csv',index=False)
    pd.DataFrame(edge_diffs).to_csv(out/'best_worst_reuse_boundary_changes.csv',index=False)
    existing_five_circuit_summary(d).to_csv(out/'all_five_existing_feature_comparison.csv',index=False)
    controls=controlled_comparisons(df,d['raw'])
    if not controls.empty:controls.to_csv(out/'equal_logical_depth_controlled_pairs.csv',index=False)
    comp.to_csv(out/'best_worst_comparison.csv',index=False)
    df_d.to_csv(out/'feature_differences.csv',index=False)
    df_c.to_csv(out/'within_circuit_exploratory_correlations.csv',index=False)
    coverage={'plan_circuits_in_CSV':sorted(set(d['fresh'].circuit)),
              'matched_original_QASM':matched, 'missing_original_QASM':missing,
              'analysed_circuits':sorted(set(df.circuit)),'analysed_plans':len(df),
              'skipped_warnings':warnings,'no_new_routing':True,'no_new_plan_generation':True}
    (out/'coverage.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2),encoding='utf-8')
    write_report(out,comp,df_d,df_c,warnings)
    with (out/'REPORT.md').open('a',encoding='utf-8') as fp:
        fp.write('\n## Existing-feature five-circuit comparison (not deep QASM reconstruction)\n\n')
        fp.write('The best/worst comparisons for **all five** original benchmarks use existing Experiment 2 features, while novel gate/chain reconstruction is limited to matched QASM files.\n\n')
        x=existing_five_circuit_summary(d)
        fp.write('| Circuit | Best / Worst | Routed depth best → worst | MaxWD best → worst |\n|---|---|---:|---:|\n')
        for row in x.itertuples(index=False):
            fp.write(f'| {row.circuit} | {row.best_plan} / {row.worst_plan} | {row.best_depth:.2f} → {row.worst_depth:.2f} | {row.best_max_weighted_degree:.0f} → {row.worst_max_weighted_degree:.0f} |\n')
        if not controls.empty:
            fp.write('\n## Matched normalized-logical-depth pairs in reconstructed circuits\n\n')
            for cc in controls.head(8).itertuples(index=False):
                fp.write(f'- {cc.circuit}: {cc.best_id} vs {cc.worst_id}, identical logical depth {cc.logical_depth}; depth gap {cc.mean_depth_gap:.3f}; same hotspot? {bool(cc.same_max_weighted_degree)}.\n')
        fp.write('\n**Important:** Five-circuit best/worst trends may be driven by the selection of extremes. They do not establish within-circuit predictive reliability for all plans.\n')
    print('Wrote results to',out.resolve())

if __name__=='__main__':main()
