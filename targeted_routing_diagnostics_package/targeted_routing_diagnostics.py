#!/usr/bin/env python3
"""Stage B: paired SABRE routing diagnostics on FROZEN fixed-width reuse plans.

Qiskit 0.45/0.46 intended. Requires locally PATCHED GidNET (reset boundary bugfix).
No GidNET plan generation. Exits loudly for QASM/CSV/width/plan/depth mismatch.
"""
from __future__ import annotations
import argparse, ast, csv, hashlib, inspect, itertools, json, sys
import math
from collections import Counter, defaultdict
from pathlib import Path

PAIR_SPECS = [('system9.qasm','P5','P4'),('sym9_146.qasm','P8','P4')]
RUN_COLS = ['circuit','mode','seed','plan_id','role','width','source_depth','normalized_depth',
            'expected_normalized_depth','routed_depth','swap_count','cx_count','measurements','conditional_x',
            'routed_overhead','swap_critical_count','cx_critical_count','swap_layers',
            'max_swaps_in_layer','max_consecutive_swap_layers','total_2q_layers','asap_depth',
            'initial_layout_json','routed_qpy','circuit_fingerprint']


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def parse_chains(value):
    chains=ast.literal_eval(value)
    if not isinstance(chains,(list,tuple)) or not chains:raise ValueError('Empty/invalid chains')
    clean=[];seen=set()
    for chain in chains:
        if not isinstance(chain,(list,tuple)) or not chain:raise ValueError('Invalid reuse chain')
        row=[]
        for q in chain:
            if isinstance(q,bool) or not isinstance(q,int) or q<0 or q in seen:
                raise ValueError('Duplicate/invalid logical qubit in plan')
            row.append(q);seen.add(q)
        clean.append(tuple(row))
    return tuple(clean)


def match_chains(a,b):
    """Unlabelled chain matching: bijection best-index -> worst-index, max overlap."""
    if len(a)!=len(b):raise ValueError('Not fixed width')
    width=len(a); scores=[[len(set(x)&set(y)) for y in b] for x in a]
    # Widths in selected benchmarks are <=7; exact enumeration avoids SciPy dependence.
    if width>9:raise ValueError('This small-pair audit expects <=9 chains; use Hungarian for larger width')
    best=-1;win=[]
    for perm in itertools.permutations(range(width)):
        score=sum(scores[i][j] for i,j in enumerate(perm))
        if score>best:best=score;win=[perm]
        elif score==best:win.append(perm)
    if len(win)!=1:
        raise ValueError(f'Chain correspondence is ambiguous ({len(win)} equal-best assignments). '
                         'Do not claim a common physical layout without manually fixing the correspondence.')
    return list(win[0]),best


def reposition_layout(anchor_physical, match_better_to_worse, anchor_role):
    """Map physical sites to aligned chains in the other plan, no width change."""
    n=len(anchor_physical)
    if sorted(match_better_to_worse)!=list(range(n)):raise ValueError('Not a bijection')
    other=[None]*n
    if anchor_role=='better':
        for better_i,worse_i in enumerate(match_better_to_worse):other[worse_i]=anchor_physical[better_i]
    elif anchor_role=='worse':
        for better_i,worse_i in enumerate(match_better_to_worse):other[better_i]=anchor_physical[worse_i]
    else:raise ValueError(anchor_role)
    assert sorted(other)==sorted(anchor_physical)
    return other


def collect_csv(root, name):
    matches=list(Path(root).rglob(name))
    if len(matches)!=1:raise FileNotFoundError(f'Expected exactly one {name} in {root}, got {len(matches)}: {matches}')
    with matches[0].open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    return rows,matches[0]


def read_input_plan_data(root):
    fresh,path_f=collect_csv(root,'experiment1_fresh30_plan_summary.csv')
    raw,path_r=collect_csv(root,'experiment1_fresh30_raw.csv')
    out={};old={}
    for circuit,better,worse in PAIR_SPECS:
        members=[r for r in fresh if r['circuit']==circuit and r['plan_id'] in (better,worse)]
        if len(members)!=2 or len(set(r['plan_id'] for r in members))!=2:raise ValueError(f'Missing/duplicate plan in {circuit}')
        out[circuit]={r['plan_id']:r for r in members}
        assert int(out[circuit][better]['width'])==int(out[circuit][worse]['width'])
        for plan_id in (better,worse):
            row=out[circuit][plan_id]
            assert str(parse_chains(row['plan_signature']))==row['plan_signature']
            previous=[r for r in raw if r['circuit']==circuit and r['plan_id']==plan_id]
            seeds=[int(r['routing_seed']) for r in previous]
            if len(seeds)!=30 or len(set(seeds))!=30:
                raise ValueError(f'Expected 30 archived distinct seeds for {circuit}:{plan_id}')
            if any(r['plan_signature']!=row['plan_signature'] for r in previous):
                raise ValueError(f'Signature differs between archived routing and plan summary: {circuit}:{plan_id}')
            old[(circuit,plan_id)]={int(r['routing_seed']):r for r in previous}
    return out,old,[str(path_f),str(path_r)]


def check_quotient_features(original,chains,ref,circuit,plan_id):
    """Reconstruct three Experiment-2 quotient metrics from actual input QASM."""
    owner={q:idx for idx,c in enumerate(chains) for q in c}
    pair_weight=Counter();weighted=[0]*len(chains)
    for op,qargs,_ in op_items(original):
        if len(qargs)!=2:continue
        qa,qb=(original.find_bit(q).index for q in qargs)
        if qa not in owner or qb not in owner:raise ValueError('Missing active original logical qubit')
        a,b=owner[qa],owner[qb]
        if a==b:raise ValueError(f'Simultaneously interacting logical qubits mapped to same chain: {qa}, {qb}')
        pair_weight[tuple(sorted((a,b)))]+=1
        weighted[a]+=1;weighted[b]+=1
    mean=sum(weighted)/len(weighted)
    cv=(math.sqrt(sum((v-mean)**2 for v in weighted)/len(weighted))/mean) if mean else 0.0
    actual={'max_weighted_degree':max(weighted,default=0),'weighted_degree_cv':cv,
            'quotient_unique_edges':len(pair_weight)}
    for feature,value in actual.items():
        if not math.isclose(float(ref[feature]),float(value),rel_tol=0,abs_tol=1e-7):
            raise ValueError(f'Experiment 2 QASM/plan audit failure {circuit}:{plan_id} {feature} '
                             f'expected {ref[feature]}, got {value}. Check QASM provenance.')
    return actual


def find_qasm(root, filename):
    matches=list(Path(root).rglob(filename))
    if len(matches)!=1:raise FileNotFoundError(f'Expected one original QASM named {filename} under {root}, found {matches}')
    return matches[0]


def op_items(qc):
    for item in qc.data:
        if hasattr(item,'operation'): yield item.operation,list(item.qubits),list(item.clbits)
        else:yield item[0],list(item[1]),list(item[2])


def lower_active_resets(qc):
    """Same measure+conditional-X convention as Experiment 1 (static QASM)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ClassicalRegister
    import copy
    lowered=QuantumCircuit(*qc.qregs,*qc.cregs,name=qc.name)
    lowered.global_phase=qc.global_phase
    lowered.metadata=copy.deepcopy(qc.metadata)
    last_measure={}; names={reg.name for reg in lowered.cregs}; ix=0
    for op,qargs,cargs in op_items(qc):
        if op.name=='reset':
            if len(qargs)!=1 or getattr(op,'condition',None) is not None:raise ValueError('Unsupported non-single/non-unconditional reset')
            q=qargs[0];c=last_measure.get(q)
            if c is None:
                while f'reset_aux_{ix}' in names:ix+=1
                reg=ClassicalRegister(1,f'reset_aux_{ix}');names.add(reg.name);ix+=1
                lowered.add_register(reg);c=reg[0];lowered.measure(q,c)
            lowered.x(q).c_if(c,1)
            last_measure.pop(q,None)
        else:
            lowered.append(op,qargs,cargs)
            for q in qargs:last_measure.pop(q,None)
            for c in cargs:
                for prior in [q for q,bit in last_measure.items() if bit==c]:last_measure.pop(prior,None)
            if op.name=='measure' and len(qargs)==len(cargs)==1 and getattr(op,'condition',None) is None:
                last_measure[qargs[0]]=cargs[0]
    return lowered


def conditional_x(qc):
    return sum(op.name=='x' and getattr(op,'condition',None) is not None for op,_,_ in op_items(qc))


def circuit_fingerprint(qc):
    data=[]
    for op,qargs,cargs in op_items(qc):
        cond=getattr(op,'condition',None)
        data.append((op.name,tuple(qc.find_bit(q).index for q in qargs),
                     tuple(qc.find_bit(c).index for c in cargs),repr(cond),tuple(map(str,op.params))))
    return hashlib.sha256(repr((qc.num_qubits,qc.num_clbits,data)).encode()).hexdigest()


def compile_fixed_plan(original,chains,expected_depth):
    from gidnet.qubitreuse import GidNET
    from reuse_validity import ReuseValidityChecker
    checker=ReuseValidityChecker(original)
    validity=checker.validate(chains,target_width=len(chains))
    if not validity.is_valid:raise ValueError(f'Static checker rejects plan: {validity.reason}: {validity.details}')
    inst=GidNET(original)
    # Inject exact frozen chain assignment. GidNET *compilation*, not GidNET *plan search*.
    inst.compute_optimized_reuse_sequences=lambda iterations: [list(c) for c in chains]
    native=inst.compile_to_dynamic_circuit(iterations=1,draw=False)
    if native is None or native.num_qubits!=len(chains):raise ValueError('Fixed-plan compilation returned wrong width')
    need_reset=sum(len(c)-1 for c in chains)
    if int(native.count_ops().get('reset',0))!=need_reset:
        raise ValueError(f'Expected {need_reset} reset instructions, got {native.count_ops()} — local GidNET fix missing?')
    norm=lower_active_resets(native)
    if norm.num_qubits!=len(chains) or norm.count_ops().get('reset',0)!=0:
        raise ValueError('Normalization changed width or did not remove resets')
    if int(norm.count_ops().get('measure',0))!=need_reset or conditional_x(norm)!=need_reset:
        # This benchmark QASM contains no measurements initially.
        raise ValueError('Reset lowering counts mismatch; check GidNET patch and input QASM')
    depth=norm.depth()
    if int(depth)!=int(expected_depth):
        raise ValueError(f'Normalized logical depth {depth} != archived {expected_depth}. '
                         'STOP: cannot compare with Experiment 1 under inconsistent compilation/version.')
    return norm, native,inst


def physical_layout_from_route(routed,original):
    """Read actual initial_layout for original virtual qubits, never inferred from final mapping."""
    layout=getattr(routed,'layout',None)
    first=getattr(layout,'initial_layout',None)
    if first is None:raise ValueError('No transpiler initial_layout; cannot freeze physical layout')
    mapping=[]
    for q in original.qubits:
        try:site=first[q]
        except Exception as exc:
            raise ValueError('Could not trace input virtual qubit in initial_layout; abort fixed-layout comparison') from exc
        if not isinstance(site,int):raise ValueError(f'Unexpected layout site {site}')
        mapping.append(int(site))
    if len(set(mapping))!=len(mapping):raise ValueError('Layout is not one-to-one')
    return mapping


def graph_layers(qc):
    """Actual transpiled DAG ASAP layers + critical op membership (unit operation depth)."""
    from qiskit.converters import circuit_to_dag
    from qiskit.dagcircuit import DAGOpNode
    dag=circuit_to_dag(qc)
    nodes=list(dag.topological_op_nodes())
    byid={n._node_id:n for n in nodes}
    lv={};succ=defaultdict(list)
    for n in nodes:
        pred=[p._node_id for p in dag.predecessors(n) if isinstance(p,DAGOpNode)]
        lv[n._node_id]=1+max([lv[k] for k in pred],default=0)
        for p in pred:succ[p].append(n._node_id)
    depth=max(lv.values(),default=0)
    tail={}
    for n in reversed(nodes):
        uid=n._node_id
        tail[uid]=1+max([tail[x] for x in succ[uid]],default=0)
    event_rows=[];perlayer=defaultdict(Counter);byedge=defaultdict(Counter)
    for order,n in enumerate(nodes):
        ident=n._node_id;op=n.op;layer=lv[ident];crit=layer+tail[ident]-1==depth
        qs=[qc.find_bit(q).index for q in n.qargs]
        if op.name=='swap':cat='swap'
        elif op.name=='cx':cat='cx'
        elif op.name=='measure':cat='measure'
        elif op.name=='x' and getattr(op,'condition',None) is not None:cat='conditional_x'
        elif len(qs)==2:cat='other_2q'
        else:cat='other'
        perlayer[layer][cat]+=1
        if len(qs)==2:byedge[tuple(sorted(qs))][cat]+=1
        event_rows.append({'op_index':order,'asap_layer':layer,'op':op.name,'category':cat,
                           'physical_qubits':','.join(map(str,qs)),'critical_asap':int(crit),
                           'reverse_tail':tail[ident]})
    for layer in range(1,depth+1):perlayer[layer]  # materialize empty
    swap_layers=sum(1 for x in perlayer.values() if x['swap']>0)
    consecutive=0;longest=0
    for idx in range(1,depth+1):
        consecutive=consecutive+1 if perlayer[idx]['swap'] else 0
        longest=max(longest,consecutive)
    stats={'asap_depth':depth,'swap_critical_count':sum(e['category']=='swap' and e['critical_asap'] for e in event_rows),
           'cx_critical_count':sum(e['category']=='cx' and e['critical_asap'] for e in event_rows),
           'swap_layers':swap_layers,'max_swaps_in_layer':max((v['swap'] for v in perlayer.values()),default=0),
           'max_consecutive_swap_layers':longest,
           'total_2q_layers':sum(1 for x in perlayer.values() if x['swap']+x['cx']+x['other_2q']>0)}
    layers=[{'asap_layer':layer,**{k:int(perlayer[layer].get(k,0)) for k in ['swap','cx','measure','conditional_x','other_2q','other']}}
            for layer in range(1,depth+1)]
    edges=[{'physical_a':edge[0],'physical_b':edge[1],
            'swap':int(val['swap']),'cx':int(val['cx']),'other_2q':int(val['other_2q'])}
           for edge,val in sorted(byedge.items())]
    return stats,event_rows,layers,edges


def write_csv(path, rows, columns=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if columns is None:
        columns=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)


def summarise_runs(rows):
    import statistics
    out=[]
    for circuit,better,worse in PAIR_SPECS:
        subset=[r for r in rows if r['circuit']==circuit]
        modes=sorted(set(r['mode'] for r in subset))
        for mode in modes:
            a={int(r['seed']):r for r in subset if r['mode']==mode and r['plan_id']==better}
            b={int(r['seed']):r for r in subset if r['mode']==mode and r['plan_id']==worse}
            for seed in sorted(set(a)&set(b)):
                one=a[seed];two=b[seed]
                z={'circuit':circuit,'mode':mode,'seed':seed,'better':better,'worse':worse}
                for metric in ['routed_depth','swap_count','swap_critical_count','swap_layers','routed_overhead',
                               'cx_critical_count','max_consecutive_swap_layers']:
                    z[f'better_{metric}']=float(one[metric]);z[f'worse_{metric}']=float(two[metric]);
                    z[f'delta_{metric}']=float(two[metric])-float(one[metric])
                out.append(z)
    means=[]
    for circuit,_,_ in PAIR_SPECS:
        for mode in sorted({x['mode'] for x in out if x['circuit']==circuit}):
            values=[x for x in out if x['circuit']==circuit and x['mode']==mode]
            if values:
                x={'circuit':circuit,'mode':mode,'n_matched_seeds':len(values)}
                for metric in ['routed_depth','swap_count','swap_critical_count','swap_layers','routed_overhead',
                               'cx_critical_count','max_consecutive_swap_layers']:
                    diffs=[t[f'delta_{metric}'] for t in values]
                    x[f'mean_delta_{metric}']=statistics.mean(diffs)
                    x[f'min_delta_{metric}']=min(diffs);x[f'max_delta_{metric}']=max(diffs)
                means.append(x)
    return out,means


def generate_figures(out, rows, paired):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path=out/'figures';path.mkdir(parents=True,exist_ok=True)
    for circuit,better,worse in PAIR_SPECS:
        d=[r for r in rows if r['circuit']==circuit]
        if not d:continue
        fig,ax=plt.subplots(figsize=(10,5))
        for mode in sorted(set(r['mode'] for r in d)):
            y=[x for x in paired if x['circuit']==circuit and x['mode']==mode]
            ax.plot([int(x['seed']) for x in y],[x['delta_routed_depth'] for x in y],marker='o',label=mode)
        ax.axhline(0,linewidth=1,color='black');ax.set(title=f'{circuit}: worse minus better routed depth',xlabel='SABRE seed',ylabel='depth difference')
        ax.legend();fig.tight_layout();fig.savefig(path/f'{circuit.replace(".qasm","")}_paired_depth.png',dpi=170);plt.close(fig)


def make_report(out, rows, paired,means, older, seeds, anchors):
    lines=['# Stage B — Targeted SABRE diagnostics','',
       'All numbers below are NEW runs on locally compiled fixed plans, not direct reuse of archived routed circuits.',
       'A common SABRE routing seed does not force an identical internal routing path. Fixed layouts control only the **initial** placement, not the subsequent heuristic choices.',
       'Correspondence between plans is a maximum-overlap matching of reuse chains; no unique physical identity exists for an unlabelled chain. No causal proof is claimed.',
       '','Routing: `CouplingMap.from_heavy_hex(7)`, `routing_method=sabre`, `layout_method=sabre` for free placement, `optimization_level=1`.','',
       f'Seeds: {seeds}. Anchors from free SABRE layouts at seed {anchors}.',
       '', '| Circuit | Mode | Matched seeds | Mean depth difference (worse−better) | Mean SWAP difference | Mean SWAPs on critical DAG path difference |',
       '|---|---|---:|---:|---:|---:|']
    for r in means:
        lines.append(f"| {r['circuit']} | {r['mode']} | {r['n_matched_seeds']} | {r['mean_delta_routed_depth']:+.3f} | {r['mean_delta_swap_count']:+.3f} | {r['mean_delta_swap_critical_count']:+.3f} |")
    lines+=['','## Archived 30-seed means (reference ONLY)','']
    for circuit,better,worse in PAIR_SPECS:
        ref=older[circuit];lines.append(f'- {circuit}: {better} mean routed depth {float(ref[better]["fresh_mean_routed_depth"]):.3f}; '
                f'{worse} {float(ref[worse]["fresh_mean_routed_depth"]):.3f}. '
                'If the NEW values differ, compare versions/compilation paths; never silently combine experiments.')
    lines+=['','## Interpretation','',
      '- If a depth gap persists for **both** physical layout anchors, the gap is not explained solely by SABRE choosing different initial placements for these particular anchors.',
      '- If it shrinks strongly under common layouts, initial placement may be contributing, but anchor asymmetry and routing stochasticity remain possible.',
      '- Fewer total SWAPs need not imply shorter routed depth: compare SWAP-bearing ASAP layers and SWAP operations on routed-DAG critical paths.',
      '- ASAP layers use unit operation duration, not device-calibrated execution times. Transpilation may rewrite gates: physical operations cannot in general be matched one-to-one with original QASM gate IDs.',
      '- With only a few seeds this is a pilot diagnostic. Do not call individual differences statistically conclusive or representative of other plan pairs.',
      '- Native GidNET reset insertion and active-reset normalization must reproduce Experiment 1 logical depth; otherwise the script aborts before routing.',
      '']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,required=True,help='Folder containing GPT CSV files')
    ap.add_argument('--qasm-dir',type=Path,required=True,help='Original CaQR benchmark QASMs (recursive)')
    ap.add_argument('--out-dir',type=Path,default=Path('targeted_routing_results'))
    ap.add_argument('--seeds',nargs='+',type=int,default=[100,101,102])
    ap.add_argument('--anchor-seed',type=int,default=100)
    ap.add_argument('--resume',action='store_true',help='Reuse output of interrupted run only if config matches')
    ap.add_argument('--dry-run',action='store_true',help='Compile and verify frozen plans, no SABRE transpilation')
    args=ap.parse_args(argv)
    if len(set(args.seeds))!=len(args.seeds) or args.anchor_seed not in args.seeds:ap.error('Seeds must be distinct and include anchor seed')
    if not args.data_dir.exists() or not args.qasm_dir.exists():ap.error('data-dir / qasm-dir must exist')
    # Script is inside project-root/package/. Import the user's locally patched GidNET.
    project_root=Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:sys.path.insert(0,str(project_root))
    try:
        import qiskit
        from qiskit import QuantumCircuit,transpile
        from qiskit.transpiler import CouplingMap,Layout
        from qiskit import qpy
        from gidnet.qubitreuse import GidNET
    except Exception as e:raise RuntimeError('Use .venv_gidnet with Qiskit 0.45/0.46 and local GidNET installed.') from e
    allplans,old,_=read_input_plan_data(args.data_dir)
    experiment2, exp2_path=collect_csv(args.data_dir,'experiment2_plan_features.csv')
    experiment2_lookup={(row['circuit'],row['plan_id']):row for row in experiment2}
    out=args.out_dir;out.mkdir(parents=True,exist_ok=True)
    coupling=CouplingMap.from_heavy_hex(7)
    hardware=set(map(tuple,coupling.get_edges()))
    gidnet_source=inspect.getfile(GidNET)
    qasms={c:find_qasm(args.qasm_dir,c) for c,_,_ in PAIR_SPECS}
    conf={'schema':1,'seeds':args.seeds,'anchor_seed':args.anchor_seed,'pairs':PAIR_SPECS,
          'qasm_sha256':{k:sha256(v) for k,v in qasms.items()},
          'inputs_sha256':{name:sha256(collect_csv(args.data_dir,name)[1]) for name in
             ['experiment1_fresh30_plan_summary.csv','experiment1_fresh30_raw.csv','experiment2_plan_features.csv']},
          'gidnet_python_sha256':sha256(gidnet_source),'qiskit_version':qiskit.__version__,
          'coupling_edges_sha256':hashlib.sha256(repr(sorted(hardware)).encode()).hexdigest(),
          'mode':['free','anchor_better','anchor_worse'],'opt':1,
          'script_sha256':sha256(__file__)}
    configfile=out/'config.json';runfile=out/'routing_runs.csv'
    if configfile.exists():
        saved=json.loads(configfile.read_text(encoding='utf-8'))
        if saved!=conf:raise RuntimeError('Existing result folder uses different inputs/code/version/seeds. Use a NEW --out-dir.')
        if not args.resume and runfile.exists() and not args.dry_run:
            raise RuntimeError('Existing routed runs found. To resume use --resume, or choose a NEW --out-dir.')
    else:
        if runfile.exists():raise RuntimeError('Output directory has routing_runs.csv but no config.json')
        configfile.write_text(json.dumps(conf,ensure_ascii=False,indent=2),encoding='utf-8')
    existing=[]
    if runfile.exists():
        with runfile.open(encoding='utf-8-sig',newline='') as f:existing=list(csv.DictReader(f))
    completed={(r['circuit'],r['mode'],int(r['seed']),r['plan_id']) for r in existing}
    normalized={}; metadata={};alignments={}
    for circuit,better,worse in PAIR_SPECS:
        original=QuantumCircuit.from_qasm_file(str(qasms[circuit]))
        # Selected QASMs are static, gate-only; do not silently reinterpret dynamic input.
        unexpected={op.name for op,_,_ in op_items(original) if op.name in ('measure','reset','if_else','while_loop','for_loop')}
        if unexpected:raise ValueError(f'Input QASM includes dynamic instructions {unexpected} not supported by frozen experiment')
        chains={}
        for plan_id in (better,worse):
            info=allplans[circuit][plan_id]
            chains[plan_id]=parse_chains(info['plan_signature'])
            if len(chains[plan_id])!=int(info['width']):raise AssertionError('width != chain count')
            quo=check_quotient_features(original,chains[plan_id],experiment2_lookup[(circuit,plan_id)],circuit,plan_id)
            norm,raw,_=compile_fixed_plan(original,chains[plan_id],info['normalized_logical_depth'])
            normalized[(circuit,plan_id)]=norm
            with (out/f'{circuit[:-5]}__{plan_id}__normalized_input.qpy').open('wb') as f:qpy.dump(norm,f)
            metadata[f'{circuit}:{plan_id}']={'native_depth':raw.depth(),'native_count_ops':dict(raw.count_ops()),
                'normalized_depth':norm.depth(),'normalized_count_ops':dict(norm.count_ops()),
                'fingerprint':circuit_fingerprint(norm),'chains':chains[plan_id],
                'reconstructed_exp2':quo}
        alignments[circuit],score=match_chains(chains[better],chains[worse])
        print(f'[OK] {circuit}: width {len(chains[better])}, matched chains score={score}, '\
              f'logical depths={metadata[circuit+":"+better]["normalized_depth"]}/{metadata[circuit+":"+worse]["normalized_depth"]}')
    (out/'compiled_plan_audit.json').write_text(json.dumps({'plans':metadata,'matching':alignments},indent=2,ensure_ascii=False),encoding='utf-8')
    if args.dry_run:
        print('DRY-RUN PASS: fixed-plan compilation, static validation, reset normalization and archived logical depth all match. No routing run.');return
    # Collect all rows in memory for lightweight 2-pair pilot. Commit each routed run to CSV for recoverability.
    runs=existing.copy()
    total=len(PAIR_SPECS)*len(args.seeds)*2*3; done=len(runs)
    for circuit,better,worse in PAIR_SPECS:
        plans=(better,worse)
        anchor_physical={}
        layout_match=alignments[circuit]
        for mode in ['free','anchor_better','anchor_worse']:
            for seed in args.seeds:
                for plan_id in plans:
                    key=(circuit,mode,seed,plan_id)
                    if key in completed:continue
                    if mode!='free':
                        anchor_role=better if mode=='anchor_better' else worse
                        anchor_key=(circuit,'free',args.anchor_seed,anchor_role)
                        ref=next((r for r in runs if (r['circuit'],r['mode'],int(r['seed']),r['plan_id'])==anchor_key),None)
                        if ref is None:raise RuntimeError('Must generate free anchor before fixed routing; missing '+str(anchor_key))
                        phys=json.loads(ref['initial_layout_json'])
                        if plan_id!=anchor_role:
                            phys=reposition_layout(phys,layout_match,'better' if anchor_role==better else 'worse')
                    else:phys=None
                    inp=normalized[(circuit,plan_id)]
                    kw={'coupling_map':coupling,'routing_method':'sabre',
                        'optimization_level':1,'seed_transpiler':int(seed)}
                    if mode=='free':kw['layout_method']='sabre'
                    else:kw['initial_layout']=Layout({q:int(p) for q,p in zip(inp.qubits,phys)})
                    routed=transpile(inp,**kw)
                    used=physical_layout_from_route(routed,inp)
                    if phys is not None and used!=phys:
                        raise RuntimeError(f'Fixed initial layout was silently changed: requested {phys}, actual {used}')
                    count=dict(routed.count_ops());gstats,events,layers,edges=graph_layers(routed)
                    if gstats['asap_depth']!=routed.depth():
                        raise RuntimeError(f'ASAP DAG analysis depth {gstats["asap_depth"]} != routed.depth()={routed.depth()}')
                    qpyname=f'{circuit[:-5]}__{plan_id}__{mode}__seed{seed}.qpy'
                    folder=out/'routed_qpy';folder.mkdir(exist_ok=True)
                    with (folder/qpyname).open('wb') as f:qpy.dump(routed,f)
                    pref=f'{circuit[:-5]}__{plan_id}__{mode}__seed{seed}'
                    for name,items in [('events',events),('layers',layers),('edges',edges)]:
                        write_csv(out/'diagnostics'/f'{pref}__{name}.csv',items)
                    row={'circuit':circuit,'mode':mode,'seed':seed,'plan_id':plan_id,
                         'role':'better' if plan_id==better else 'worse','width':inp.num_qubits,
                         'source_depth':int(metadata[f'{circuit}:{plan_id}']['native_depth']),
                         'normalized_depth':int(inp.depth()),'expected_normalized_depth':int(allplans[circuit][plan_id]['normalized_logical_depth']),
                         'routed_depth':int(routed.depth()),'swap_count':int(count.get('swap',0)),
                         'cx_count':int(count.get('cx',0)),'measurements':int(count.get('measure',0)),
                         'conditional_x':int(conditional_x(routed)),
                         'routed_overhead':int(routed.depth()-inp.depth()),
                         'initial_layout_json':json.dumps(used),'routed_qpy':str(Path('routed_qpy')/qpyname),
                         'circuit_fingerprint':metadata[f'{circuit}:{plan_id}']['fingerprint'],**gstats}
                    runs.append(row);completed.add(key);done+=1;write_csv(runfile,runs,RUN_COLS)
                    print(f'[{done}/{total}] {circuit}:{plan_id} {mode} seed={seed} depth={routed.depth()} swap={count.get("swap",0)}')
    pair_rows,means=summarise_runs(runs)
    write_csv(out/'paired_seed_differences.csv',pair_rows)
    write_csv(out/'mode_summary.csv',means)
    generate_figures(out,runs,pair_rows)
    make_report(out,runs,pair_rows,means,allplans,args.seeds,args.anchor_seed)
    print('\nDONE: See REPORT.md, paired_seed_differences.csv, mode_summary.csv, routing_runs.csv and routed_qpy/')


if __name__=='__main__':main()
