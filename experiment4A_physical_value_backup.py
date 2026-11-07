#!/usr/bin/env python3
"""Experiment 4A: physical value of exhaustive one-step fixed-width reuse neighbors.

PREPARE: input validation, candidate canonicalization and source memberships.
EXPLORE: cached physical routing on discovery seeds, freeze A/B/C winners.
VALIDATE: held-out paired-seed routing of frozen choices; no reselection.

The Full Coverage files establish STATIC feasibility, not semantic equivalence.
The compilation bridge must preserve the actual Experiment-3 pipeline.
This is an opportunity study, NOT a budget-matched repair comparison.
"""
from __future__ import annotations
import argparse
import ast
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import pandas as pd

OPS = {"suffix_swap", "chain_transfer", "node_swap_reconnect", "segment_reverse"}
POOLS = {
    "A_suffix": {"suffix_swap"},
    "B_suffix_chain": {"suffix_swap", "chain_transfer"},
    "C_all": OPS,
}
METRICS = ("swap_count", "routed_depth")
TRAIN = (42, 43, 44, 45, 46)
HOLDOUT = tuple(range(100, 130))


def cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "explore", "validate"])
    p.add_argument("--mutations", type=Path, default=Path("experiment4a_inputs/adversarial_mutation_cases.csv"))
    p.add_argument("--audit", type=Path, default=Path("experiment4a_inputs/width_and_original_plan_audit.csv"))
    p.add_argument("--plans", type=Path, default=Path("experiment2_results/experiment2_plan_features.csv"))
    p.add_argument("--bench-dir", type=Path, help="Path to directory containing original QASM files")
    p.add_argument("--out", type=Path, default=Path("experiment4a_results"))
    p.add_argument("--adapter", type=Path, help="OPTIONAL verified Experiment-3 pipeline adapter module")
    p.add_argument("--train-seeds", type=int, nargs="+", default=list(TRAIN))
    p.add_argument("--holdout-seeds", type=int, nargs="+", default=list(HOLDOUT))
    p.add_argument("--depth-cap-pct", type=float, default=5.0)
    p.add_argument("--smoke-routes", type=int, default=0, help="Route at most N new runs and safely stop")
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--allow-other-dataset", action="store_true", help="Disable 8642/50 full-coverage guard")
    return p.parse_args(argv)


def ensure(cond, message):
    if not cond:
        raise RuntimeError(message)


def read_csv(path):
    ensure(path.is_file(), f"Missing required input: {path.resolve()}")
    return pd.read_csv(path, dtype={"plan_id": "str", "candidate_key": "str"})


def true_values(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def canonical(raw):
    if isinstance(raw, str):
        raw = ast.literal_eval(raw)
    ensure(isinstance(raw, (list, tuple)) and len(raw) > 0, "Malformed reuse_chains")
    chains = tuple(tuple(int(q) for q in ch) for ch in raw)
    ensure(all(ch for ch in chains), "Empty reuse chain")
    flat = [q for ch in chains for q in ch]
    ensure(all(q >= 0 for q in flat) and len(flat) == len(set(flat)), "Invalid qubit partition")
    return tuple(sorted(chains))


def key_for(circ, chains, width):
    return hashlib.sha256(f"{circ}|{width}|{repr(chains)}".encode()).hexdigest()[:24]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def config_guard(out, config):
    path = out / "run_manifest.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf8"))
        ensure(old == config, f"Inputs/settings changed; do not mix with old results in {out}. Use a new --out.")
    else:
        out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf8")


def inventory(args):
    m = read_csv(args.mutations)
    a = read_csv(args.audit)
    p = read_csv(args.plans)
    req_m = {"circuit", "plan_id", "operator", "candidate_chains", "checker_is_valid",
             "reference_is_valid", "oracle_agrees", "preserved_width", "preserved_partition"}
    req_a = {"circuit", "plan_id", "saved_width", "checker_is_valid", "reference_is_valid"}
    req_p = {"circuit", "plan_id", "width", "reuse_chains"}
    for name, df, required in [("mutation",m,req_m), ("audit",a,req_a), ("plan",p,req_p)]:
        ensure(required <= set(df.columns), f"{name} CSV missing columns: {sorted(required-set(df.columns))}")
    ensure(not a.duplicated(["circuit", "plan_id"]).any(), "Duplicate original plan keys in audit")
    ensure(not p.duplicated(["circuit", "plan_id"]).any(), "Duplicate original plan keys in plans")
    keys = set(zip(a.circuit.astype(str), a.plan_id.astype(str)))
    ensure(set(zip(m.circuit.astype(str),m.plan_id.astype(str))) == keys, "Audit vs mutation source keys differ")
    ensure(keys <= set(zip(p.circuit.astype(str),p.plan_id.astype(str))), "Original-plan ledger missing source plans")
    ensure(true_values(a.checker_is_valid).all() and true_values(a.reference_is_valid).all(),
           "Some original plans are not statically valid")
    ensure(true_values(m.oracle_agrees).all() and true_values(m.preserved_width).all() and
           true_values(m.preserved_partition).all(), "Full Coverage integrity check failed")
    ensure((true_values(m.checker_is_valid) == true_values(m.reference_is_valid)).all(),
           "Checker disagrees with reference oracle")
    ensure(set(m.operator.astype(str)) <= OPS, "Unsupported Full Coverage operator")
    valid = m.loc[true_values(m.checker_is_valid)]
    if not args.allow_other_dataset:
        ensure(len(m) == 8642 and len(keys) == 50 and len(valid) == 485,
               f"Expected exhaustive Full Coverage (8642 mutations, 50 sources, 485 static valid), "
               f"got ({len(m)}, {len(keys)}, {len(valid)}). Check that this is the FULL file.")
    if "candidate_signature" in m:
        for row in valid.itertuples(index=False):
            ensure(repr(canonical(row.candidate_chains)) == str(row.candidate_signature),
                   f"Inconsistent candidate signature: {row.circuit}/{row.plan_id}")
    aud = {(str(r.circuit),str(r.plan_id)):int(r.saved_width) for r in a.itertuples(index=False)}
    plans = {(str(r.circuit),str(r.plan_id)):r for r in p.itertuples(index=False)}
    cat, memb = {}, defaultdict(set)
    origins = {}
    for (circ,pid) in sorted(keys):
        width = aud[(circ,pid)]
        r = plans[(circ,pid)]
        ch = canonical(r.reuse_chains)
        ensure(width == int(r.width) == len(ch), f"Original width mismatch {circ}/{pid}")
        k = key_for(circ,ch,width)
        origins[(circ,pid)] = k
        cat[(circ,k)] = {"circuit":circ,"candidate_key":k,"width":width,"reuse_chains":repr(ch)}
    for r in valid.itertuples(index=False):
        circ,pid = str(r.circuit),str(r.plan_id)
        width = aud[(circ,pid)]
        ch = canonical(r.candidate_chains)
        ensure(len(ch) == width, f"Candidate width mismatch {circ}/{pid}")
        orig_qubits = sorted(q for chain in canonical(plans[(circ,pid)].reuse_chains) for q in chain)
        ensure(sorted(q for chain in ch for q in chain) == orig_qubits,
               f"Candidate active-qubit coverage mismatch: {circ}/{pid}")
        k = key_for(circ,ch,width)
        item = {"circuit":circ,"candidate_key":k,"width":width,"reuse_chains":repr(ch)}
        ensure((circ,k) not in cat or cat[(circ,k)] == item, "Hash collision / inconsistent plan")
        cat[(circ,k)] = item
        if k != origins[(circ,pid)]:
            memb[(circ,pid,k)].add(str(r.operator))
    members = []
    for (circ,pid),k in sorted(origins.items()):
        members.append({"circuit":circ,"plan_id":pid,"candidate_key":k,
                        "operators":"original", "is_original":True,
                        **{f"in_{pool}":False for pool in POOLS}})
    for (circ,pid,k),ops in sorted(memb.items()):
        members.append({"circuit":circ,"plan_id":pid,"candidate_key":k,
                        "operators":";".join(sorted(ops)), "is_original":False,
                        **{f"in_{pool}":bool(ops & o) for pool,o in POOLS.items()}})
    catalog = pd.DataFrame([cat[k] for k in sorted(cat)])
    membership = pd.DataFrame(members)
    counts = []
    for (circ,pid),g in membership.groupby(["circuit","plan_id"]):
        rec = {"circuit":circ,"plan_id":pid}
        for pool in POOLS:
            rec[pool] = int((g[f"in_{pool}"] & ~g.is_original).sum())
        counts.append(rec)
    coverage = pd.DataFrame(counts)
    if not args.allow_other_dataset:
        totals = tuple(int(coverage[k].sum()) for k in POOLS)
        ensure(totals == (130,289,291), f"Distinct candidate counts mismatch: expected (130, 289, 291), got {totals}")
    save(catalog,args.out/"candidate_catalog.csv")
    save(membership,args.out/"source_memberships.csv")
    save(coverage,args.out/"coverage_per_source.csv")
    report={"mutations":len(m),"valid_static_mutations":len(valid),"sources":len(keys),
            "circuits":len(set(x[0] for x in keys)),"unique_physical_plans_including_originals":len(catalog),
            "cumulative_neighborhood_counts":{k:int(coverage[k].sum()) for k in POOLS},
            "semantic_equivalence_proven":False}
    (args.out/"inventory_report.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print("INVENTORY:",json.dumps(report,indent=2),flush=True)
    return catalog,membership,origins


def pipeline(adapter):
    """Use user-verified Exp3 adapter, else patched GidNET's fixed-sequence compiler.

    Default preserves exact stored chains, but parity with Exp3 must be confirmed.
    """
    if adapter:
        ensure(adapter.is_file(),f"Adapter not found: {adapter}")
        spec=importlib.util.spec_from_file_location("exp4a_user_adapter",str(adapter.resolve()))
        ensure(spec is not None and spec.loader is not None,"Cannot load adapter")
        mod=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        build=getattr(mod,"build_dynamic_circuit",None)
        normalize=getattr(mod,"normalize_for_routing",lambda qc:qc)
        ensure(callable(build) and callable(normalize),"Adapter must define build_dynamic_circuit(qc, chains) and optionally normalize_for_routing(qc)")
        return build,normalize,"verified-user-adapter:"+digest(adapter)
    try:
        from gidnet.qubitreuse import GidNET
        from gidnet.reset_lowering import lower_reset_to_measure_x
    except ImportError as e:
        raise RuntimeError("Could not import patched gidnet. Run from project root with the Experiment-3 environment, or pass --adapter.") from e
    def build(qc, chains):
        worker=GidNET(qc.copy())
        worker.compute_optimized_reuse_sequences=lambda iterations: [list(c) for c in chains]
        return worker.compile_to_dynamic_circuit(iterations=0)
    return build,lower_reset_to_measure_x,"patched-GidNET-injected-sequences;verify-Exp3-parity"


def run_routes(args,catalog,needed,seeds,phase):
    try:
        from qiskit import qasm2,transpile
        from qiskit.transpiler import CouplingMap
        import qiskit
    except ImportError as e:
        raise RuntimeError("Qiskit missing in this Python environment. Activate your Experiment-3 Python first.") from e
    ensure(args.bench_dir and args.bench_dir.is_dir(),"--bench-dir must point to QASM benchmark folder")
    build,normalize,tag=pipeline(args.adapter)
    meta={"compiler":tag,"qiskit":qiskit.__version__,
          "coupling":"heavy_hex(7)","layout":"sabre","routing":"sabre","optimization":1,
          "train_seeds":sorted(args.train_seeds),"holdout_seeds":sorted(args.holdout_seeds),
          "mutations_sha256":digest(args.mutations),"audit_sha256":digest(args.audit),
          "plans_sha256":digest(args.plans),
          "qasm_sha256":{f:digest(args.bench_dir/f) for f in sorted(catalog.circuit.unique())},
          "depth_cap_pct":args.depth_cap_pct,
          "semantic_check":"not proved by Experiment 4A; see separate semantic audit"}
    config_guard(args.out,meta)
    coupling=CouplingMap.from_heavy_hex(7)
    filename=args.out/("routes_train.csv" if phase=="train" else "routes_holdout.csv")
    cached={}
    if filename.is_file():
        past=pd.read_csv(filename,dtype={"candidate_key":"str"})
        for r in past.to_dict("records"):
            key=(str(r["circuit"]),str(r["candidate_key"]),int(r["seed"]))
            ensure(key not in cached,"Duplicate existing route entry; stopping")
            cached[key]=r
    qasms={}
    done=0
    # Start with smaller circuits so --smoke-routes 1 tests the cheapest known case.
    preferred={"system9.qasm":0,"sym9_146.qasm":1,"rd73_140.qasm":2,
               "mini_alu_305.qasm":3,"alu-bdd_288.qasm":4}
    ordered=sorted(catalog.itertuples(index=False),
                   key=lambda r:(preferred.get(str(r.circuit),99),str(r.circuit),str(r.candidate_key)))
    for r in ordered:
        ck=(str(r.circuit),str(r.candidate_key))
        if ck not in needed: continue
        todo=[s for s in seeds if ck+(s,) not in cached or
              (args.retry_errors and cached[ck+(s,)]["status"]!="ok")]
        if not todo: continue
        if r.circuit not in qasms:
            qasms[r.circuit]=qasm2.load(str(args.bench_dir/r.circuit))
        qc=qasms[r.circuit]
        chains=canonical(r.reuse_chains)
        active=set(q for ch in chains for q in ch)
        ensure(max(active)<qc.num_qubits, f"Candidate contains non-existent qubit: {ck}")
        try:
            dyn=build(qc,chains)
            ensure(dyn.num_qubits==int(r.width), f"Dynamic width mismatch for {ck}: got {dyn.num_qubits}, expected {r.width}; check inactive qubits/compiler")
            prepared=normalize(dyn)
            ensure(prepared.num_qubits==int(r.width),f"Reset-lowered width mismatch {ck}")
            logical_depth=int(prepared.depth())
        except Exception as e:
            raise RuntimeError(f"Dynamic compilation failed for {ck}. No fake routing values saved. {type(e).__name__}: {e}") from e
        for seed in todo:
            start=time.perf_counter()
            try:
                routed=transpile(prepared,coupling_map=coupling,layout_method="sabre",routing_method="sabre",
                                optimization_level=1,seed_transpiler=int(seed))
                new={"phase":phase,"circuit":ck[0],"candidate_key":ck[1],"seed":int(seed),
                     "status":"ok","swap_count":int(routed.count_ops().get("swap",0)),
                     "routed_depth":int(routed.depth()),"logical_depth":logical_depth,
                     "seconds":time.perf_counter()-start,"error":""}
            except Exception as e:
                new={"phase":phase,"circuit":ck[0],"candidate_key":ck[1],"seed":int(seed),
                     "status":"error","swap_count":float("nan"),"routed_depth":float("nan"),
                     "logical_depth":logical_depth,"seconds":time.perf_counter()-start,
                     "error":repr(e)[:500]}
            cached[ck+(seed,)]=new
            save(pd.DataFrame(list(cached.values())),filename)   # checkpoint per route
            done+=1
            if done%25==0: print(f"{phase}: saved {done} new routes; {len(cached)} total",flush=True)
            if new["status"]!="ok": print(f"ERROR in {ck} seed={seed}: {new['error']}",file=sys.stderr,flush=True)
            if args.smoke_routes and done>=args.smoke_routes:
                print("SMOKE ROUTES COMPLETE. Resume with --smoke-routes 0 (no result summary yet).",flush=True)
                return False
    return True


def strict_routes(path,needed,seeds):
    ensure(path.is_file(),f"Missing {path}; run explore fully first")
    routes=pd.read_csv(path,dtype={"candidate_key":"str"})
    want={(c,k,s) for c,k in needed for s in seeds}
    actual={(str(r.circuit),str(r.candidate_key),int(r.seed)) for r in routes.itertuples(index=False)}
    ensure(len(actual)==len(routes),"Duplicate physical routing rows")
    ensure(want<=actual,f"Incomplete physical routes; {len(want-actual)} remain. Resume execution.")
    filtered=routes.loc[routes.apply(lambda r:(r.circuit,r.candidate_key,int(r.seed)) in want,axis=1)].copy()
    ensure(filtered.status.eq("ok").all(),"Some physical routes failed; see routes CSV; retry errors or fix compiler")
    for metric in METRICS:
        filtered[metric]=pd.to_numeric(filtered[metric],errors="coerce")
        ensure(filtered[metric].notna().all(),f"Metric {metric} missing")
    return filtered


def choose(args,members,routes):
    averages=routes.groupby(["circuit","candidate_key"],as_index=False).agg(
        mean_swap_count=("swap_count","mean"),mean_routed_depth=("routed_depth","mean"),
        logical_depth=("logical_depth","first"), std_swaps=("swap_count","std"),std_depth=("routed_depth","std"))
    save(averages,args.out/"candidate_training_means.csv")
    look={(r.circuit,r.candidate_key):r for r in averages.itertuples(index=False)}
    choices=[]
    for (c,pid),group in members.groupby(["circuit","plan_id"]):
        original=group.loc[group.is_original]
        ensure(len(original)==1,f"Must have exactly one original: {c}/{pid}")
        ori=original.iloc[0].candidate_key
        source=look[(c,ori)]
        for metric in METRICS:
            mean_metric="mean_"+metric
            original_cost=float(getattr(source,mean_metric))
            for cap in [False,True]:
                limit=source.logical_depth*(1+args.depth_cap_pct/100.)
                for pool in POOLS:
                    choices_pool=group.loc[group[f"in_{pool}"] & ~group.is_original]
                    if cap:
                        choices_pool=choices_pool.loc[choices_pool.candidate_key.map(lambda k:look[(c,k)].logical_depth<=limit+1e-9)]
                    if choices_pool.empty:
                        neighbor=""
                        cost_neighbor=float("nan")
                        chosen=ori
                        no_neighbor=True
                    else:
                        neighbor=min(choices_pool.candidate_key,
                                     key=lambda k:(float(getattr(look[(c,k)],mean_metric)),k))
                        cost_neighbor=float(getattr(look[(c,neighbor)],mean_metric))
                        chosen=neighbor if cost_neighbor<original_cost else ori
                        no_neighbor=False
                    selected=float(getattr(look[(c,chosen)],mean_metric))
                    choices.append({"circuit":c,"plan_id":pid,"objective":metric,"pool":pool,
                                    "depth_cap":cap,"original_key":ori,"best_neighbor_key":neighbor,
                                    "selected_key":chosen,"no_eligible_neighbor":no_neighbor,
                                    "mean_original":original_cost,"mean_best_neighbor":cost_neighbor,
                                    "mean_selected":selected,"gain_vs_original":original_cost-selected,
                                    "gain_vs_original_pct":100*(original_cost-selected)/original_cost if original_cost else 0,
                                    "depth_original":source.logical_depth})
    df=pd.DataFrame(choices)
    save(df,args.out/"frozen_winners.csv")
    source=df.loc[df.pool=="A_suffix",["circuit","plan_id","objective","depth_cap","mean_selected"]].rename(columns={"mean_selected":"suffix_selected_mean"})
    df=df.merge(source,on=["circuit","plan_id","objective","depth_cap"],validate="many_to_one")
    df["incremental_gain_over_suffix"] = df.suffix_selected_mean-df.mean_selected
    save(df,args.out/"source_opportunity_comparison.csv")
    summary=df.groupby(["objective","pool","depth_cap"],as_index=False).agg(
        source_count=("plan_id","size"),wins_over_suffix=("incremental_gain_over_suffix",lambda s:int((s>1e-9).sum())),
        mean_gain_over_suffix=("incremental_gain_over_suffix","mean"),
        improved_over_original=("gain_vs_original",lambda s:int((s>1e-9).sum())),
        no_eligible_neighbor=("no_eligible_neighbor","sum"))
    save(summary,args.out/"opportunity_summary.csv")
    return df


def heldout_report(args,frozen,routes):
    look={(r.circuit,r.candidate_key,int(r.seed)):r for r in routes.itertuples(index=False)}
    pairs=[]
    for r in frozen.itertuples(index=False):
        for seed in args.holdout_seeds:
            ori=look[(r.circuit,r.original_key,int(seed))]
            win=look[(r.circuit,r.selected_key,int(seed))]
            base=float(getattr(ori,r.objective))
            got=float(getattr(win,r.objective))
            pairs.append({"circuit":r.circuit,"plan_id":r.plan_id,"objective":r.objective,"pool":r.pool,
                          "depth_cap":r.depth_cap,"seed":int(seed),"original":base,"selected":got,
                          "paired_improvement":base-got,"original_key":r.original_key,"selected_key":r.selected_key})
    paired=pd.DataFrame(pairs)
    save(paired,args.out/"heldout_paired_seeds.csv")
    agg=paired.groupby(["circuit","plan_id","objective","pool","depth_cap"],as_index=False).agg(
        heldout_original_mean=("original","mean"),heldout_selected_mean=("selected","mean"),
        heldout_gain_vs_original=("paired_improvement","mean"),holdout_seeds=("seed","nunique"))
    reference=agg.loc[agg.pool=="A_suffix",["circuit","plan_id","objective","depth_cap","heldout_selected_mean"]].rename(columns={"heldout_selected_mean":"heldout_suffix_mean"})
    agg=agg.merge(reference,on=["circuit","plan_id","objective","depth_cap"],validate="many_to_one")
    agg["heldout_incremental_gain_over_suffix"]=agg.heldout_suffix_mean-agg.heldout_selected_mean
    save(agg,args.out/"heldout_opportunity_comparison.csv")
    summary=agg.groupby(["objective","pool","depth_cap"],as_index=False).agg(
        sources=("plan_id","size"),sources_better_than_original=("heldout_gain_vs_original",lambda s:int((s>1e-9).sum())),
        sources_better_than_suffix=("heldout_incremental_gain_over_suffix",lambda s:int((s>1e-9).sum())),
        mean_gain_from_original=("heldout_gain_vs_original","mean"),
        mean_extra_gain_from_suffix=("heldout_incremental_gain_over_suffix","mean"))
    save(summary,args.out/"heldout_summary.csv")
    by_circuit=agg.groupby(["circuit","objective","pool","depth_cap"],as_index=False).agg(
        n_plans=("plan_id","size"),mean_extra_gain=("heldout_incremental_gain_over_suffix","mean"),
        plans_better_than_suffix=("heldout_incremental_gain_over_suffix",lambda s:int((s>1e-9).sum())))
    save(by_circuit,args.out/"heldout_by_circuit.csv")


def main(argv=None):
    args=cli(argv)
    ensure(args.depth_cap_pct>=0,"Depth cap must be nonnegative")
    ensure(len(args.train_seeds)==len(set(args.train_seeds)) and
           len(args.holdout_seeds)==len(set(args.holdout_seeds)) and
           not set(args.train_seeds)&set(args.holdout_seeds),"Routing seeds must be distinct across phases")
    catalog,members,origins=inventory(args)
    if args.stage=="prepare":
        print("PREPARE complete: candidate_catalog.csv, source_memberships.csv, coverage_per_source.csv")
        return
    ensure(args.bench_dir is not None,"Routing needs --bench-dir with the five original QASM files")
    if args.stage=="explore":
        required={(r.circuit,r.candidate_key) for r in catalog.itertuples(index=False)}
        completed=run_routes(args,catalog,required,args.train_seeds,"train")
        if completed:
            training=strict_routes(args.out/"routes_train.csv",required,args.train_seeds)
            choose(args,members,training)
            print("EXPLORE complete: see opportunity_summary.csv and frozen_winners.csv",flush=True)
    else:
        frozen_path=args.out/"frozen_winners.csv"
        ensure(frozen_path.is_file(),"Run EXPLORE fully before VALIDATE")
        frozen=pd.read_csv(frozen_path,dtype={"plan_id":"str","original_key":"str","selected_key":"str"})
        ensure(len(frozen)==len(origins)*len(METRICS)*len(POOLS)*2,
               "Frozen winner ledger count mismatch; do not use partial data")
        needed={(r.circuit,r.original_key) for r in frozen.itertuples(index=False)} | {
                (r.circuit,r.selected_key) for r in frozen.itertuples(index=False)}
        completed=run_routes(args,catalog,needed,args.holdout_seeds,"holdout")
        if completed:
            held=strict_routes(args.out/"routes_holdout.csv",needed,args.holdout_seeds)
            heldout_report(args,frozen,held)
            print("VALIDATE complete: see heldout_summary.csv, heldout_by_circuit.csv",flush=True)

if __name__=="__main__":
    main()
