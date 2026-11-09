#!/usr/bin/env python3
"""Training-only, topology-agnostic structural beam vs random beam.

NEW vs the Oct 2026 experiment: no Ridge anywhere; a fixed structural objective
is used for both beam expansion (guided policy only) and final candidate choice
(both methods). The random-uniform third control uses the exact random-beam pool.
The two main policies use identical proposal generation and caps.

Stages: search (NO routing), route (only selected plans; explicitly requested),
        analyze (requires complete fresh independent routing seeds).
Qiskit 0.45.3 + existing previously audited project modules are needed for real runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

POLICIES = ("structural_beam", "random_beam", "random_uniform")
OPS = ("suffix_swap", "chain_transfer")
SCHEMA = "structural_beam_control_v1_2026_11_04"


def read_csv(p):
    with Path(p).open("r", encoding="utf-8-sig", newline="") as h:
        return list(csv.DictReader(h))


def write_csv(p, records, fields):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8-sig", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({x: r.get(x, "") for x in fields})


def append_csv(p, record, fields):
    p = Path(p)
    is_new = not p.exists()
    with p.open("a", encoding="utf-8-sig" if is_new else "utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore")
        if is_new:
            w.writeheader()
        w.writerow({k: record.get(k, "") for k in fields})


def jwrite(p, data):
    Path(p).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str,parts)).encode()).digest()[:8], "big")


def signature(plan):
    import hashlib
    b = json.dumps([list(c) for c in tuple(sorted(tuple(int(q) for q in ch) for ch in plan))],
                   separators=(",", ":"))
    return hashlib.sha256(b.encode()).hexdigest()[:24]


def structural_score(plan, twoq_pairs):
    """Pre-registered topology-AGNOSTIC quotient-interaction score.

    Ordering: few distinct inter-chain interacting pairs, higher repeated-use
    edge HHI, lower maximum unweighted degree. Ties are exact ties; no Sig
    in score, because unchanged scores must result in No Repair.
    Uses only logical 2-qubit gate incidences and a valid plan, not routing.
    This is a *hypothesis*, not a calibrated SWAP predictor.
    """
    chain_of = {q: i for i, chain in enumerate(plan) for q in chain}
    edge = Counter()
    degree = [set() for _ in plan]
    for a,b in twoq_pairs:
        x,y = chain_of[a], chain_of[b]
        if x == y: continue
        u,v = sorted((x,y))
        edge[(u,v)] += 1
    for u,v in edge:
        degree[u].add(v); degree[v].add(u)
    total = sum(edge.values())
    hhi = sum(n*n for n in edge.values()) / (total*total) if total else 0.0
    return (len(edge), -round(hhi, 12), max(map(len, degree),default=0))


def score_fields(score):
    return {"unique_edges":int(score[0]),"edge_weight_hhi":-float(score[1]),"max_degree":int(score[2])}


def search_core(start, propose, score, *, policy, seed, beam_width=5,
                max_depth=5, budget=120, raw_budget=2400):
    """Pure search core to unit-test without Qiskit.

    propose(plan, signature, op, seed, cap) -> ([(signature, plan)], raw_attempts).
    Calls to propose for the same state are policy-independent due to node seed.
    All visited valid unique candidates count toward budget, no routed costs.
    """
    assert policy in ("structural_beam", "random_beam")
    if min(beam_width, max_depth, budget, raw_budget) < 1:
        raise ValueError("Positive budgets required")
    rng = random.Random(stable_seed(seed, policy, "beam"))
    start_sig = signature(start)
    start_score = score(start)
    seen = {start_sig: (start, 0, start_score)}
    trace = []
    frontier = [start_sig]
    raw_attempts = 0
    n_expanded_nodes = 0
    score_seconds = 0.0
    gen_seconds = 0.0
    stop_reason = "depth_limit"
    for depth in range(1, max_depth+1):
        layer = []
        for parent_sig in frontier:
            parent_plan = seen[parent_sig][0]
            n_expanded_nodes += 1
            for op in OPS:
                remaining_raw = raw_budget - raw_attempts
                if remaining_raw <= 0:
                    stop_reason = "raw_attempt_budget"
                    break
                began = time.perf_counter()
                pairs, attempts = propose(parent_plan, parent_sig, op,
                      stable_seed(seed, "node", parent_sig, op), remaining_raw)
                gen_seconds += time.perf_counter() - began
                if attempts < 0 or attempts > remaining_raw:
                    raise AssertionError("Proposal generator violated raw mutation budget")
                raw_attempts += attempts
                for sig, plan in pairs:
                    if sig in seen: continue
                    if len(seen)-1 >= budget:
                        stop_reason = "valid_budget"
                        break
                    began = time.perf_counter()
                    sc = score(plan)
                    score_seconds += time.perf_counter()-began
                    seen[sig] = (plan, depth, sc)
                    trace.append(dict(sig=sig,parent_sig=parent_sig,operator=op,
                             path_length=depth,**score_fields(sc)))
                    layer.append(sig)
                if len(seen)-1 >= budget or raw_attempts >= raw_budget:break
            if len(seen)-1 >= budget or raw_attempts >= raw_budget:break
        if not layer:
            stop_reason="exhausted_no_new_valid_neighbors"
            break
        if policy=="structural_beam":
            frontier=sorted(layer,key=lambda s:(seen[s][2],s))[:beam_width]
        else:
            rng.shuffle(layer)
            frontier=layer[:beam_width]
        if len(seen)-1 >= budget:break
        if raw_attempts >= raw_budget:
            stop_reason="raw_attempt_budget"
            break
    # Same final selector for both policies, including Original/No Repair.
    best_sig = min(seen, key=lambda s: (seen[s][2],s))
    if not (seen[best_sig][2] < start_score):best_sig=start_sig
    # Random-uniform uses exactly the random-beam set, costs NO extra proposals.
    uniform_sig = random.Random(stable_seed(seed,"uniform_final")).choice(sorted(seen))
    return dict(best_sig=best_sig,random_pick_sig=uniform_sig,
                seen=seen,trace=trace,n_evaluated=len(seen)-1,
                n_raw_attempts=raw_attempts, n_expanded_nodes=n_expanded_nodes,
                score_seconds=score_seconds, generation_seconds=gen_seconds,
                stop_reason=stop_reason)


def get_training_sources(train_dir, ledger_path, chosen, max_originals=0):
    """Only use original plans from the existing train-only Prepare stage."""
    train = Path(train_dir)
    ledger=read_csv(ledger_path)
    split={r["circuit_id"]:r for r in ledger}
    src=read_csv(train/"selected_candidates.csv")
    cat={(r["circuit_id"],r["sig"]):r for r in read_csv(train/"plan_catalog.csv")}
    manifest=json.loads((train/"manifest.json").read_text(encoding="utf-8"))
    origin=[]
    for r in src:
        if str(r["is_original"]) not in ("1","True","true"):continue
        cid=r["circuit_id"]
        if cid not in split:raise ValueError(f"Unknown split ledger circuit {cid}")
        if split[cid]["split"]!="train":raise ValueError(f"Non-train circuit in source data: {cid}")
        if chosen and cid not in chosen:continue
        key=cid,r["sig"]
        if key not in cat:raise ValueError(f"Original plan missing from catalog: {key}")
        if cid not in manifest["qasm_paths"]:raise ValueError(f"QASM missing from manifest: {cid}")
        origin.append(dict(**r, family_group=split[cid]["family_group"],
                           qasm_path=manifest["qasm_paths"][cid],
                           plan=tuple(sorted(tuple(int(q) for q in ch)
                               for ch in json.loads(cat[key]["plan_json"])))) )
    origin.sort(key=lambda r:(r["circuit_id"],r["original_sig"]))
    if max_originals:
        n=defaultdict(int)
        tmp=[]
        for r in origin:
            if n[r["circuit_id"]]<max_originals:
                tmp.append(r); n[r["circuit_id"]]+=1
        origin=tmp
    if chosen and not {x["circuit_id"] for x in origin} == chosen:
        raise ValueError(f"Requested circuits with no TRAIN originals: {chosen-{x['circuit_id'] for x in origin}}")
    if len({(r["circuit_id"],r["original_sig"]) for r in origin})!=len(origin):
        raise ValueError("Duplicate Original rows")
    return origin


def acquire_circuit(qasm_path):
    from qiskit import QuantumCircuit
    from build_multistep_pilot_validity_aware import compact_active, make_checker
    from reuse_adversarial import reference_static_feasible
    from validity_aware_neighbors import ValidityAwareEngine
    path=Path(qasm_path)
    if not path.exists():raise FileNotFoundError(f"QASM not found: {path}")
    qc,_ = compact_active(QuantumCircuit.from_qasm_file(str(path)))
    ck=make_checker(qc)
    engine=ValidityAwareEngine(ck)
    pairs=[]
    for item in qc.data:
        qs=[qc.find_bit(q).index for q in item.qubits]
        if len(qs)==2:pairs.append(tuple(qs))
    return qc,ck,engine,pairs,reference_static_feasible


def stage_search(args):
    from reuse_adversarial import canonical
    import qiskit
    if qiskit.__version__!="0.45.3":raise RuntimeError(f"Qiskit 0.45.3 required, got {qiskit.__version__}")
    out=Path(args.out_dir)
    if out.exists() and any(out.iterdir()):raise FileExistsError("Search output exists: choose NEW --out-dir to avoid overwriting prior experiments")
    chosen=set(x.strip() for x in args.circuits.split(",") if x.strip())
    origins=get_training_sources(args.train_dir,args.split_ledger,chosen,args.max_originals)
    if args.max_circuits:
        keep=set(sorted(set(x["circuit_id"] for x in origins))[:args.max_circuits])
        origins=[x for x in origins if x["circuit_id"] in keep]
    if not origins:raise ValueError("No training originals found")
    out.mkdir(parents=True,exist_ok=True)
    decisions,trace,selected_plans=[],[],{}
    caches={}
    for idx,orig in enumerate(origins,1):
        cid=orig["circuit_id"]
        if cid not in caches:caches[cid]=acquire_circuit(orig["qasm_path"])
        qc,ck,engine,twopairs,static_check=caches[cid]
        start=canonical(orig["plan"])
        if signature(start)!=orig["original_sig"]:raise AssertionError("Original hash mismatch")
        if not ck.validate(start,target_width=len(start)).is_valid or not static_check(qc,start,len(start)):
            raise AssertionError("Original invalid")
        def score(plan):return structural_score(plan,twopairs)
        def propose(plan, node_sig, op, node_seed, remaining_raw):
            node=engine.prepare(plan)
            found,stats=node.scan(op,max_attempts=min(args.max_attempts_per_op, remaining_raw),
                        verify_survivors=args.verify_fast_survivors)
            rng=random.Random(node_seed)
            rng.shuffle(found)
            chosen=found[:args.per_op_keep]
            return [(signature(p),p) for p in chosen],int(stats['attempted'])
        for rep in range(args.repetitions):
            base_seed=stable_seed(args.seed,cid,orig["original_sig"],rep)
            res={}
            for policy in ("structural_beam","random_beam"):
                res[policy]=search_core(start,propose,score,policy=policy,
                       seed=base_seed, beam_width=args.beam_width,max_depth=args.max_depth,
                       budget=args.budget,raw_budget=args.raw_budget)
            # Every candidate is independently valid; final plans also pass independent scheduler.
            for policy in POLICIES:
                result=res["random_beam"] if policy=="random_uniform" else res[policy]
                sig=result["random_pick_sig"] if policy=="random_uniform" else result["best_sig"]
                picked,depth,sc=result["seen"][sig]
                if len(picked)!=len(start) or not ck.validate(picked,target_width=len(start)).is_valid or not static_check(qc,picked,len(start)):
                    raise AssertionError(f"Selected invalid plan: {cid}/{sig}")
                for planid,pln in ((sig,picked),(orig["original_sig"],start)):
                    key=cid,planid
                    record=dict(circuit_id=cid,source=orig["source"],filename=orig["filename"],
                                sig=planid,active_width=len(start),plan_json=json.dumps([list(c) for c in pln],separators=(",", ":")))
                    if key in selected_plans and selected_plans[key]["plan_json"]!=record["plan_json"]:
                        raise AssertionError("Signature collision")
                    selected_plans[key]=record
                decisions.append(dict(circuit_id=cid,source=orig["source"],filename=orig["filename"],
                   family_group=orig["family_group"],original_sig=orig["original_sig"],
                   repetition=rep, policy=policy,pick_sig=sig,path_length=depth,
                   n_valid_scored=result["n_evaluated"],n_expanded_nodes=result["n_expanded_nodes"],raw_mutation_attempts=result["n_raw_attempts"],
                   score_time_sec=result["score_seconds"],generate_time_sec=result["generation_seconds"],
                   reason=result["stop_reason"],
                   original_unique_edges=score(start)[0],selected_unique_edges=sc[0],
                   original_hhi=-score(start)[1],selected_hhi=-sc[1]))
            for policy in ("structural_beam","random_beam"):
                for t in res[policy]["trace"]:
                    trace.append(dict(circuit_id=cid,original_sig=orig["original_sig"],
                                      repetition=rep,policy=policy,**t))
        print(f"[search {idx}/{len(origins)}] {cid} original={orig['original_sig']}",flush=True)
    write_csv(out/"search_decisions.csv",decisions,["circuit_id","source","filename","family_group", "original_sig",
         "repetition","policy","pick_sig","path_length","n_valid_scored","n_expanded_nodes","raw_mutation_attempts",
         "score_time_sec","generate_time_sec","reason","original_unique_edges","selected_unique_edges","original_hhi","selected_hhi"])
    write_csv(out/"selected_plan_catalog.csv",list(selected_plans.values()),["circuit_id","source","filename","sig","active_width","plan_json"])
    write_csv(out/"search_trace.csv",trace,["circuit_id","original_sig","repetition","policy","sig","parent_sig","operator","path_length","unique_edges","edge_weight_hhi","max_degree"])
    # Distinct from routed evaluation: this diagnostic is structural-only and
    # determines whether it is worth paying for fresh routing at all.
    paired=defaultdict(dict)
    for d in decisions:
        if d["policy"] in ("structural_beam","random_beam"):
            paired[(d["circuit_id"],d["original_sig"],d["repetition"])][d["policy"]]=d
    if any(len(v)!=2 for v in paired.values()):raise AssertionError("Missing paired policy")
    jwrite(out/"search_diagnostic_summary.json",dict(schema=SCHEMA,
         train_only=True,n_originals=len(origins),n_paired_runs=len(paired),
         same_selected_plan=sum(v["structural_beam"]["pick_sig"]==v["random_beam"]["pick_sig"] for v in paired.values()),
         structural_score_better=sum((v["structural_beam"]["selected_unique_edges"],-v["structural_beam"]["selected_hhi"]) <
              (v["random_beam"]["selected_unique_edges"],-v["random_beam"]["selected_hhi"])
              for v in paired.values()),
         both_no_repair=sum(all(d["pick_sig"]==d["original_sig"] for d in v.values()) for v in paired.values()),
         equal_n_valid_scored=sum(v["structural_beam"]["n_valid_scored"]==v["random_beam"]["n_valid_scored"] for v in paired.values()),
         n_unique_final_plan_plus_original=len(selected_plans),
         note="Search-only, not routed cost. If most plans identical or no structural changes, postpone routing and redesign the heuristic."))
    jwrite(out/"search_manifest.json",dict(schema=SCHEMA,status="search_only_no_routing",train_dir=str(Path(args.train_dir).resolve()),
        split_ledger=str(Path(args.split_ledger).resolve()),
        qasm_paths={o["circuit_id"]:o["qasm_path"] for o in origins},
        search_config=dict(seed=args.seed,repetitions=args.repetitions,budget=args.budget,
             beam_width=args.beam_width,max_depth=args.max_depth,raw_budget=args.raw_budget,
             per_op_keep=args.per_op_keep,
             max_attempts_per_op=args.max_attempts_per_op,operator_list=list(OPS)),
        n_originals=len(origins),n_decisions=len(decisions),n_unique_routing_plans=len(selected_plans),
        qiskit_version=qiskit.__version__,
        scoring="quotient_unique_edges ASC, edge_weight_HHI DESC, max_degree ASC; strict improvement to repair",
        note="ALL INPUT TRAIN only. No routing or labels read during search. RandomBeam shares final structural selector; RandomUniform uses random-beam visited pool."))
    print(f"[PASS] search only: {len(origins)} Originals, {len(decisions)} decisions, {len(selected_plans)} unique selected+original plans; ROUTING NOT RUN")


ROUTE_FIELDS=["circuit_id","source","filename","sig","seed","swap_count","routed_depth","routing_time_sec","status","error"]

def stage_route(args):
    out=Path(args.out_dir)
    meta=json.loads((out/"search_manifest.json").read_text(encoding="utf-8"))
    if meta.get("schema")!=SCHEMA:raise RuntimeError("Incompatible search output")
    seeds=[int(s) for s in args.seeds.split(",") if s.strip()]
    if len(seeds)<2 or len(set(seeds))!=len(seeds) or any(s<300 for s in seeds):
        raise ValueError("Use fresh unique seeds >=300, at least 2. Existing training/heldout seeds are prohibited.")
    lock=dict(schema=SCHEMA, seed_set=seeds,router="router_template.py heavy_hex(7) sabre v0.45.3",
             gidnet_source=str(Path(args.gidnet_file).resolve()))
    lockpath=out/"routing_protocol_lock.json"
    if lockpath.exists() and json.loads(lockpath.read_text(encoding="utf-8")) != lock:
        raise RuntimeError("Routing protocol lock differs; use same seed set/source or a new output dir")
    plans=read_csv(out/"selected_plan_catalog.csv")
    jobs={(p["circuit_id"],p["sig"],s) for p in plans for s in seeds}
    existing=read_csv(out/"new_seed_routing_results.csv") if (out/"new_seed_routing_results.csv").exists() else []
    found={}
    for r in existing:
        key=(r["circuit_id"],r["sig"],int(r["seed"]))
        if key in found:raise RuntimeError(f"Duplicate routing job: {key}")
        if key not in jobs:raise RuntimeError(f"Unexpected routing result: {key}")
        found[key]=r
    left=sorted(jobs-set(found))
    print(f"[route] existing={len(found)}, remaining={len(left)} (only final selected and original plans)")
    if not left:return
    # Reuse EXACT already-tested patched compiler and router from previous project.
    from build_multistep_pilot_validity_aware import compact_active,_load_local_gidnet
    import sys,importlib
    from qiskit import QuantumCircuit
    from reuse_validity import ReuseValidityChecker
    from reuse_adversarial import reference_static_feasible
    _load_local_gidnet(Path(args.gidnet_file).resolve(),Path(args.utils_file).resolve())
    adapter=importlib.import_module("experiment3_pipeline_adapter")
    router=importlib.import_module("router_template")
    if adapter.GidNET is not sys.modules["gidnet.qubitreuse"].GidNET:
        raise RuntimeError("GidNET adapter loaded different compiler")
    if not lockpath.exists():jwrite(lockpath,lock)
    byplan={(p["circuit_id"],p["sig"]):p for p in plans}
    cache={}
    # Compile once per unique selected plan.
    compiled={}
    count=0
    for cid,sig,seed in left:
        if args.max_new_jobs and count>=args.max_new_jobs:break
        p=byplan[(cid,sig)]
        if cid not in cache:
            qc,_=compact_active(QuantumCircuit.from_qasm_file(meta["qasm_paths"][cid]))
            ck=ReuseValidityChecker(qc)
            if ck.unsupported is not None:raise RuntimeError(f"Unsupported QASM: {cid}")
            cache[cid]=(qc,ck)
        qc,ck=cache[cid]
        if (cid,sig) not in compiled:
            plan=tuple(sorted(tuple(x) for x in json.loads(p["plan_json"])))
            if signature(plan)!=sig or len(plan)!=int(p["active_width"]):
                raise AssertionError("Corrupt plan catalog")
            if not ck.validate(plan,target_width=len(plan)).is_valid or not reference_static_feasible(qc,plan,len(plan)):
                raise AssertionError("Invalid selected plan before routing")
            dyn=adapter.build_dynamic_circuit(qc,plan)
            norm=adapter.normalize_for_routing(dyn)
            if dyn.num_qubits!=len(plan) or norm.num_qubits!=len(plan):
                raise AssertionError("Physical width changed")
            compiled[(cid,sig)]=norm
        began=time.perf_counter()
        try:
            result=router.route(compiled[(cid,sig)],seed)
            record=dict(circuit_id=cid,source=p["source"],filename=p["filename"],sig=sig,seed=seed,
                        swap_count=float(result["swap_count"]),routed_depth=float(result["routed_depth"]),
                        routing_time_sec=time.perf_counter()-began,status="ok",error="")
        except Exception as e:
            record=dict(circuit_id=cid,source=p["source"],filename=p["filename"],sig=sig,seed=seed,
                        swap_count="",routed_depth="",routing_time_sec=time.perf_counter()-began,
                        status="error",error=f"{type(e).__name__}: {e}")
        append_csv(out/"new_seed_routing_results.csv",record,ROUTE_FIELDS)
        count+=1
        if count%20==0:print(f"[route] new jobs {count}/{len(left)} last={cid} {sig} seed={seed} {record['status']}",flush=True)
    print(f"[route] completed {count} jobs; resumable (total needed {len(jobs)})")


def stage_analyze(args):
    out=Path(args.out_dir)
    meta=json.loads((out/"search_manifest.json").read_text(encoding="utf-8"))
    if meta.get("schema")!=SCHEMA:raise ValueError("Incompatible data")
    lock=json.loads((out/"routing_protocol_lock.json").read_text(encoding="utf-8"))
    seeds=lock["seed_set"]
    plans=read_csv(out/"selected_plan_catalog.csv")
    results=read_csv(out/"new_seed_routing_results.csv")
    expected={(r["circuit_id"],r["sig"],int(s)) for r in plans for s in seeds}
    rows={}
    for r in results:
        k=(r["circuit_id"],r["sig"],int(r["seed"]))
        if k in rows:raise RuntimeError(f"Duplicated route key: {k}")
        rows[k]=r
    if set(rows)!=expected:
        raise RuntimeError(f"Routing incomplete: missing={len(expected-set(rows))}, extraneous={len(set(rows)-expected)}")
    if any(r["status"]!="ok" for r in results):
        raise RuntimeError(f"Routing failures: {sum(r['status']!='ok' for r in results)}; do not silently ignore")
    values={}
    for (cid,sig) in {(r["circuit_id"],r["sig"]) for r in plans}:
        rs=[rows[(cid,sig,s)] for s in seeds]
        values[cid,sig]=(sum(float(x["swap_count"]) for x in rs)/len(seeds),
                         sum(float(x["routed_depth"]) for x in rs)/len(seeds))
    decisions=read_csv(out/"search_decisions.csv")
    evaluated=[]
    for d in decisions:
        cid,org,pick=d["circuit_id"],d["original_sig"],d["pick_sig"]
        base=values[cid,org]; val=values[cid,pick]
        evaluated.append({**d,"original_swap_mean":base[0],"selected_swap_mean":val[0],
            "swap_gain_abs":base[0]-val[0],
            "swap_gain_pct":(100*(base[0]-val[0])/base[0] if base[0]>0 else ""),
            "original_depth_mean":base[1],"selected_depth_mean":val[1],
            "depth_gain_abs":base[1]-val[1]})
    fields=list(evaluated[0])
    write_csv(out/"evaluated_decisions.csv",evaluated,fields)
    # Families are analysis units; repeated Original/rep not independent.
    import statistics
    by_policy=defaultdict(list)
    for r in evaluated:by_policy[r["policy"]].append(r)
    summary=[]
    for policy, subset in sorted(by_policy.items()):
        byfamily=defaultdict(list)
        for r in subset:byfamily[r["family_group"]].append(float(r["swap_gain_abs"]))
        family_means=[statistics.mean(v) for v in byfamily.values()]
        summary.append(dict(policy=policy,n_original_rep=len(subset),n_families=len(byfamily),
           n_repair=sum(r["original_sig"]!=r["pick_sig"] for r in subset),
           n_harm=sum(float(r["swap_gain_abs"])<0 for r in subset),
           n_better=sum(float(r["swap_gain_abs"])>0 for r in subset),
           mean_swap_gain_abs=statistics.mean(float(r["swap_gain_abs"]) for r in subset),
           mean_family_swap_gain_abs=statistics.mean(family_means),
           mean_swap_gain_pct_defined=statistics.mean(float(r["swap_gain_pct"]) for r in subset if r["swap_gain_pct"]!="") if any(r["swap_gain_pct"]!="" for r in subset) else "",
           mean_depth_change_abs=statistics.mean(float(r["depth_gain_abs"]) for r in subset),
           mean_valid_evaluated=statistics.mean(int(r["n_valid_scored"]) for r in subset),
           mean_raw_attempts=statistics.mean(int(r["raw_mutation_attempts"]) for r in subset),
           mean_search_time_sec=statistics.mean(float(r["score_time_sec"])+float(r["generate_time_sec"]) for r in subset)))
    write_csv(out/"method_comparison.csv",summary,list(summary[0]))
    paired={}
    idx={(r["circuit_id"],r["original_sig"],r["repetition"],r["policy"]):r for r in evaluated}
    for key in {(x[0],x[1],x[2]) for x in idx}:
        a=idx[key+("structural_beam",)]; b=idx[key+("random_beam",)]
        paired[key]=float(a["swap_gain_abs"])-float(b["swap_gain_abs"])
    deltas=list(paired.values())
    matched_keys=[k for k in paired if int(idx[k+("structural_beam",)]["n_valid_scored"])==int(idx[k+("random_beam",)]["n_valid_scored"])]
    equality=len(matched_keys)
    family_deltas=defaultdict(list)
    for k,delta in paired.items():
        family_deltas[idx[k+("structural_beam",)]["family_group"]].append(delta)
    jwrite(out/"analysis_summary.json",dict(schema=SCHEMA,train_only=True,n_paired_runs=len(paired),
       matched_valid_budget_runs=equality,unmatched_valid_budget_runs=len(paired)-equality,
       mean_paired_swap_gain_difference=sum(deltas)/len(deltas),
       matched_only_mean_paired_swap_gain_difference=(sum(paired[k] for k in matched_keys)/equality if equality else None),
       mean_of_family_paired_swap_gain_difference=statistics.mean(statistics.mean(d) for d in family_deltas.values()),
       structural_wins=sum(v>0 for v in deltas),random_wins=sum(v<0 for v in deltas),ties=sum(v==0 for v in deltas),
       routing_seeds=seeds,note="Exploratory on TRAIN circuits. This evaluates objective-independent, topology-agnostic structural search with fresh routing labels. No final-test claim. Control proposal attempt counts are reported, not always equal. Depth is secondary only."))
    for row in summary:print(f"{row['policy']:20s} mean_swap_gain={row['mean_swap_gain_abs']:+.4f} harm={row['n_harm']} mean_valid={row['mean_valid_evaluated']:.1f}")
    print(f"[PASS] completed analysis {len(evaluated)} decisions, {len(paired)} paired search runs; matched valid budget {equality}/{len(paired)}")


def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,description=__doc__)
    ap.add_argument("stage",choices=("search","route","analyze"))
    ap.add_argument("--train-dir",type=Path,default=Path("ml_train_prepare_v1"))
    ap.add_argument("--split-ledger",type=Path,default=Path("ml_split_v1/split_ledger.csv"))
    ap.add_argument("--out-dir",type=Path,default=Path("ml_structural_beam_v1"))
    ap.add_argument("--circuits",default="",help="Optional comma-separated TRAIN circuit_id filter")
    ap.add_argument("--max-circuits",type=int,default=0)
    ap.add_argument("--max-originals",type=int,default=0)
    ap.add_argument("--beam-width",type=int,default=5)
    ap.add_argument("--max-depth",type=int,default=5)
    ap.add_argument("--budget",type=int,default=120)
    ap.add_argument("--raw-budget",type=int,default=2400,help="Hard cap on attempted raw mutations per beam run")
    ap.add_argument("--max-attempts-per-op",type=int,default=180)
    ap.add_argument("--per-op-keep",type=int,default=10)
    ap.add_argument("--repetitions",type=int,default=3)
    ap.add_argument("--seed",type=int,default=20261104)
    ap.add_argument("--verify-fast-survivors",action="store_true")
    ap.add_argument("--seeds",default="300,301,302,303,304")
    ap.add_argument("--gidnet-file",type=Path,default=Path("qubitreuse.py"))
    ap.add_argument("--utils-file",type=Path,default=Path("utils.py"))
    ap.add_argument("--max-new-jobs",type=int,default=0)
    a=ap.parse_args()
    if a.stage=="search":stage_search(a)
    if a.stage=="route":stage_route(a)
    if a.stage=="analyze":stage_analyze(a)

if __name__=="__main__":main()
