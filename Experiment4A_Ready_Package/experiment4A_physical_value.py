#!/usr/bin/env python3
"""Experiment 4A: physical value of expanding one-step qubit-reuse neighborhoods.

Important scientific scope:
  * Input candidate validity is STATIC; this script does NOT prove dynamic semantics.
  * This is an exhaustive one-step opportunity/oracle study, NOT equal-budget repair.
  * Comparisons are conditional on fixed-width, original circuit, transpilation settings.
  * Reuses Experiment-3-style heavy-hex7/SABRE seed ensemble by default, but the
    exact Experiment-3 compilation implementation MUST be matched before claiming
    direct numerical comparability with Experiment 3. An optional adapter hook is
    provided for that purpose.

Stages:
  prepare: verify/full-coverage input, enumerate/deduplicate valid candidates.
  explore: route original and all valid unique candidates with training SABRE seeds;
           choose best per original plan for SWAP and routed depth, independently;
           freeze choices, report their mean performance.
  validate: route only frozen winners plus originals using independent seeds;
            output paired held-out comparisons.

Requires Python with pandas; explore/validate also require compatible Qiskit 0.45.x,
project's patched gidnet package, GidNET dependencies, and original QASM files.

Example:
  python experiment4A_physical_value.py prepare \
    --mutations adversarial_mutation_cases.csv \
    --audit width_and_original_plan_audit.csv

  python experiment4A_physical_value.py explore \
    --mutations adversarial_mutation_cases.csv \
    --audit width_and_original_plan_audit.csv \
    --plans experiment2_results/experiment2_plan_features.csv \
    --bench-dir ../CaQR-main/benchmarks

  python experiment4A_physical_value.py validate [the same paths]

Authoring note: do not change inclusion or routing choices after looking at hold-out.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import pandas as pd

OPS = ("suffix_swap", "chain_transfer", "node_swap_reconnect", "segment_reverse")
GROUP_OPS = {
    "suffix_only": frozenset(["suffix_swap"]),
    "suffix_plus_chain": frozenset(["suffix_swap", "chain_transfer"]),
    "all_operators": frozenset(OPS),
}
GROUP_ORDER = tuple(GROUP_OPS)
OBJECTIVES = ("swap_count", "routed_depth")
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_HOLDOUT = tuple(range(100, 130))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("prepare", "explore", "validate"))
    p.add_argument("--mutations", type=Path, default=Path("adversarial_mutation_cases.csv"))
    p.add_argument("--audit", type=Path, default=Path("width_and_original_plan_audit.csv"))
    p.add_argument("--plans", type=Path, default=None, help="Experiment-2 CSV with circuit, plan_id, width, reuse_chains; required for routing")
    p.add_argument("--bench-dir", type=Path, default=None, help="Directory holding QASM originals; required for routing")
    p.add_argument("--out", type=Path, default=Path("experiment4a_results"))
    p.add_argument("--adapter", type=Path, default=None,
                   help="Optional .py module with build_dynamic_circuit(qc, chains) and normalize_for_routing(qc) functions")
    p.add_argument("--train-seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--holdout-seeds", type=int, nargs="+", default=list(DEFAULT_HOLDOUT))
    p.add_argument("--depth-cap-percent", type=float, default=5.0)
    p.add_argument("--max-new-routes", type=int, default=None,
                   help="Smoke-test only: stop after writing this many new routes; no summaries until all routes exist")
    p.add_argument("--retry-errors", action="store_true")
    return p.parse_args(argv)


def req(ok: bool, msg: str):
    if not ok:
        raise ValueError(msg)


def sha(s: str):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:20]


def chains_of(value):
    x = ast.literal_eval(value) if isinstance(value, str) else value
    req(isinstance(x, (list, tuple)) and len(x) > 0, "Empty or malformed chains: " + str(value)[:100])
    chains = tuple(tuple(int(k) for k in chain) for chain in x)
    req(all(len(c) > 0 for c in chains), "Empty reuse chain")
    req(all(k >= 0 for c in chains for k in c), "Negative qubit index")
    flat = [k for c in chains for k in c]
    req(len(flat) == len(set(flat)), "Repeated original qubit index in reuse plan")
    return tuple(sorted(chains))  # Permuting whole chains does not alter a plan.


def truth(series):
    return series.astype(str).str.strip().str.lower().eq("true")


def load_table(path: Path):
    if not path.is_file():
        raise FileNotFoundError("Required CSV not found: " + str(path))
    return pd.read_csv(path, dtype={"plan_id": str})


def safe_save(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_inventory(args):
    """Build candidate and membership tables, with strong completeness checks."""
    raw = load_table(args.mutations)
    audit = load_table(args.audit)
    required_m = {"circuit", "plan_id", "operator", "candidate_chains", "checker_is_valid",
                  "reference_is_valid", "oracle_agrees", "preserved_width", "preserved_partition"}
    required_a = {"circuit", "plan_id", "saved_width", "checker_is_valid", "reference_is_valid"}
    req(required_m <= set(raw.columns), "Missing mutation columns: " + str(required_m - set(raw.columns)))
    req(required_a <= set(audit.columns), "Missing audit columns: " + str(required_a - set(audit.columns)))
    req(not audit.duplicated(["circuit", "plan_id"]).any(), "Duplicate audit plan keys")
    req(truth(audit.checker_is_valid).all() and truth(audit.reference_is_valid).all(),
        "Original plan(s) failed static validity; stop")
    req(set(raw.operator.unique()) <= set(OPS), "Unknown operator(s) in mutation CSV")
    req(truth(raw.oracle_agrees).all() and truth(raw.preserved_width).all() and truth(raw.preserved_partition).all(),
        "Mismatch/partition/width error in full-coverage input; stop")
    req((truth(raw.checker_is_valid) == truth(raw.reference_is_valid)).all(),
        "Independent validity oracle disagreement")
    if "candidate_signature" in raw.columns:
        expected = raw.candidate_chains.map(lambda x: repr(chains_of(x)))
        req((expected == raw.candidate_signature).all(),
            "Candidate chain/signature disagreement: source CSV may be corrupted")
    keys = set(zip(audit.circuit, audit.plan_id))
    req(set(zip(raw.circuit, raw.plan_id)) == keys, "Audit/mutations do not have the same 50 source keys")

    audit_map = {(r.circuit, r.plan_id): int(r.saved_width) for r in audit.itertuples()}
    train = sorted(set(args.train_seeds))
    holdout = sorted(set(args.holdout_seeds))
    req(len(train) == len(args.train_seeds) and len(holdout) == len(args.holdout_seeds), "Duplicate seeds")
    req(not set(train).intersection(holdout), "Training/held-out seed sets must be disjoint")

    orig_map = {}
    if args.plans is not None:
        source = load_table(args.plans)
        req({"circuit", "plan_id", "width", "reuse_chains"} <= set(source.columns),
            "Plans file must contain circuit, plan_id, width, reuse_chains")
        req(not source.duplicated(["circuit", "plan_id"]).any(), "Duplicate keys in original plans CSV")
        filtered = source.set_index(["circuit", "plan_id"])
        req(keys <= set(filtered.index), "Original plan CSV is missing full-coverage source plans")
        for key in sorted(keys):
            r = filtered.loc[key]
            v = chains_of(r.reuse_chains)
            width = audit_map[key]
            req(int(r.width) == width == len(v), f"Width discrepancy in original plan {key}")
            orig_map[key] = v

    # One canonical candidate is routed once per circuit, regardless of operator
    # aliases or number of source plans that generate it.
    inventory = {}
    memberships = {}
    valid = raw.loc[truth(raw.checker_is_valid)].copy()
    for row in valid.itertuples(index=False):
        width = audit_map[(row.circuit, row.plan_id)]
        cand = chains_of(row.candidate_chains)
        req(len(cand) == width, "Unexpected candidate chain count " + str((row.circuit, row.plan_id)))
        identity = sha(f"{row.circuit}|{width}|{repr(cand)}")
        key = (str(row.circuit), identity)
        item = {"circuit": row.circuit, "candidate_key": identity, "width": width,
                "reuse_chains": repr(cand)}
        if key in inventory:
            req(inventory[key] == item, "Candidate key collision or inconsistent width")
        inventory[key] = item
        src_key = (row.circuit, row.plan_id, identity)
        memberships.setdefault(src_key, set()).add(row.operator)

    if orig_map:
        for (circ, pid), chains in orig_map.items():
            width = audit_map[(circ, pid)]
            identity = sha(f"{circ}|{width}|{repr(chains)}")
            key = (circ, identity)
            item = {"circuit": circ, "candidate_key": identity, "width": width,
                    "reuse_chains": repr(chains)}
            if key in inventory:
                req(inventory[key] == item, "Original candidate conflict")
            inventory[key] = item
            memberships.setdefault((circ, pid, identity), set()).add("original")

    catalog = pd.DataFrame([inventory[k] for k in sorted(inventory)])
    members = pd.DataFrame([
        {"circuit": circ, "plan_id": pid, "candidate_key": identity,
         "operators": ",".join(sorted(ops)), "is_original": "original" in ops,
         "in_suffix_only": bool(ops.intersection(GROUP_OPS["suffix_only"])),
         "in_suffix_plus_chain": bool(ops.intersection(GROUP_OPS["suffix_plus_chain"])),
         "in_all_operators": bool(ops.intersection(GROUP_OPS["all_operators"]))}
        for (circ, pid, identity), ops in sorted(memberships.items())
    ])
    candidates = members.loc[~members.is_original]
    # A mutation might recreate the original under a different chain order:
    # it is not a *new* neighbor. Exclude original identity from pools.
    if orig_map:
        originals = members.loc[members.is_original, ["circuit", "plan_id", "candidate_key"]]
        orig_set = set(map(tuple, originals.itertuples(index=False, name=None)))
        original_neighbor = candidates.apply(
            lambda r: (r.circuit, r.plan_id, r.candidate_key) in orig_set, axis=1)
        if original_neighbor.any():
            members = members.loc[~(members.index.isin(candidates[original_neighbor].index))].copy()
            candidates = members.loc[~members.is_original]
    counts = []
    for (circ, pid), g in candidates.groupby(["circuit", "plan_id"]):
        a = set(g.loc[g.in_suffix_only, "candidate_key"])
        b = set(g.loc[g.in_suffix_plus_chain, "candidate_key"])
        c = set(g.loc[g.in_all_operators, "candidate_key"])
        counts.append({"circuit": circ, "plan_id": pid, "suffix_count": len(a),
                       "suffix_chain_count": len(b), "all_count": len(c),
                       "new_from_chain": len(b-a), "new_from_other": len(c-b)})
    coverage = pd.DataFrame(counts)
    args.out.mkdir(parents=True, exist_ok=True)
    safe_save(catalog, args.out/"candidate_catalog.csv")
    safe_save(members, args.out/"source_memberships.csv")
    safe_save(coverage, args.out/"coverage_per_source.csv")
    inventory_report = {
        "input_mutations": len(raw), "statically_valid_mutations": len(valid),
        "sources": len(audit), "circuits": int(audit.circuit.nunique()),
        "all_distinct_physical_plans_including_originals_if_supplied": len(catalog),
        "unique_valid_neighborhoods_per_source_total": int(coverage.all_count.sum()),
        "unique_suffix_per_source_total": int(coverage.suffix_count.sum()),
        "extra_from_chain_total": int(coverage.new_from_chain.sum()),
        "extra_from_other_total": int(coverage.new_from_other.sum()),
        "originals_included": bool(orig_map), "train_seeds": train, "holdout_seeds": holdout,
        "no_semantic_equivalence_claim": True,
    }
    (args.out/"inventory_report.json").write_text(json.dumps(inventory_report, indent=2), encoding="utf8")
    return catalog, members, coverage, inventory_report


def project_builder(adapter_file: Path | None):
    """Inject project's exact Experiment-3 builder, or use patched GidNET default.

    Adapter contract: build_dynamic_circuit(original_qasm_qc, chain_tuples) -> QC
                      normalize_for_routing(dynamic_qc) -> QC  [optional]
    """
    if adapter_file is not None:
        req(adapter_file.is_file(), "Adapter file missing: " + str(adapter_file))
        spec = importlib.util.spec_from_file_location("experiment4a_adapter", str(adapter_file.resolve()))
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        req(callable(getattr(module,"build_dynamic_circuit",None)),
            "Adapter must implement build_dynamic_circuit(original_qc, chains)")
        builder = module.build_dynamic_circuit
        normalizer = getattr(module, "normalize_for_routing", lambda qc: qc)
        req(callable(normalizer), "normalize_for_routing must be callable")
        return builder, normalizer, "custom adapter: " + str(adapter_file)

    try:
        from gidnet.qubitreuse import GidNET
        from gidnet.reset_lowering import lower_reset_to_measure_x
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import patched gidnet modules. Run from the GidNET repo root with its "
            "environment enabled, or provide --adapter with the exact Experiment-3 compiler. "
            f"Details: {exc}") from exc

    def build_dynamic_circuit(qc, chains):
        runner = GidNET(qc.copy())
        # Standard method normally *generates* reuse sequences. Freeze the exact
        # one-step neighbor instead: reuse GidNET's own insertion/topological remap.
        runner.compute_optimized_reuse_sequences = lambda iterations: [list(c) for c in chains]
        return runner.compile_to_dynamic_circuit(iterations=0)

    return build_dynamic_circuit, lower_reset_to_measure_x, "patched GidNET fixed-sequence + active reset lowering"


def load_existing_results(path: Path, fields):
    if path.is_file():
        d = pd.read_csv(path, dtype={"candidate_key": str, "circuit": str})
        req(set(fields) <= set(d.columns), f"Existing routing CSV missing columns: {path}")
        return d.to_dict("records")
    return []


def save_route_rows(path: Path, rows):
    safe_save(pd.DataFrame(rows), path)


def route_selected(args, catalog: pd.DataFrame, selected: set, phase: str, seeds: list[int]):
    """Physical runs, cached per (circuit, candidate, seed, phase)."""
    from qiskit import qasm2, transpile
    from qiskit.transpiler import CouplingMap
    import qiskit

    req(args.bench_dir is not None and args.bench_dir.is_dir(), "--bench-dir must be an existing directory")
    builder, normalize, compile_tag = project_builder(args.adapter)
    coupling = CouplingMap.from_heavy_hex(7)
    cp = args.out / ("routes_train.csv" if phase == "train" else "routes_holdout.csv")
    rows = load_existing_results(cp, {"circuit","candidate_key","seed","phase","status",
                                       "swap_count", "routed_depth", "logical_depth","compile_seconds"})
    prev = {(r["circuit"], r["candidate_key"], int(r["seed"])):r for r in rows}
    old_meta = args.out / "routing_settings.json"
    subset = catalog.loc[catalog.apply(lambda r:(r.circuit,r.candidate_key) in selected,axis=1)]
    settings = {
        "coupling_map": "CouplingMap.from_heavy_hex(7)", "layout_method":"sabre", "routing_method":"sabre",
        "optimization_level":1, "qiskit_version":qiskit.__version__, "compiler":compile_tag,
        "mutations_sha256": hashlib.sha256(args.mutations.read_bytes()).hexdigest(),
        "audit_sha256": hashlib.sha256(args.audit.read_bytes()).hexdigest(),
        "plans_sha256": hashlib.sha256(args.plans.read_bytes()).hexdigest() if args.plans else None,
        "depth_cap_percent": args.depth_cap_percent,
        "train_seeds":sorted(set(args.train_seeds)), "holdout_seeds":sorted(set(args.holdout_seeds)),
        "canonicalize_whole_chain_order": True,
        "semantic_validity":"NOT tested; static feasibility only",
        "qasm_sha256": {name:hashlib.sha256((args.bench_dir/name).read_bytes()).hexdigest()
                         for name in sorted(catalog.circuit.unique())},
    }
    # Abort if existing raw results were computed with a different experiment config.
    if old_meta.is_file():
        prior = json.loads(old_meta.read_text(encoding="utf8"))
        req(settings==prior, "Routing settings/input files changed. Choose a NEW --out directory; do not mix experiments")
    else:
        old_meta.write_text(json.dumps(settings, indent=2), encoding="utf8")

    print(f"{phase}: {len(subset)} distinct plans × {len(seeds)} seeds; cached entries={len(prev)}", flush=True)
    qasms = {}
    dynamic = {}
    added = 0
    for item in subset.itertuples(index=False):
        ck = (item.circuit,item.candidate_key)
        todo = [seed for seed in seeds if (item.circuit,item.candidate_key,seed) not in prev or
                (args.retry_errors and str(prev[(item.circuit,item.candidate_key,seed)]["status"])!="ok")]
        if not todo:
            continue
        qc = qasms.get(item.circuit)
        if qc is None:
            file = args.bench_dir / item.circuit
            req(file.is_file(), "Original QASM not found: " + str(file))
            qc = qasm2.load(str(file))
            qasms[item.circuit] = qc
        chains = chains_of(item.reuse_chains)
        req(sorted(q for c in chains for q in c)==list(range(qc.num_qubits)),
            "Candidate does not partition original circuit qubits: " + str(ck))
        try:
            raw_dynamic = builder(qc, chains)
            req(raw_dynamic.num_qubits == int(item.width), f"Compiled width != {item.width}, {ck}")
            prepared = normalize(raw_dynamic)
            req(prepared.num_qubits == int(item.width), "Reset lowering changed width")
            logdepth = int(prepared.depth())
        except Exception as exc:
            raise RuntimeError(f"Fixed-plan compilation failed for {ck}; refusing to silently skip: {exc}") from exc

        for seed in todo:
            start = time.perf_counter()
            try:
                routed = transpile(
                    prepared, coupling_map=coupling,
                    layout_method="sabre", routing_method="sabre",
                    optimization_level=1, seed_transpiler=int(seed),
                )
                row = {"phase":phase,"circuit":item.circuit,"candidate_key":item.candidate_key,
                       "seed":seed,"status":"ok","swap_count":int(routed.count_ops().get("swap",0)),
                       "routed_depth":int(routed.depth()), "logical_depth":logdepth,
                       "compile_seconds": round(time.perf_counter()-start,5),"error":""}
            except Exception as exc:
                row = {"phase":phase,"circuit":item.circuit,"candidate_key":item.candidate_key,
                       "seed":seed,"status":"error", "swap_count":float("nan"),
                       "routed_depth":float("nan"),"logical_depth":logdepth,
                       "compile_seconds":round(time.perf_counter()-start,5),"error":repr(exc)[:1000]}
            key = (item.circuit,item.candidate_key,seed)
            prev[key] = row
            rows = list(prev.values())
            save_route_rows(cp,rows)  # crash-safe checkpoint after each transpile.
            added += 1
            if row["status"]=="error":
                print("TRANSPILE ERROR",ck,"seed",seed,row["error"],file=sys.stderr,flush=True)
            if added % 25 == 0:
                print(f"  routed/checkpointed {added} new evaluations",flush=True)
            if args.max_new_routes and added >= args.max_new_routes:
                print("Smoke-test stopping; resume without --max-new-routes to finish.")
                return False
    return True


def strict_expected_route_table(path: Path, catalog: pd.DataFrame, chosen: set, seeds: list[int]):
    route_df = pd.read_csv(path, dtype={"candidate_key": str})
    ids = {(c,k,int(s)) for c,k in chosen for s in seeds}
    selected = route_df.loc[route_df.apply(
        lambda r:(r.circuit,r.candidate_key,int(r.seed)) in ids,axis=1)].copy()
    keys = list(zip(selected.circuit, selected.candidate_key, selected.seed.astype(int)))
    req(len(keys)==len(set(keys)), "Duplicate route keys found in "+str(path))
    req(set(keys)==ids, f"Incomplete routing data in {path}: {len(ids)-len(set(keys))} missing")
    bad = selected.loc[selected.status!="ok"]
    req(bad.empty, f"{len(bad)} physical compile failures; inspect {path}; rerun --retry-errors")
    for metric in OBJECTIVES:
        req(pd.to_numeric(selected[metric],errors='coerce').notna().all(), "Missing metric: "+metric)
    return selected


def frozen_best_explore(args, members:pd.DataFrame, routes:pd.DataFrame):
    means = routes.groupby(["circuit","candidate_key"],as_index=False).agg(
        mean_swap_count=("swap_count","mean"), mean_routed_depth=("routed_depth","mean"),
        logical_depth=("logical_depth","first"))
    safe_save(means,args.out/"candidate_routing_means.csv")
    look = {(r.circuit,r.candidate_key):r for r in means.itertuples()}
    output = []
    for (circ,pid),g in members.groupby(["circuit","plan_id"]):
        origins = g[g.is_original]
        req(len(origins)==1, "Exactly one original plan required for "+str((circ,pid)))
        original_key = origins.iloc[0].candidate_key
        src_m = look[(circ,original_key)]
        orig_logdepth = src_m.logical_depth
        for obj in OBJECTIVES:
            metric = 'mean_'+obj
            for pool in GROUP_ORDER:
                neighbors = g.loc[g["in_"+pool] & ~g.is_original]
                for capped in (False,True):
                    if capped:
                        if orig_logdepth==0:
                            filt = neighbors.loc[neighbors.candidate_key.map(
                                lambda key:look[(circ,key)].logical_depth <= 0)]
                        else:
                            cutoff = orig_logdepth*(1.0 + args.depth_cap_percent/100.)
                            filt = neighbors.loc[neighbors.candidate_key.map(
                                lambda key:look[(circ,key)].logical_depth <= cutoff + 1e-9)]
                    else:
                        filt = neighbors
                    if len(filt)==0:
                        # Report absent neighbors. Optimization can always return original.
                        winner = original_key
                        no_neighbor = True
                    else:
                        # Best NEIGHBOR, not automatically best-of-original. A separate
                        # best-of-original comparison is derived below.
                        winner = min(filt.candidate_key,
                                     key=lambda key:(float(getattr(look[(circ,key)],metric)),key))
                        no_neighbor = False
                    won = look[(circ,winner)]
                    orig_cost = float(getattr(src_m,metric))
                    neighbor_cost = float(getattr(won,metric))
                    best_or_orig = min(orig_cost, neighbor_cost)
                    chosen = winner if neighbor_cost < orig_cost else original_key
                    output.append({"selected_key":chosen,"circuit":circ,"plan_id":pid,"objective":obj,"pool":pool,
                                   "depth_cap": bool(capped), "original_key":original_key,
                                   "neighbor_key":winner,"no_valid_neighbor":no_neighbor,
                                   "original_mean":orig_cost,"best_neighbor_mean":neighbor_cost,
                                   "best_of_original_and_neighbor":best_or_orig,
                                   "gain_from_original_pct":(100.*(orig_cost-best_or_orig)/orig_cost
                                                             if orig_cost>0 else 0.),
                                   "logical_depth_original":int(orig_logdepth),
                                   "logical_depth_neighbor":int(won.logical_depth)})
    frozen = pd.DataFrame(output)
    safe_save(frozen,args.out/"frozen_winners.csv")
    comparisons=[]
    baseline = frozen.loc[frozen.pool=="suffix_only"].set_index(["circuit","plan_id","objective","depth_cap"])
    for r in frozen.itertuples(index=False):
        s = baseline.loc[(r.circuit,r.plan_id,r.objective,r.depth_cap)]
        # Improvement possible with the extra available operator pool.
        difference = float(s.best_of_original_and_neighbor)-r.best_of_original_and_neighbor
        comparisons.append({**r._asdict(),
            "extra_gain_over_suffix":round(difference,7),
            "strictly_better_than_suffix": bool(difference>1e-8),
            "best_suffix_mean":float(s.best_of_original_and_neighbor),
            "expanded_winner_not_suffix_neighbor":bool(
                r.neighbor_key not in set(members.loc[(members.circuit==r.circuit)&
                    (members.plan_id==r.plan_id)&members.in_suffix_only,"candidate_key"]))})
    df = pd.DataFrame(comparisons)
    safe_save(df,args.out/"source_opportunity_comparison.csv")
    summary = df.groupby(["objective","pool","depth_cap"],as_index=False).agg(
        source_count=("plan_id","count"),strict_beneficiaries=("strictly_better_than_suffix","sum"),
        mean_extra_gain=("extra_gain_over_suffix","mean"),
        mean_gain_vs_original_pct=("gain_from_original_pct","mean"),
        no_valid_neighbor=("no_valid_neighbor","sum"))
    safe_save(summary,args.out/"opportunity_summary.csv")
    return frozen


def validate_holdout(args,members, frozen, holdout):
    route_file = args.out/"routes_holdout.csv"
    # Use only the frozen winner chosen from the training seeds. Never reselect
    # based on hold-out results.
    rows=[]
    valid_ids=set(zip(holdout.circuit,holdout.candidate_key))
    for r in frozen.itertuples(index=False):
        req((r.circuit,r.original_key) in valid_ids and (r.circuit,r.selected_key) in valid_ids,
            "Missing held-out frozen choice")
        for seed in sorted(set(args.holdout_seeds)):
            org = holdout.loc[(holdout.circuit==r.circuit)&(holdout.candidate_key==r.original_key)&(holdout.seed==seed)]
            nb = holdout.loc[(holdout.circuit==r.circuit)&(holdout.candidate_key==r.selected_key)&(holdout.seed==seed)]
            req(len(org)==len(nb)==1,"Paired held-out row missing")
            obj = r.objective
            base = float(org.iloc[0][obj]); cost = float(nb.iloc[0][obj])
            rows.append({"circuit":r.circuit,"plan_id":r.plan_id,"objective":obj,"pool":r.pool,
                         "depth_cap":r.depth_cap,"seed":seed,"original_key":r.original_key,
                         "frozen_selected_key":r.selected_key,"original_value":base,
                         "neighbor_value":cost,"paired_gain":base-cost})
    paired = pd.DataFrame(rows)
    safe_save(paired,args.out/"fresh_seed_pairs.csv")
    agg = paired.groupby(["circuit","plan_id","objective","pool","depth_cap"],as_index=False).agg(
        original_heldout_mean=("original_value","mean"),
        neighbor_heldout_mean=("neighbor_value","mean"),
        mean_paired_gain_vs_original=("paired_gain","mean"),
        holdout_seed_count=("seed","nunique"))
    ref=agg.loc[agg.pool=="suffix_only", ["circuit","plan_id","objective","depth_cap","neighbor_heldout_mean"]].rename(
        columns={"neighbor_heldout_mean":"suffix_neighbor_heldout_mean"})
    agg=agg.merge(ref,on=["circuit","plan_id","objective","depth_cap"],validate="many_to_one")
    agg["heldout_gain_vs_frozen_suffix"] = agg["suffix_neighbor_heldout_mean"]-agg["neighbor_heldout_mean"]
    safe_save(agg,args.out/"fresh_seed_comparison.csv")
    return agg


def main():
    args = parse_args()
    catalog, members, coverage, inventory = build_inventory(args)
    print("Input audit:", json.dumps(inventory,indent=2),flush=True)
    if args.stage=="prepare":
        print("Prepared candidate_catalog.csv, source_memberships.csv, coverage_per_source.csv")
        if args.plans is None:
            print("Note: no original plan CSV provided. Prepare-only is complete; routing needs --plans and --bench-dir.")
        return
    req(args.plans is not None and args.bench_dir is not None,
        "Routing requires BOTH --plans <experiment2_plan_features.csv> and --bench-dir <original QASM directory>")
    originals=members.loc[members.is_original]
    req(len(originals)==inventory["sources"],"Original plans missing")
    training=list(args.train_seeds)
    holdout_seeds=list(args.holdout_seeds)
    chosen=set(map(tuple,catalog[["circuit","candidate_key"]].itertuples(index=False,name=None)))
    if args.stage=="explore":
        finished=route_selected(args,catalog,chosen,"train",training)
        if not finished: return
        training_df = strict_expected_route_table(args.out/"routes_train.csv",catalog,chosen,training)
        frozen_best_explore(args,members,training_df)
        print("Exploration complete; see opportunity_summary.csv and frozen_winners.csv")
    elif args.stage=="validate":
        require=args.out/"frozen_winners.csv"
        req(require.is_file(),"Run 'explore' fully before 'validate': missing frozen_winners.csv")
        frozen=pd.read_csv(require,dtype={"plan_id":str,"candidate_key":str,
                                               "original_key":str,"neighbor_key":str,"selected_key":str})
        needed={(c,k) for c,k in zip(frozen.circuit,frozen.original_key)} | {
            (c,k) for c,k in zip(frozen.circuit,frozen.selected_key)}
        finished=route_selected(args,catalog,needed,"holdout",holdout_seeds)
        if not finished:return
        held = strict_expected_route_table(args.out/"routes_holdout.csv",catalog,needed,holdout_seeds)
        validate_holdout(args,members,frozen,held)
        print("Held-out results complete; see fresh_seed_comparison.csv")


if __name__=="__main__":
    main()
