#!/usr/bin/env python3
"""Bounded, auditable multistep fixed-width reuse dataset pilot.

Stages are deliberately separate:
  prepare: no routing; generate sampled paths, validate, select candidates, extract 25 features.
  route:   explicit request only; compile with the Experiment-3 adapter and route 2 smoke seeds.
  labels:  compute sample means and separate SWAP/depth labels from completed route rows.

This is a DATA-PIPELINE SMOKE TEST, not a performance experiment or final training set.
Qiskit 0.45.3 and the supplied, patched GidNET implementation are required.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import random
import sys
import time
import traceback
import types
from collections import Counter, defaultdict
from pathlib import Path

CIRCUITS = [
    ("CaQR", "cnt3-5_179.qasm"),
    ("CaQR", "sym6_316.qasm"),
    ("CaQR", "multiply_n13.qasm"),
    ("QASMBench", "multiply_n13.qasm"),
    ("QASMBench", "gcm_h6.qasm"),
    ("QASMBench", "lpn_n5.qasm"),
]
OPS = ("suffix_swap", "chain_transfer")


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as h:
        return list(csv.DictReader(h))


def write_csv(path, rows, fields):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8-sig", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if (isinstance(v, float) and not math.isfinite(v)) else v)
                        for k, v in row.items()})


def json_out(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def canonical(plan):
    return tuple(sorted(tuple(int(q) for q in chain) for chain in plan))


def sig(plan):
    return hashlib.sha256(json.dumps([list(c) for c in canonical(plan)],
                                     separators=(",", ":")).encode()).hexdigest()[:24]


def circuit_id(source, filename):
    return source + "::" + filename


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def read_sources(dirs):
    circuits, originals = {}, defaultdict(list)
    for d in dirs:
        for row in read_csv(Path(d) / "circuit_audit.csv"):
            key = (row["source"], row["filename"])
            if key in circuits:
                raise ValueError(f"Duplicate circuit in audit dirs: {key}")
            circuits[key] = row
        for row in read_csv(Path(d) / "original_plans.csv"):
            originals[(row["source"], row["filename"])].append(row)
    return circuits, originals


def compact_active(qc):
    """Same semantics as reuse_feasibility_audit_fixed.py: retain terminal measures.

    No unsupported measurement-only wire may be silently dropped.
    """
    from qiskit import QuantumCircuit
    from reuse_validity import active_qubits_from_circuit
    active = sorted(active_qubits_from_circuit(qc))
    if not active:
        raise ValueError("No gate-active qubits")
    remap = {old: i for i, old in enumerate(active)}
    out = QuantumCircuit(len(active), qc.num_clbits)
    out.global_phase = qc.global_phase
    for item in qc.data:
        if item.operation.name == "barrier":
            continue
        qi = [qc.find_bit(q).index for q in item.qubits]
        if any(q not in remap for q in qi):
            raise ValueError("Cannot drop an inactive qubit referenced by an instruction/measurement")
        out.append(item.operation.copy(), [out.qubits[remap[q]] for q in qi],
                   [out.clbits[qc.find_bit(c).index] for c in item.clbits])
    return out, active


def load_feature_module(path):
    spec = importlib.util.spec_from_file_location("_pilot_feature_code", str(Path(path).resolve()))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load feature module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_checker(qc):
    from reuse_validity import ReuseValidityChecker
    ck = ReuseValidityChecker(qc)
    if ck.unsupported is not None:
        raise ValueError(f"Unsupported circuit: {ck.unsupported.reason}")
    return ck


def legal_neighbors(plan, ck, rng, *, max_attempts_per_op, per_op_keep):
    """Bounded valid one-step neighbors using the same supplied 2 operators.

    Operators are scanned deterministically and valid results shuffled with the local RNG.
    Capping attempts means sampling is NOT uniform over the full legal neighborhood.
    """
    from reuse_adversarial import OPERATORS
    found = set()
    result = []
    stats = Counter()
    for op in OPS:
        candidates = []
        for i, cand in enumerate(OPERATORS[op](plan)):
            if i >= max_attempts_per_op:
                break
            stats["attempted"] += 1
            if not cand or not all(cand):
                continue
            cand = canonical(cand)
            if cand == plan or cand in found:
                continue
            found.add(cand)
            if ck.validate(cand, target_width=len(plan)).is_valid:
                stats["valid"] += 1
                candidates.append((op, cand))
        rng.shuffle(candidates)
        result.extend(candidates[:per_op_keep])
    rng.shuffle(result)
    return result, stats


def discover(start, ck, *, seed, max_depth, max_frontier, max_attempts_per_op,
             per_op_keep, max_discovered):
    """Budgeted breadth-first exploration, not an exact neighborhood enumeration.

    'discovery_move_depth' records the sampled path length. It is not an
    independently proven minimum distance in the full state graph.
    """
    rng = random.Random(seed)
    original_sig = sig(start)
    observed = {original_sig: {"sig": original_sig, "plan": start,
                               "discovery_move_depth": 0, "parent_sig": "",
                               "operator": "original"}}
    frontier = [original_sig]
    counts = Counter()
    for depth in range(1, max_depth + 1):
        nxt = []
        for parent_sig in frontier:
            plan = observed[parent_sig]["plan"]
            suggestions, stats = legal_neighbors(plan, ck, rng,
                max_attempts_per_op=max_attempts_per_op, per_op_keep=per_op_keep)
            counts.update(stats)
            for op, cand in suggestions:
                s = sig(cand)
                if s in observed:
                    continue
                observed[s] = {"sig": s, "plan": cand,
                               "discovery_move_depth": depth,
                               "parent_sig": parent_sig, "operator": op}
                nxt.append(s)
                if len(observed) >= max_discovered + 1:
                    break
            if len(observed) >= max_discovered + 1:
                break
        counts[f"found_at_{depth}"] = len(nxt)
        if len(observed) >= max_discovered + 1 or not nxt:
            break
        rng.shuffle(nxt)
        frontier = nxt[:max_frontier]
    return observed, counts


def select_stratified(records, n, seed):
    """Deterministic approximate depth-balanced sample. Original is included separately."""
    rng = random.Random(seed)
    depth_groups = defaultdict(list)
    for r in records.values():
        if r["discovery_move_depth"] > 0:
            depth_groups[r["discovery_move_depth"]].append(r)
    for group in depth_groups.values():
        group.sort(key=lambda r: r["sig"])
        rng.shuffle(group)
    levels = sorted(depth_groups)
    chosen = []
    while len(chosen) < n and any(depth_groups.values()):
        for depth in levels:
            if depth_groups[depth]:
                chosen.append(depth_groups[depth].pop())
                if len(chosen) >= n:
                    break
    return chosen


def prepare(a):
    from qiskit import QuantumCircuit
    import qiskit
    if qiskit.__version__ != "0.45.3":
        raise RuntimeError(f"Pilot expects Qiskit 0.45.3, got {qiskit.__version__}")
    from reuse_adversarial import reference_static_feasible
    if a.max_depth < 1 or a.max_originals < 1 or a.max_candidates < 1:
        raise ValueError("Depth, source count and candidate limits must be positive")
    fmap = load_feature_module(a.feature_code)
    circuits, originals = read_sources(a.audit_dirs)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths, catalog = {}, {}
    discovered, selected, source_summary, issues, feature_rows, pairs = [], [], [], [], [], []
    chosen = CIRCUITS if not a.circuits else [tuple(x.split("::", 1)) for x in a.circuits.split(",")]
    if any(len(c) != 2 for c in chosen):
        raise ValueError("Use --circuits CaQR::foo.qasm,QASMBench::bar.qasm")
    for source, filename in chosen:
        cid = circuit_id(source, filename)
        try:
            key = (source, filename)
            audit = circuits[key]
            if audit["status"] != "screened":
                raise ValueError(f"Not a screened reuse circuit: status={audit['status']}")
            path = Path(audit["qasm_path"])
            if not path.is_file():
                raise FileNotFoundError(f"Original QASM unavailable: {path}")
            qc_original = QuantumCircuit.from_qasm_file(str(path))
            qc, old_active = compact_active(qc_original)
            ck = make_checker(qc)
            feature_qc, remap = fmap.prepare_gidnet_input(qc)
            if (feature_qc.num_qubits != qc.num_qubits or
                    any(old != new for old, new in remap.items())):
                raise ValueError(f"Unexpected feature/compile index mapping: {remap}")
            profiles, timing = fmap.build_interaction_profiles(feature_qc)
            candidates = []
            for row in sorted(originals[key], key=lambda r: r["plan_id"]):
                p = canonical(json.loads(row["chains_json"]))
                if len(p) >= qc.num_qubits:
                    continue  # must have reused at least one active qubit
                if sig(p) != row["plan_id"]:
                    raise ValueError(f"Source plan signature mismatch for {row['plan_id']}")
                if not ck.validate(p, target_width=len(p)).is_valid:
                    raise ValueError(f"Source plan failed checker: {row['plan_id']}")
                if not reference_static_feasible(qc, p, len(p)):
                    raise ValueError(f"Source plan failed independent static scheduler: {row['plan_id']}")
                candidates.append((row["plan_id"], p))
            if not candidates:
                raise ValueError("No reused Original Plan in audit output")
            paths[cid] = str(path)
            selected_by_sig = {}
            for original_sig, p in candidates[:a.max_originals]:
                observations, counts = discover(p, ck, seed=stable_seed(a.seed, cid, original_sig),
                    max_depth=a.max_depth, max_frontier=a.max_frontier,
                    max_attempts_per_op=a.max_attempts_per_op,
                    per_op_keep=a.per_op_keep, max_discovered=a.max_discovered)
                sampled = select_stratified(observations, a.max_candidates,
                                          stable_seed(a.seed, cid, original_sig, "selection"))
                keep = [observations[original_sig]] + sampled
                for r in observations.values():
                    rr = dict(circuit_id=cid, source=source, filename=filename,
                              original_sig=original_sig, sig=r["sig"],
                              parent_sig=r["parent_sig"],
                              discovery_move_depth=r["discovery_move_depth"],
                              operator=r["operator"])
                    discovered.append(rr)
                for r in keep:
                    p2 = r["plan"]
                    if len(p2) != len(p) or not ck.validate(p2, target_width=len(p)).is_valid:
                        raise AssertionError("Discovered candidate violates width or validity")
                    if not reference_static_feasible(qc, p2, len(p)):
                        raise AssertionError("Independent oracle rejected selected candidate")
                    row = dict(circuit_id=cid, source=source, filename=filename,
                               family=audit["family"], original_sig=original_sig,
                               sig=r["sig"], is_original=int(r["sig"] == original_sig),
                               discovery_move_depth=r["discovery_move_depth"],
                               parent_sig=r["parent_sig"], operator=r["operator"],
                               active_width=len(p), active_qubits=qc.num_qubits)
                    selected.append(row)
                    catalog[(cid, r["sig"])] = dict(circuit_id=cid, source=source,
                        filename=filename, sig=r["sig"], active_width=len(p),
                        plan_json=json.dumps([list(ch) for ch in p2], separators=(",", ":")))
                    selected_by_sig[r["sig"]] = p2
                source_summary.append(dict(circuit_id=cid, original_sig=original_sig,
                    width=len(p), observed_plans=len(observations),
                    selected_candidates=len(sampled), **dict(counts)))
            # Features are extracted per unique selected plan, then converted to pair deltas.
            local_feat = {}
            for signature, plan in selected_by_sig.items():
                local_feat[signature] = fmap.plan_features(feature_qc, profiles, timing, plan)
            # Handle plans that are selected for multiple source plans (dedup by CID+sig).
            for r in [v for v in selected if v["circuit_id"] == cid]:
                signature, original_sig = r["sig"], r["original_sig"]
                if signature not in local_feat:
                    # Could have been selected under a different original source only.
                    local_feat[signature] = fmap.plan_features(feature_qc, profiles, timing,
                                                canonical(json.loads(catalog[(cid, signature)]["plan_json"])))
                if original_sig not in local_feat:
                    local_feat[original_sig] = fmap.plan_features(feature_qc, profiles, timing,
                                                canonical(json.loads(catalog[(cid, original_sig)]["plan_json"])))
                delta = {}
                for k, v in local_feat[signature].items():
                    base = local_feat[original_sig][k]
                    delta["d_" + k] = float(v) - float(base) if (
                        v is not None and base is not None and math.isfinite(float(v))
                        and math.isfinite(float(base))) else float("nan")
                pairs.append({**r, **delta})
            for s, f in local_feat.items():
                feature_rows.append(dict(circuit_id=cid, sig=s, **f))
            print(f"[prepare] {cid}: {len(candidates[:a.max_originals])} original(s); "
                  f"{sum(r['selected_candidates'] for r in source_summary if r['circuit_id']==cid)} candidate(s); "
                  f"{len(set(x['sig'] for x in selected if x['circuit_id']==cid))} unique selected plans", flush=True)
        except Exception as e:
            issues.append({"circuit_id": cid, "error": f"{type(e).__name__}: {e}",
                           "trace": traceback.format_exc(limit=2)})
            print(f"[ERROR] {cid}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    if not selected:
        json_out(out / "prepare_errors.json", issues)
        raise RuntimeError("No pilot candidates generated. Inspect prepare_errors.json")
    # Remove any partial circuit output when an exception happened midway.
    failed_ids = {e["circuit_id"] for e in issues}
    selected = [r for r in selected if r["circuit_id"] not in failed_ids]
    discovered = [r for r in discovered if r["circuit_id"] not in failed_ids]
    source_summary = [r for r in source_summary if r["circuit_id"] not in failed_ids]
    pairs = [r for r in pairs if r["circuit_id"] not in failed_ids]
    feature_rows = [r for r in feature_rows if r["circuit_id"] not in failed_ids]
    catalog = {k:v for k,v in catalog.items() if k[0] not in failed_ids}
    paths = {k:v for k,v in paths.items() if k not in failed_ids}
    if not selected:
        json_out(out / "prepare_errors.json", issues)
        raise RuntimeError("All pilot circuits failed; inspect prepare_errors.json")
    selected.sort(key=lambda r: (r["circuit_id"], r["original_sig"], -r["is_original"], r["discovery_move_depth"], r["sig"]))
    discovered.sort(key=lambda r: (r["circuit_id"], r["original_sig"], r["discovery_move_depth"], r["sig"]))
    featkeys = list(feature_rows[0].keys())[2:] if feature_rows else []
    paircols = ["d_"+k for k in featkeys]
    selcols = ["circuit_id", "source", "filename", "family", "original_sig", "sig",
               "is_original", "discovery_move_depth", "parent_sig", "operator", "active_width", "active_qubits"]
    write_csv(out / "selected_candidates.csv", selected, selcols)
    write_csv(out / "discovery.csv", discovered, ["circuit_id", "source", "filename", "original_sig", "sig", "parent_sig", "discovery_move_depth", "operator"])
    write_csv(out / "plan_catalog.csv", list(catalog.values()), ["circuit_id", "source", "filename", "sig", "active_width", "plan_json"])
    write_csv(out / "features.csv", feature_rows, ["circuit_id", "sig"] + featkeys)
    write_csv(out / "feature_pairs.csv", pairs, selcols + paircols)
    json_out(out / "source_summary.json", source_summary)
    json_out(out / "prepare_errors.json", issues)
    json_out(out / "manifest.json", dict(
        status="prepared_only_not_labeled", qiskit=qiskit.__version__,
        data_protocol="sampled_depth_1_to_3_reuse_pilot_v1", qasm_paths=paths,
        original_active_compaction="reuse_feasibility_audit_fixed semantics",
        feature_code=str(a.feature_code), feature_count=len(featkeys),
        routing_config="heavy_hex(7), sabre layout and routing, optimization_level=1, seed_transpiler",
        circuits_requested=[circuit_id(*x) for x in chosen],
        circuits_prepared=sorted(paths), errors=len(issues),
        selected_source_candidate_rows=len(selected),
        selected_unique_plans=len(catalog), total_routing_jobs_for_2_seeds=len(catalog)*2,
        parameter_settings=dict(max_originals=a.max_originals,max_depth=a.max_depth,
            max_candidates=a.max_candidates,per_op_keep=a.per_op_keep,
            max_frontier=a.max_frontier,max_attempts_per_op=a.max_attempts_per_op,
            max_discovered=a.max_discovered,seed=a.seed),
        warnings=["No routing or learning has occurred.",
                  "Discovery depth is sampled path length, not global shortest path.",
                  "Measured features are NOT calibrated to a physical objective.",
                  "Reserved final-test circuits must not be chosen after inspecting outcomes."]))
    print(f"[done] prepare only. {len(selected)} source/candidate rows; "
          f"{len(catalog)} unique plans; {len(catalog)*2} seed jobs at 2 seeds; errors={len(issues)}")


def _load_local_gidnet(gidnet_file: Path, utils_file: Path):
    """Import exactly the supplied patched compiler, with a minimal .utils helper.

    This avoids extra benchmark/QNET imports in the original utils.py. Equivalent
    to the private loader used for the successful reuse-feasibility audit.
    """
    import numpy as np
    import networkx as nx
    from qiskit.converters import circuit_to_dag
    # Load only user's merge_subsets function by AST, not other utils imports.
    tr = ast.parse(utils_file.read_text(encoding="utf-8-sig"), filename=str(utils_file))
    nodes = [n for n in tr.body if isinstance(n, ast.FunctionDef) and n.name == "merge_subsets"]
    if len(nodes) != 1:
        raise RuntimeError("Expected exactly one merge_subsets() in utils.py")
    module_ast = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module_ast)
    ns = {}
    exec(compile(module_ast, str(utils_file), "exec"), ns)  # user's trusted local source
    pkg = types.ModuleType("gidnet")
    pkg.__path__ = [str(gidnet_file.parent)]
    sub = types.ModuleType("gidnet.utils")
    sub.merge_subsets = ns["merge_subsets"]
    sys.modules["gidnet"] = pkg
    sys.modules["gidnet.utils"] = sub
    sp = importlib.util.spec_from_file_location("gidnet.qubitreuse", str(gidnet_file))
    if sp is None or sp.loader is None:
        raise RuntimeError(f"Cannot load GidNET: {gidnet_file}")
    m = importlib.util.module_from_spec(sp)
    sys.modules["gidnet.qubitreuse"] = m
    sp.loader.exec_module(m)
    GidNET = m.GidNET
    # Gate against accidentally using the older, reset-dropping compiler.
    import inspect
    code = inspect.getsource(GidNET.compile_to_dynamic_circuit)
    if "reset_after_node_qubits" not in code:
        raise RuntimeError("Supplied GidNET lacks the previously verified reset insertion fix")

    def quantum_only(self):
        copy = self.circuit.copy()
        copy.remove_final_measurements(inplace=True)
        dag = circuit_to_dag(copy)
        graph = self._to_networkx(dag)
        qs = list(dag.qubits)
        roots = [dag.input_map[q] for q in qs]
        ends = [dag.output_map[q] for q in qs]
        n = len(qs)
        bm = np.zeros((n, n), dtype=int)
        for i, root in enumerate(roots):
            reached = nx.descendants(graph, root)
            for j, end in enumerate(ends):
                if end in reached or root == end:
                    bm[i, j] = 1
        cm = np.ones((n, n), dtype=int) - bm.T
        np.fill_diagonal(cm, 0)
        return bm, cm
    GidNET.compute_intial_biadjacency_and_candidate_matrix = quantum_only
    return GidNET


def route_stage(a):
    import qiskit
    if qiskit.__version__ != "0.45.3":
        raise RuntimeError(f"Pilot needs Qiskit 0.45.3, got {qiskit.__version__}")
    if not a.seeds:
        raise ValueError("Routing is explicit: pass --seeds 42,43 for the 2-seed smoke test")
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique comma-separated integers")
    # Avoid accidentally using held-out seeds as pilot labels.
    if any(s in range(100, 130) for s in seeds):
        raise ValueError("Seeds 100-129 are reserved from previous held-out evaluation; do not use for pilot")
    out = Path(a.out_dir)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    catalog = {(r["circuit_id"], r["sig"]): r for r in read_csv(out / "plan_catalog.csv")}
    chosen = read_csv(out / "selected_candidates.csv")
    # First group by Original, selecting an original + a bounded number of neighbors.
    grouped = defaultdict(list)
    for r in chosen:
        grouped[(r["circuit_id"], r["original_sig"])].append(r)
    key_order = sorted(grouped)
    if a.max_route_groups:
        key_order = key_order[:a.max_route_groups]
    targets = {}
    for key in key_order:
        g = sorted(grouped[key], key=lambda r:(-int(r["is_original"]), int(r["discovery_move_depth"]), r["sig"]))
        originals = [r for r in g if int(r["is_original"]) == 1]
        if len(originals) != 1:
            raise AssertionError(f"Expected exactly one original in {key}")
        neighbors = [r for r in g if int(r["is_original"]) == 0]
        if a.max_route_candidates_per_group:
            neighbors = neighbors[:a.max_route_candidates_per_group]
        for r in originals + neighbors:
            targets[(r["circuit_id"], r["sig"])] = r
    required = len(targets)*len(seeds)
    results_file = out / "routing_results.csv"
    existing = read_csv(results_file) if results_file.exists() else []
    done = {(r["circuit_id"], r["sig"], int(r["seed"])) for r in existing}
    remaining = sum((cid,sig_,seed) not in done for cid,sig_ in targets for seed in seeds)
    print(f"[preflight] routing target: {len(targets)} unique plans × {len(seeds)} seeds; "
          f"{required} total, {remaining} remaining. Heavy-hex SABRE, optimization_level=1.")
    # Enforce the same setup on resumed executions.
    previous = out / "routing_setup.json"
    settings = {"seeds": seeds, "router": "router_template:route",
                "backend": "Heavy-hex distance 7", "layout_method": "sabre",
                "routing_method": "sabre", "optimization_level": 1,
                "qiskit": "0.45.3"}
    if previous.exists():
        old = json.loads(previous.read_text(encoding="utf-8"))
        if old != settings:
            raise ValueError("Routing setup differs from previous run. Use a NEW out-dir; do not mix labels")
    if remaining == 0:
        print("[route] all requested jobs already present; no routing performed")
        return
    # Build the supplied GidNET class, then load the UNMODIFIED Experiment-3 adapter.
    gidnet = _load_local_gidnet(a.gidnet_file.resolve(), a.utils_file.resolve())
    adapter = importlib.import_module("experiment3_pipeline_adapter")
    if adapter.GidNET is not gidnet:
        raise AssertionError("Experiment-3 adapter did not pick up the supplied GidNET class")
    router = importlib.import_module("router_template")
    from qiskit import QuantumCircuit
    from reuse_validity import ReuseValidityChecker
    from reuse_adversarial import reference_static_feasible
    # Save the settings only when we are ready to route.
    json_out(previous, settings)
    compiled_path = out / "compiled_features.csv"
    compiled = read_csv(compiled_path) if compiled_path.exists() else []
    compiled_seen = {(r["circuit_id"],r["sig"]) for r in compiled}
    cache = {}
    route_cols = ["circuit_id", "source", "filename", "sig", "seed", "swap_count",
                  "routed_depth", "routing_time_sec", "status", "error"]
    comp_cols = ["circuit_id", "sig", "logical_depth", "norm_depth", "dyn_width", "status", "error"]
    def append(row, path, cols):
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8-sig" if not exists else "utf-8") as h:
            w = csv.DictWriter(h, fieldnames=cols, extrasaction="ignore")
            if not exists:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in cols})
    for i, ((cid,s), r) in enumerate(sorted(targets.items()), 1):
        missing = [seed for seed in seeds if (cid,s,seed) not in done]
        if not missing:
            continue
        source, filename = r["source"], r["filename"]
        try:
            if cid not in cache:
                sourcepath = Path(manifest["qasm_paths"][cid])
                raw = QuantumCircuit.from_qasm_file(str(sourcepath))
                qc, _ = compact_active(raw)
                ck = ReuseValidityChecker(qc)
                if ck.unsupported is not None:
                    raise RuntimeError(f"Unsupported source circuit: {ck.unsupported.reason}")
                cache[cid] = (qc, ck)
            qc, ck = cache[cid]
            # Original GidNET's fixed-plan compiler indexes the first n DAG
            # input/output values as quantum wires. Refuse to use a DAG whose
            # order violates that assumption (classical wires could be misread).
            from qiskit.converters import circuit_to_dag
            dag_check = circuit_to_dag(qc)
            q_inputs = [dag_check.input_map[q] for q in dag_check.qubits]
            q_outputs = [dag_check.output_map[q] for q in dag_check.qubits]
            if (list(dag_check.input_map.values())[:qc.num_qubits] != q_inputs or
                    list(dag_check.output_map.values())[:qc.num_qubits] != q_outputs):
                raise RuntimeError("GidNET DAG input/output ordering includes classical wires before qubits")
            plan = canonical(json.loads(catalog[(cid,s)]["plan_json"]))
            if not ck.validate(plan, target_width=len(plan)).is_valid or not reference_static_feasible(qc, plan, len(plan)):
                raise RuntimeError("Candidate failed independent static validity before compilation")
            dyn = adapter.build_dynamic_circuit(qc, plan)
            norm = adapter.normalize_for_routing(dyn)
            if dyn.num_qubits != len(plan) or norm.num_qubits != len(plan):
                raise RuntimeError("Compiled physical width differs from candidate active width")
            if (cid,s) not in compiled_seen:
                append(dict(circuit_id=cid,sig=s,logical_depth=dyn.depth(),norm_depth=norm.depth(),
                            dyn_width=dyn.num_qubits,status="ok",error=""),compiled_path,comp_cols)
                compiled_seen.add((cid,s))
        except Exception as e:
            error=f"{type(e).__name__}: {e}"
            if (cid,s) not in compiled_seen:
                append(dict(circuit_id=cid,sig=s,logical_depth="",norm_depth="",dyn_width="",
                            status="compile_error",error=error),compiled_path,comp_cols)
                compiled_seen.add((cid,s))
            for seed in missing:
                append(dict(circuit_id=cid,source=source,filename=filename,sig=s,seed=seed,
                            swap_count="",routed_depth="",routing_time_sec="",status="compile_error",
                            error=error),results_file,route_cols)
                done.add((cid,s,seed))
            print(f"[route {i}/{len(targets)}] {cid}/{s}: compile error: {error}", flush=True)
            continue
        for seed in missing:
            start=time.perf_counter()
            try:
                response=router.route(norm, seed)
                swap=float(response["swap_count"])
                depth=float(response["routed_depth"])
                if not math.isfinite(swap) or not math.isfinite(depth):
                    raise ValueError("Router returned non-finite physical cost")
                rec=dict(circuit_id=cid,source=source,filename=filename,sig=s,seed=seed,
                         swap_count=swap,routed_depth=depth,routing_time_sec=time.perf_counter()-start,
                         status="ok",error="")
            except Exception as e:
                rec=dict(circuit_id=cid,source=source,filename=filename,sig=s,seed=seed,
                         swap_count="",routed_depth="",routing_time_sec=time.perf_counter()-start,
                         status="route_error",error=f"{type(e).__name__}: {e}")
            append(rec,results_file,route_cols)
            done.add((cid,s,seed))
            print(f"[route {i}/{len(targets)}] {cid}/{s} seed={seed}: "
                  f"{rec['status']} swap={rec['swap_count']} depth={rec['routed_depth']}", flush=True)
    print("[route] wrote",results_file, "(existing rows retained; failures recorded, not hidden)")


def labels_stage(a):
    if not a.seeds:
        raise ValueError("Provide --seeds matching the routing run, e.g. --seeds 42,43")
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    out = Path(a.out_dir)
    path=out/"routing_results.csv"
    if not path.exists():
        raise FileNotFoundError("No routing results: run the explicit route stage first")
    routing = defaultdict(dict)
    for row in read_csv(path):
        routing[(row["circuit_id"],row["sig"])][int(row["seed"])]=row
    source_pairs = read_csv(out / "selected_candidates.csv")
    rows, missing=[],[]
    for item in source_pairs:
        cid, orig, s = item["circuit_id"], item["original_sig"], item["sig"]
        orig_runs = routing[(cid,orig)]
        cand_runs = routing[(cid,s)]
        if not all(seed in orig_runs and seed in cand_runs and
                   orig_runs[seed]["status"]=="ok" and cand_runs[seed]["status"]=="ok"
                   for seed in seeds):
            missing.append(dict(circuit_id=cid,original_sig=orig,sig=s,
                                issue="Missing/failed routing rows for the exact specified seed set"))
            continue
        for obj in ("swap_count", "routed_depth"):
            before=sum(float(orig_runs[seed][obj]) for seed in seeds)/len(seeds)
            after=sum(float(cand_runs[seed][obj]) for seed in seeds)/len(seeds)
            absolute=before-after
            gain_pct=100*absolute/before if before>0 else ""
            rows.append(dict(circuit_id=cid,source=item["source"],filename=item["filename"],
                family=item["family"], original_sig=orig,sig=s,
                is_original=item["is_original"],discovery_move_depth=item["discovery_move_depth"],
                objective=obj,baseline_mean=before,candidate_mean=after,abs_gain=absolute,
                gain_pct=gain_pct,n_seeds=len(seeds),harm=int(absolute<0)))
    cols=["circuit_id","source","filename","family","original_sig","sig","is_original",
          "discovery_move_depth","objective","baseline_mean","candidate_mean","abs_gain",
          "gain_pct","n_seeds","harm"]
    write_csv(out / "labels.csv", rows, cols)
    write_csv(out / "unlabeled.csv", missing, ["circuit_id","original_sig","sig","issue"])
    json_out(out / "label_summary.json", dict(
        seed_set=seeds, rows_labeled=len(rows), incomplete_source_candidate_rows=len(missing),
        purpose="Two-seed smoke test. NOT training-ready; not held-out evaluation.",
        gain_pct_undefined_when_baseline_zero=True,
        separate_physical_objectives=["swap_count","routed_depth"]))
    print(f"[labels] {len(rows)} rows, {len(missing)} incomplete plan-pairs. "
          "Pilot labels are NOT statistically reliable training results.")


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("prepare","route","labels"))
    ap.add_argument("--audit-dirs",nargs="+",type=Path,default=[Path("fixed_full"),Path("wave2")])
    ap.add_argument("--out-dir",type=Path,default=Path("multistep_pilot_out"))
    ap.add_argument("--feature-code",type=Path,default=Path("exp4d_features_robust_audit_v2.py"))
    ap.add_argument("--gidnet-file",type=Path,default=Path("qubitreuse.py"))
    ap.add_argument("--utils-file",type=Path,default=Path("utils.py"))
    ap.add_argument("--circuits",default="",help="comma separated source::filename keys")
    ap.add_argument("--max-originals",type=int,default=2)
    ap.add_argument("--max-candidates",type=int,default=20,help="Non-original selected plans per source")
    ap.add_argument("--max-depth",type=int,default=3)
    ap.add_argument("--max-frontier",type=int,default=8)
    ap.add_argument("--per-op-keep",type=int,default=4)
    ap.add_argument("--max-attempts-per-op",type=int,default=180)
    ap.add_argument("--max-discovered",type=int,default=200)
    ap.add_argument("--seed",type=int,default=2026)
    ap.add_argument("--seeds",default="",help="Required for route/labels, e.g. 42,43")
    ap.add_argument("--max-route-groups",type=int,default=0,help="Smoke: 1 group; 0 means all")
    ap.add_argument("--max-route-candidates-per-group",type=int,default=0,help="Smoke: 1 neighbor per group; 0 all")
    a=ap.parse_args()
    if a.stage=="prepare": prepare(a)
    elif a.stage=="route": route_stage(a)
    else: labels_stage(a)

if __name__=="__main__":
    main()
