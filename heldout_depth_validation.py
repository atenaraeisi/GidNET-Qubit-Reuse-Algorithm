#!/usr/bin/env python3
"""Frozen held-out validation: 3-step vs 5-step fixed-width reuse plans.

No candidate discovery or training occurs here. Three explicit stages:
    select  -- From DEVELOPMENT seeds only; freezes winning plans, including No Repair.
    route   -- Routes ONLY frozen plans on fresh evaluation seeds; resume-safe.
    analyze -- Read-only computation of post-selection, held-out comparisons.

Independent validation means seeds distinct from the five seeds used to choose
plans. Do not treat these six development circuits as an independent final test.
Requires Qiskit 0.45.3 only for route; select/analyze are stdlib-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

FIELDS_ROUTE = ("circuit_id", "source", "filename", "sig", "seed", "swap_count",
                "routed_depth", "routing_time_sec", "status", "error")
FIELDS_COMP = ("circuit_id", "sig", "logical_depth", "norm_depth", "dyn_width", "status", "error")
FIELDS_WINNER = ("circuit_id", "source", "filename", "family", "original_sig",
                 "objective", "pool", "chosen_sig", "chosen_move_depth",
                 "is_no_repair", "development_baseline_mean", "development_candidate_mean",
                 "development_abs_gain", "development_gain_pct")
FIELDS_CATALOG = ("circuit_id", "source", "filename", "sig", "active_width", "plan_json")
DEV_SEEDS = tuple(range(42, 47))
DEFAULT_EVAL_SEEDS = tuple(range(200, 230))
OBJECTIVES = ("swap_count", "routed_depth")


def read_csv(p):
    with Path(p).open("r", newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError(f"Missing CSV header: {p}")
        return list(r)


def write_csv_new(p, rows, columns):
    with Path(p).open("x", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


def append_csv(p, row, columns):
    p = Path(p)
    exists = p.exists()
    with p.open("a", newline="", encoding="utf-8" if exists else "utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in columns})
        f.flush()


def write_json_new(p, x):
    with Path(p).open("x", encoding="utf-8") as f:
        json.dump(x, f, ensure_ascii=False, indent=2, allow_nan=False)


def read_json(p):
    with Path(p).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def idx_unique(rows, cols, label):
    idx = {}
    for row in rows:
        key = tuple(row[c] for c in cols)
        if key in idx:
            raise ValueError(f"Duplicate {label}: {key}")
        idx[key] = row
    return idx


def parse_plan(r):
    plan = json.loads(r["plan_json"])
    return tuple(sorted(tuple(int(q) for q in c) for c in plan))


def canonical_sig(plan):
    return hashlib.sha256(json.dumps([list(c) for c in plan],
                                     separators=(",", ":")).encode()).hexdigest()[:24]


def parse_seeds(s):
    if not s:
        return DEFAULT_EVAL_SEEDS
    out = tuple(int(x.strip()) for x in s.split(",") if x.strip())
    if len(out) < 2 or len(out) != len(set(out)) or any(x < 0 for x in out):
        raise ValueError("Evaluation seeds must be distinct nonnegative integers (at least two)")
    if set(out) & set(DEV_SEEDS):
        raise ValueError("Evaluation seeds must not overlap development seeds 42-46")
    return out


def check_mean(label_row, route_idx):
    cid, orig, cand, obj = (label_row[k] for k in
                            ("circuit_id", "original_sig", "sig", "objective"))
    if obj not in OBJECTIVES:
        raise ValueError(f"Unexpected objective: {obj}")
    vals_before, vals_after = [], []
    for seed in DEV_SEEDS:
        b = route_idx.get((cid, orig, str(seed)))
        c = route_idx.get((cid, cand, str(seed)))
        if b is None or c is None or b["status"] != "ok" or c["status"] != "ok":
            raise ValueError(f"Development route missing/failed for {cid} original {orig} candidate {cand} seed {seed}")
        vals_before.append(float(b[obj]))
        vals_after.append(float(c[obj]))
    baseline = sum(vals_before) / len(vals_before)
    after = sum(vals_after) / len(vals_after)
    for name, value in (("baseline_mean", baseline), ("candidate_mean", after),
                        ("abs_gain", baseline - after)):
        if not math.isclose(float(label_row[name]), value, abs_tol=1e-7, rel_tol=1e-9):
            raise ValueError(f"Stale/corrupt development label {cid}:{orig}/{cand}, {name}")
    if int(label_row["n_seeds"]) != len(DEV_SEEDS):
        raise ValueError("Wrong n_seeds in development label")
    if baseline > 0:
        expected_pct = 100 * (baseline - after) / baseline
        if not math.isclose(float(label_row["gain_pct"]), expected_pct, abs_tol=1e-7, rel_tol=1e-9):
            raise ValueError("Wrong development percentage label")
    elif label_row["gain_pct"] != "":
        raise ValueError("Zero-baseline development gain should have undefined percentage")


def stage_select(a):
    old, new = Path(a.depth3_dir).resolve(), Path(a.depth5_dir).resolve()
    target = Path(a.out_dir).resolve()
    if old == new or target in (old, new):
        raise ValueError("Separate depth3, depth5 and output directories are required")
    eval_seeds = parse_seeds(a.eval_seeds)
    manifest = {"depth3": read_json(old / "manifest.json"),
                "depth5": read_json(new / "manifest.json")}
    datasets = {}
    fingerprints = {}
    for pool, source in (("depth3", old), ("depth5", new)):
        names = ("manifest.json", "routing_setup.json", "plan_catalog.csv",
                 "selected_candidates.csv", "labels.csv", "routing_results.csv")
        fingerprints[pool] = {n: sha256_file(source / n) for n in names}
        m = manifest[pool]
        if m.get("qiskit") != "0.45.3" or m.get("errors") != 0:
            raise ValueError(f"{pool}: unexpected Qiskit version or prepare errors")
        if m.get("parameter_settings", {}).get("max_depth") != (3 if pool == "depth3" else 5):
            raise ValueError(f"{pool}: wrong max_depth in manifest")
        if (m.get("parameter_settings", {}).get("max_originals") != 2 or
                m.get("parameter_settings", {}).get("max_candidates") != 20):
            raise ValueError(f"{pool}: expected 2 originals and 20 candidates per original")
        setup = read_json(source / "routing_setup.json")
        expected_setup = {"router": "router_template:route", "backend": "Heavy-hex distance 7",
                          "layout_method": "sabre", "routing_method": "sabre",
                          "optimization_level": 1, "qiskit": "0.45.3"}
        for k, expected in expected_setup.items():
            if setup.get(k) != expected:
                raise ValueError(f"{pool}: router setup differs at {k}")
        if not set(DEV_SEEDS) <= set(map(int, setup.get("seeds", []))):
            raise ValueError(f"{pool}: development seed coverage insufficient")
        if m.get("routing_config") != "heavy_hex(7), sabre layout and routing, optimization_level=1, seed_transpiler":
            raise ValueError(f"{pool}: manifest router setup differs")
        selected = idx_unique(read_csv(source / "selected_candidates.csv"),
                              ("circuit_id", "original_sig", "sig"), pool + " selected")
        catalog = idx_unique(read_csv(source / "plan_catalog.csv"),
                             ("circuit_id", "sig"), pool + " catalog")
        routes = idx_unique(read_csv(source / "routing_results.csv"),
                            ("circuit_id", "sig", "seed"), pool + " routing")
        labels = idx_unique(read_csv(source / "labels.csv"),
                            ("circuit_id", "original_sig", "sig", "objective"), pool + " labels")
        if len(selected) != int(m["selected_source_candidate_rows"]):
            raise ValueError(f"{pool}: selected count mismatch")
        if set(catalog) != {(v["circuit_id"], v["sig"]) for v in selected.values()}:
            raise ValueError(f"{pool}: catalog/selected plans mismatch")
        if len(labels) != len(selected) * 2 or len(routes) < len(catalog) * len(DEV_SEEDS):
            raise ValueError(f"{pool}: incomplete labels or routing")
        for row in catalog.values():
            plan = parse_plan(row)
            if not plan or any(not c for c in plan) or len(plan) != int(row["active_width"]):
                raise ValueError(f"{pool}: malformed plan {row['sig']}")
            if canonical_sig(plan) != row["sig"]:
                raise ValueError(f"{pool}: plan signature mismatch {row['sig']}")
        for key, row in labels.items():
            if key[:3] not in selected:
                raise ValueError(f"{pool}: label not selected {key}")
            check_mean(row, routes)
        for key, v in selected.items():
            if key[0] != v["source"] + "::" + v["filename"]:
                raise ValueError(f"{pool}: source identity mismatch {key}")
            if (key[0], key[2]) not in catalog or (key[0], key[1]) not in catalog:
                raise ValueError(f"{pool}: selected/original missing from catalog")
            if not all(key + (obj,) in labels for obj in OBJECTIVES):
                raise ValueError(f"{pool}: selected label missing {key}")
        datasets[pool] = dict(manifest=m, selected=selected, catalog=catalog,
                              labels=labels, routes=routes)
    m3, m5 = manifest["depth3"], manifest["depth5"]
    if (m3["qasm_paths"] != m5["qasm_paths"] or
            m3["original_active_compaction"] != m5["original_active_compaction"] or
            m3["routing_config"] != m5["routing_config"] or
            m3["circuits_prepared"] != m5["circuits_prepared"]):
        raise ValueError("Depth3 and depth5 source circuits or compilation protocols differ")
    for key in set(datasets["depth3"]["catalog"]) & set(datasets["depth5"]["catalog"]):
        if parse_plan(datasets["depth3"]["catalog"][key]) != parse_plan(datasets["depth5"]["catalog"][key]):
            raise ValueError(f"Same plan signature has different chains across pools: {key}")
        for seed in DEV_SEEDS:
            r3 = datasets["depth3"]["routes"][(key[0], key[1], str(seed))]
            r5 = datasets["depth5"]["routes"][(key[0], key[1], str(seed))]
            for obj in OBJECTIVES:
                if r3[obj] != r5[obj]:
                    raise ValueError(f"Development route disagreement for shared plan: {key}, seed {seed}, {obj}")
    sources = {}
    for pool in ("depth3", "depth5"):
        for cid, orig, sig in datasets[pool]["selected"]:
            sources.setdefault(pool, set()).add((cid, orig))
    if sources["depth3"] != sources["depth5"]:
        raise ValueError("Source originals differ between depth3 and depth5")
    if len(sources["depth3"]) != 12:
        raise ValueError(f"Expected 12 originals; got {len(sources['depth3'])}")

    winners, target_catalog = [], {}
    for cid, orig in sorted(sources["depth3"]):
        for pool in ("depth3", "depth5"):
            ds = datasets[pool]
            by_sig = {s: v for (c, o, s), v in ds["selected"].items() if (c, o) == (cid, orig)}
            if orig not in by_sig or len(by_sig) != 21 or int(by_sig[orig]["is_original"]) != 1:
                raise ValueError(f"{pool} {cid}/{orig}: expected exactly 1 original + 20 candidates")
            for obj in OBJECTIVES:
                # Choose smallest mean cost; ties favor No Repair, then fewer moves, then signature.
                scored = []
                for sig, meta in by_sig.items():
                    lab = ds["labels"][(cid, orig, sig, obj)]
                    scored.append((float(lab["candidate_mean"]), 0 if sig == orig else 1,
                                   int(meta["discovery_move_depth"]), sig, meta, lab))
                cost, _, depth, chosen_sig, metadata, winner_label = min(scored)
                winners.append(dict(circuit_id=cid, source=metadata["source"],
                    filename=metadata["filename"], family=metadata["family"],
                    original_sig=orig, objective=obj, pool=pool, chosen_sig=chosen_sig,
                    chosen_move_depth=depth, is_no_repair=int(chosen_sig == orig),
                    development_baseline_mean=winner_label["baseline_mean"],
                    development_candidate_mean=winner_label["candidate_mean"],
                    development_abs_gain=winner_label["abs_gain"],
                    development_gain_pct=winner_label["gain_pct"]))
                for s in (chosen_sig, orig):
                    catalogrow = ds["catalog"][(cid, s)]
                    tgtkey = (cid, s)
                    if tgtkey in target_catalog and parse_plan(target_catalog[tgtkey]) != parse_plan(catalogrow):
                        raise ValueError(f"Incompatible catalog plan for {tgtkey}")
                    target_catalog[tgtkey] = catalogrow
    if len(winners) != 48:
        raise AssertionError("Expected 12 sources x 2 pools x 2 objectives")
    print(f"[select] Frozen decisions: {len(winners)}; unique plans incl originals: {len(target_catalog)}; "
          f"held-out routing jobs: {len(target_catalog) * len(eval_seeds)}")
    print("[select] No Repair decisions:", sum(int(w["is_no_repair"]) for w in winners))
    if a.dry_run:
        print("[DRY RUN] No files written. Omit --dry-run to freeze and write selection files.")
        return
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Output must be new/empty to avoid changing locked decisions: {target}")
    qasm_digests = {}
    for cid, path in m3["qasm_paths"].items():
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Cannot freeze source QASM hash; file does not exist: {p}")
        qasm_digests[cid] = sha256_file(p)
    target.mkdir(parents=True, exist_ok=True)
    selected_path = target / "frozen_winners.csv"
    catalog_path = target / "frozen_plan_catalog.csv"
    write_csv_new(selected_path, winners, FIELDS_WINNER)
    write_csv_new(catalog_path, [target_catalog[key] for key in sorted(target_catalog)], FIELDS_CATALOG)
    lock = dict(protocol="fixed_width_heldout_winners_v1", selection_seeds=list(DEV_SEEDS),
                evaluation_seeds=list(eval_seeds), pools=["depth3", "depth5"],
                winner_rows=len(winners), unique_plans=len(target_catalog),
                unique_sources=len(sources["depth3"]), qasm_paths=m3["qasm_paths"],
                qasm_sha256=qasm_digests,
                selection_csv_sha256=sha256_file(selected_path),
                catalog_csv_sha256=sha256_file(catalog_path),
                source_input_sha256=fingerprints,
                routing_config=dict(router="router_template:route", backend="Heavy-hex distance 7",
                                    layout_method="sabre", routing_method="sabre", optimization_level=1,
                                    qiskit="0.45.3", active_compaction=m3["original_active_compaction"]),
                warning="Selection frozen using only 42-46; evaluate 200-229 without reselection. Six development circuits; not paper final test.")
    write_json_new(target / "selection_lock.json", lock)
    print("[select] Frozen:", target / "selection_lock.json")
    print("[select] Original datasets unchanged. You can now run the route stage.")


def load_locked(out):
    out = Path(out)
    lock = read_json(out / "selection_lock.json")
    if lock["protocol"] != "fixed_width_heldout_winners_v1":
        raise ValueError("Wrong selection protocol")
    if lock["selection_seeds"] != list(DEV_SEEDS):
        raise ValueError("Unexpected development seed set")
    if set(lock["evaluation_seeds"]) & set(DEV_SEEDS):
        raise ValueError("Evaluation/selection seeds overlap")
    if sha256_file(out / "frozen_winners.csv") != lock["selection_csv_sha256"]:
        raise ValueError("Frozen winner selection was modified after locking")
    if sha256_file(out / "frozen_plan_catalog.csv") != lock["catalog_csv_sha256"]:
        raise ValueError("Frozen plan catalog was modified after locking")
    winners = read_csv(out / "frozen_winners.csv")
    catalog = idx_unique(read_csv(out / "frozen_plan_catalog.csv"),
                         ("circuit_id", "sig"), "frozen plan")
    if len(winners) != lock["winner_rows"] or len(catalog) != lock["unique_plans"]:
        raise ValueError("Frozen file count inconsistent with lock")
    for cid, path in lock["qasm_paths"].items():
        p = Path(path)
        if not p.is_file() or sha256_file(p) != lock["qasm_sha256"][cid]:
            raise ValueError(f"Source QASM file changed or disappeared: {cid}, {p}")
    return lock, winners, catalog


def stage_route(a):
    out = Path(a.out_dir).resolve()
    lock, winners, catalog = load_locked(out)
    seeds = [int(s) for s in lock["evaluation_seeds"]]
    route_path = out / "heldout_routing_results.csv"
    comp_path = out / "heldout_compiled_features.csv"
    existing = idx_unique(read_csv(route_path) if route_path.exists() else [],
                          ("circuit_id", "sig", "seed"), "heldout routing")
    expected = {(cid, sig, str(seed)) for (cid, sig) in catalog for seed in seeds}
    if not set(existing) <= expected:
        raise ValueError("Unexpected routing jobs in heldout file; refusing to mix experiments")
    todo = sorted(expected - set(existing))
    print(f"[route] {len(catalog)} frozen plans × {len(seeds)} held-out seeds = {len(expected)}; "
          f"{len(todo)} remaining")
    if not todo:
        print("[route] All frozen held-out jobs already stored; nothing to route")
        return
    import qiskit
    if qiskit.__version__ != "0.45.3":
        raise RuntimeError(f"Expected Qiskit 0.45.3, got {qiskit.__version__}")
    pilot_path = Path(a.pilot_code)
    if not pilot_path.is_file():
        raise FileNotFoundError(f"Pilot module is required for identical compilation: {pilot_path}")
    spec = importlib.util.spec_from_file_location("_heldout_pilot_compiler", pilot_path.resolve())
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    gidnet = pilot._load_local_gidnet(Path(a.gidnet_file).resolve(), Path(a.utils_file).resolve())
    adapter = importlib.import_module("experiment3_pipeline_adapter")
    if adapter.GidNET is not gidnet:
        raise AssertionError("Experiment-3 adapter GidNET does not match imported patched compiler")
    router = importlib.import_module("router_template")
    from qiskit import QuantumCircuit
    from qiskit.converters import circuit_to_dag
    from reuse_adversarial import reference_static_feasible
    from reuse_validity import ReuseValidityChecker
    compiled_rows = idx_unique(read_csv(comp_path) if comp_path.exists() else [],
                               ("circuit_id", "sig"), "compiled features")
    if not set(compiled_rows) <= set(catalog):
        raise ValueError("Compiled features contain unknown plans")
    source_cache = {}
    for i, ((cid, sig), plan_row) in enumerate(sorted(catalog.items()), 1):
        pending = [seed for seed in seeds if (cid, sig, str(seed)) not in existing]
        if not pending:
            continue
        source, filename = plan_row["source"], plan_row["filename"]
        try:
            if cid not in source_cache:
                raw = QuantumCircuit.from_qasm_file(lock["qasm_paths"][cid])
                qc, _ = pilot.compact_active(raw)
                ck = ReuseValidityChecker(qc)
                if ck.unsupported is not None:
                    raise ValueError(f"Unsupported circuit: {ck.unsupported.reason}")
                dag = circuit_to_dag(qc)
                quantum_inputs = [dag.input_map[q] for q in dag.qubits]
                quantum_outputs = [dag.output_map[q] for q in dag.qubits]
                if (list(dag.input_map.values())[:qc.num_qubits] != quantum_inputs or
                        list(dag.output_map.values())[:qc.num_qubits] != quantum_outputs):
                    raise RuntimeError("DAG input/output wire ordering differs from GidNET fixed-plan assumption")
                source_cache[cid] = (qc, ck)
            qc, ck = source_cache[cid]
            plan = parse_plan(plan_row)
            if (len(plan) != int(plan_row["active_width"]) or
                    not ck.validate(plan, target_width=len(plan)).is_valid or
                    not reference_static_feasible(qc, plan, len(plan))):
                raise ValueError("Frozen plan failed width or independent static feasibility check")
            dyn = adapter.build_dynamic_circuit(qc, plan)
            norm = adapter.normalize_for_routing(dyn)
            if dyn.num_qubits != len(plan) or norm.num_qubits != len(plan):
                raise ValueError("Compiled circuit width differs from frozen plan")
            if (cid, sig) in compiled_rows:
                prev = compiled_rows[(cid, sig)]
                if (prev["status"] != "ok" or int(float(prev["dyn_width"])) != dyn.num_qubits or
                    int(float(prev["logical_depth"])) != dyn.depth() or
                    int(float(prev["norm_depth"])) != norm.depth()):
                    raise ValueError("Compiled features mismatch cached compilation")
            else:
                rec = dict(circuit_id=cid, sig=sig, logical_depth=dyn.depth(),
                           norm_depth=norm.depth(), dyn_width=dyn.num_qubits, status="ok", error="")
                append_csv(comp_path, rec, FIELDS_COMP)
                compiled_rows[(cid, sig)] = rec
        except Exception as ex:
            error = f"{type(ex).__name__}: {ex}"
            if (cid, sig) not in compiled_rows:
                rec = dict(circuit_id=cid, sig=sig, logical_depth="", norm_depth="",
                           dyn_width="", status="compile_error", error=error)
                append_csv(comp_path, rec, FIELDS_COMP)
                compiled_rows[(cid, sig)] = rec
            for seed in pending:
                rec = dict(circuit_id=cid, source=source, filename=filename, sig=sig,
                           seed=seed, swap_count="", routed_depth="", routing_time_sec="",
                           status="compile_error", error=error)
                append_csv(route_path, rec, FIELDS_ROUTE)
                existing[(cid, sig, str(seed))] = rec
            print(f"[route {i}/{len(catalog)}] {cid}/{sig}: compile_error {error}", flush=True)
            continue
        for seed in pending:
            start = time.perf_counter()
            try:
                result = router.route(norm, seed)
                sw, dp = float(result["swap_count"]), float(result["routed_depth"])
                if not all(math.isfinite(x) and x >= 0 for x in (sw, dp)):
                    raise ValueError("Router returned invalid metric")
                rec = dict(circuit_id=cid, source=source, filename=filename, sig=sig,
                           seed=seed, swap_count=sw, routed_depth=dp,
                           routing_time_sec=time.perf_counter() - start, status="ok", error="")
            except Exception as ex:
                rec = dict(circuit_id=cid, source=source, filename=filename, sig=sig,
                           seed=seed, swap_count="", routed_depth="",
                           routing_time_sec=time.perf_counter() - start,
                           status="route_error", error=f"{type(ex).__name__}: {ex}")
            append_csv(route_path, rec, FIELDS_ROUTE)
            existing[(cid, sig, str(seed))] = rec
        print(f"[route {i}/{len(catalog)}] {cid}/{sig}: {len(pending)} seeds added", flush=True)
    print("[route] Completed. Inspect statuses before analysis; failures are retained, never hidden")


def stage_analyze(a):
    out = Path(a.out_dir).resolve()
    lock, winners, catalog = load_locked(out)
    seeds = [int(s) for s in lock["evaluation_seeds"]]
    path = out / "heldout_routing_results.csv"
    if not path.is_file():
        raise FileNotFoundError("No heldout_routing_results.csv. Run 'route' first")
    routes = idx_unique(read_csv(path), ("circuit_id", "sig", "seed"), "heldout routing")
    required = {(cid, sig, str(seed)) for cid, sig in catalog for seed in seeds}
    missing = required - set(routes)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} held-out Routing jobs. Resume 'route' first")
    failed = [k for k in required if routes[k]["status"] != "ok"]
    if failed:
        raise RuntimeError(f"Found {len(failed)} failed heldout Routing jobs; inspect CSV before analysis")
    extra = set(routes) - required
    if extra:
        raise ValueError("Unexpected held-out routing jobs mixed into evaluation file")
    comparison = []
    seedpairs = []
    for w in winners:
        cid, orig, cand, obj = w["circuit_id"], w["original_sig"], w["chosen_sig"], w["objective"]
        bv, cv = [], []
        for seed in seeds:
            b, c = routes[(cid, orig, str(seed))], routes[(cid, cand, str(seed))]
            before, after = float(b[obj]), float(c[obj])
            if not math.isfinite(before) or not math.isfinite(after):
                raise ValueError("Nonfinite held-out metric")
            bv.append(before); cv.append(after)
            seedpairs.append(dict(circuit_id=cid, original_sig=orig, objective=obj, pool=w["pool"],
                                  chosen_sig=cand, seed=seed, baseline=before,
                                  candidate=after, absolute_gain=before-after))
        before, after = sum(bv) / len(bv), sum(cv) / len(cv)
        gain = before - after
        devgain = float(w["development_abs_gain"])
        comparison.append(dict(circuit_id=cid, family=w["family"], original_sig=orig,
            objective=obj, pool=w["pool"], chosen_sig=cand,
            chosen_move_depth=w["chosen_move_depth"], is_no_repair=w["is_no_repair"],
            n_seeds=len(seeds), development_baseline=float(w["development_baseline_mean"]),
            development_gain_abs=devgain,
            development_gain_pct=w["development_gain_pct"],
            heldout_baseline=before, heldout_candidate=after,
            heldout_gain_abs=gain, heldout_gain_pct=100 * gain / before if before > 0 else "",
            flipped_to_harm=int(devgain > 0 and gain < 0), harm=int(gain < 0)))
    compar_idx = idx_unique(comparison, ("circuit_id", "original_sig", "objective", "pool"), "heldout comparison")
    deltas = []
    summary = {"protocol": lock["protocol"], "evaluation_seeds": seeds,
               "unique_plans": len(catalog), "routing_jobs": len(required),
               "six_development_circuits_not_final_test": True, "objectives": {}}
    for obj in OBJECTIVES:
        group = [r for r in comparison if r["objective"] == obj]
        ss = {pool: [r for r in group if r["pool"] == pool] for pool in ("depth3", "depth5")}
        d3, d5 = ss["depth3"], ss["depth5"]
        comparable = [r for r in d3 if r["heldout_gain_pct"] != ""]
        comparable5 = [r for r in d5 if r["heldout_gain_pct"] != ""]
        summary["objectives"][obj] = {}
        for pool, rows in ss.items():
            pct = [float(r["heldout_gain_pct"]) for r in rows if r["heldout_gain_pct"] != ""]
            summary["objectives"][obj][pool] = dict(
                n_sources=len(rows), n_defined_percent=len(pct),
                mean_gain_pct_by_original=sum(pct) / len(pct) if pct else None,
                harm_count=sum(int(r["harm"]) for r in rows),
                no_repair_count=sum(int(r["is_no_repair"]) for r in rows),
                development_positive_but_heldout_harm=sum(int(r["flipped_to_harm"]) for r in rows))
        for cid, orig in sorted({(r["circuit_id"], r["original_sig"]) for r in group}):
            a3, a5 = [compar_idx[(cid, orig, obj, pool)] for pool in ("depth3", "depth5")]
            delta = a5["heldout_candidate"] - a3["heldout_candidate"]
            deltas.append(dict(circuit_id=cid, original_sig=orig, objective=obj,
                depth3_sig=a3["chosen_sig"], depth5_sig=a5["chosen_sig"],
                same_winner=int(a3["chosen_sig"] == a5["chosen_sig"]),
                depth3_heldout_cost=a3["heldout_candidate"],
                depth5_heldout_cost=a5["heldout_candidate"],
                depth5_cost_minus_depth3_cost=delta,
                depth5_better=int(delta < -1e-9), depth3_better=int(delta > 1e-9),
                tied=int(abs(delta) <= 1e-9)))
        subset = [r for r in deltas if r["objective"] == obj]
        summary["objectives"][obj]["depth5_vs_depth3"] = dict(
            depth5_better=sum(r["depth5_better"] for r in subset),
            depth3_better=sum(r["depth3_better"] for r in subset),
            tied=sum(r["tied"] for r in subset),
            same_winner=sum(r["same_winner"] for r in subset))
    write_csv_exclusive = a.overwrite_analysis
    files = (out / "heldout_comparison.csv", out / "heldout_depth5_vs_depth3.csv",
             out / "heldout_seed_pairs.csv", out / "heldout_summary.json")
    if any(p.exists() for p in files) and not write_csv_exclusive:
        raise FileExistsError("Analysis outputs already exist; use --overwrite-analysis after intentional review")
    def write_csv_safe(p, rows, keys):
        if p.exists():
            p.unlink()
        write_csv_new(p, rows, keys)
    def write_json_safe(p, obj):
        if p.exists():
            p.unlink()
        write_json_new(p, obj)
    compare_cols = ("circuit_id", "family", "original_sig", "objective", "pool", "chosen_sig",
                    "chosen_move_depth", "is_no_repair", "n_seeds", "development_baseline",
                    "development_gain_abs", "development_gain_pct", "heldout_baseline",
                    "heldout_candidate", "heldout_gain_abs", "heldout_gain_pct", "flipped_to_harm", "harm")
    delta_cols = ("circuit_id", "original_sig", "objective", "depth3_sig", "depth5_sig",
                  "same_winner", "depth3_heldout_cost", "depth5_heldout_cost",
                  "depth5_cost_minus_depth3_cost", "depth5_better", "depth3_better", "tied")
    pair_cols = ("circuit_id", "original_sig", "objective", "pool", "chosen_sig", "seed",
                 "baseline", "candidate", "absolute_gain")
    write_csv_safe(files[0], comparison, compare_cols)
    write_csv_safe(files[1], deltas, delta_cols)
    write_csv_safe(files[2], seedpairs, pair_cols)
    write_json_safe(files[3], summary)
    print("[analyze] Frozen winners:", len(winners), "; complete held-out routing jobs:", len(required))
    for obj, info in summary["objectives"].items():
        print(f"[analyze] {obj}: depth3={info['depth3']['mean_gain_pct_by_original']}, "
              f"depth5={info['depth5']['mean_gain_pct_by_original']} (% means for positive baselines)")
        print("[analyze] 5-vs-3:", info["depth5_vs_depth3"])
    print("[analyze] Outputs:", files[0], files[1], files[3])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("select", "route", "analyze"))
    p.add_argument("--depth3-dir", type=Path, default=Path("pilot_six"))
    p.add_argument("--depth5-dir", type=Path, default=Path("pilot_depth5"))
    p.add_argument("--out-dir", type=Path, default=Path("heldout_depth3_vs_5"))
    p.add_argument("--eval-seeds", default=",".join(map(str, DEFAULT_EVAL_SEEDS)),
                   help="Comma-separated fresh evaluation seeds; default 200-229, locked at select")
    p.add_argument("--dry-run", action="store_true", help="Select preview only; no changes")
    p.add_argument("--pilot-code", type=Path, default=Path("build_multistep_pilot_validity_aware.py"))
    p.add_argument("--gidnet-file", type=Path, default=Path("qubitreuse.py"))
    p.add_argument("--utils-file", type=Path, default=Path("utils.py"))
    p.add_argument("--overwrite-analysis", action="store_true")
    a = p.parse_args()
    if a.stage == "select":
        stage_select(a)
    elif a.stage == "route":
        stage_route(a)
    else:
        stage_analyze(a)


if __name__ == "__main__":
    main()
