#!/usr/bin/env python3
"""Safely import existing 5-seed pilot routing into a *new* training prepare directory.

No routing runs here. DRY RUN by default. Only exact circuit::plan matches are copied,
verified against source QASM path, canonical plan JSON, width, Qiskit and routing setup.

PowerShell example:
 python import_training_routing_cache.py --old-dirs pilot_six pilot_depth5 --new-dir ml_train_prepare_v1
 python import_training_routing_cache.py --old-dirs pilot_six pilot_depth5 --new-dir ml_train_prepare_v1 --apply
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

SEEDS = (42, 43, 44, 45, 46)
ROUTE_COLS = ("circuit_id", "source", "filename", "sig", "seed", "swap_count", "routed_depth", "routing_time_sec", "status", "error")
COMP_COLS = ("circuit_id", "sig", "logical_depth", "norm_depth", "dyn_width", "status", "error")
SETUP_FIXED = {
    "router": "router_template:route",
    "backend": "Heavy-hex distance 7",
    "layout_method": "sabre",
    "routing_method": "sabre",
    "optimization_level": 1,
    "qiskit": "0.45.3",
}


def load_json(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def csv_rows(p, required_cols):
    with p.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not set(required_cols) <= set(reader.fieldnames):
            raise ValueError(f"Missing required columns in {p}: {set(required_cols) - set(reader.fieldnames or [])}")
        return list(reader.fieldnames), list(reader)


def index_rows(rows, cols, label):
    d = {}
    for r in rows:
        k = tuple(r[x] for x in cols)
        if k in d:
            raise ValueError(f"Duplicate {label}: {k}")
        d[k] = r
    return d


def normalized_plan(r):
    p = json.loads(r["plan_json"])
    return tuple(sorted(tuple(int(q) for q in chain) for chain in p))


def checked_number(v, *, integral=False):
    x = float(v)
    if not math.isfinite(x):
        raise ValueError(f"Non-finite numerical value: {v}")
    if integral and x != int(x):
        raise ValueError(f"Noninteger value: {v}")
    return x


def equivalent(a, b, cols):
    for k in cols:
        if k in {"swap_count", "routed_depth", "logical_depth", "norm_depth", "dyn_width"}:
            if not math.isclose(checked_number(a[k]), checked_number(b[k]), rel_tol=0, abs_tol=1e-9):
                return False
        elif a[k] != b[k]:
            return False
    return True


def write_csv_new(path, cols, rows):
    with path.open("x", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old-dirs", nargs="+", required=True, type=Path,
                    help="Prior 5-seed route directories, for example pilot_six pilot_depth5")
    ap.add_argument("--new-dir", required=True, type=Path,
                    help="Already prepared ml_train_prepare_v1, with NO routing data yet")
    ap.add_argument("--apply", action="store_true", help="Commit copying cache; otherwise dry run")
    a = ap.parse_args()
    new = a.new_dir.resolve()
    olds = [x.resolve() for x in a.old_dirs]
    if new in olds or len(olds) != len(set(olds)):
        raise ValueError("Old directories must be unique and different from new directory")

    m = load_json(new / "manifest.json")
    if m.get("qiskit") != "0.45.3" or m.get("neighbor_engine") != "fast":
        raise ValueError("Expected prepared validity-aware training set with Qiskit 0.45.3")
    if m.get("status") != "prepared_only_not_labeled":
        raise ValueError("Training directory is not prepare-only")
    _, nc_rows = csv_rows(new / "plan_catalog.csv", ["circuit_id", "sig", "source", "filename", "active_width", "plan_json"])
    nc = index_rows(nc_rows, ("circuit_id", "sig"), "new plan")
    _, selected_rows = csv_rows(new / "selected_candidates.csv", ["circuit_id", "sig", "original_sig"])
    if len(selected_rows) != int(m["selected_source_candidate_rows"]):
        raise ValueError("Selected rows count inconsistent with manifest")
    if set(nc) != {(r["circuit_id"], r["sig"]) for r in selected_rows}:
        raise ValueError("Prepared catalog does not match selected plans")
    if len(nc) != int(m["selected_unique_plans"]):
        raise ValueError("Plan catalog size differs from manifest")
    if set(m["circuits_prepared"]) != set(m["qasm_paths"]):
        raise ValueError("Manifest's prepared circuits differ from QASM path index")
    # Protect the intended new directory from any cache overwrite.
    forbidden = ("routing_results.csv", "compiled_features.csv", "routing_setup.json", "labels.csv", "label_summary.json", "training_cache_transfer_report.json")
    already = [str(new / p) for p in forbidden if (new / p).exists()]
    if already:
        raise FileExistsError("New directory already contains routing/labels/cache. Refusing to modify: " + ", ".join(already))

    cached_routes = {}
    cached_comp = {}
    by_source = []
    for old in olds:
        om = load_json(old / "manifest.json")
        os_ = load_json(old / "routing_setup.json")
        for k, v in SETUP_FIXED.items():
            if os_.get(k) != v:
                raise ValueError(f"Unexpected router setup in {old}: {k}")
        if not set(SEEDS) <= set(map(int, os_.get("seeds", []))):
            raise ValueError(f"{old} lacks complete training seeds 42-46")
        for k in ("qiskit", "routing_config", "original_active_compaction"):
            if om.get(k) != m.get(k):
                raise ValueError(f"Pilot/training mismatch in {k}: {old}")
        # Old directories need only have the six pilot circuits; NOT all training circuits.
        for cid, path in om["qasm_paths"].items():
            if cid in m["qasm_paths"] and path != m["qasm_paths"][cid]:
                raise ValueError(f"QASM path changed for {cid} in {old}")

        _, oc_rows = csv_rows(old / "plan_catalog.csv", ["circuit_id", "sig", "source", "filename", "active_width", "plan_json"])
        oc = index_rows(oc_rows, ("circuit_id", "sig"), f"old plan {old}")
        common = set(nc) & set(oc)
        for key in common:
            if key[0] not in om["qasm_paths"] or key[0] not in m["qasm_paths"]:
                raise ValueError(f"Manifest missing circuit for common plan {key}")
            x, y = oc[key], nc[key]
            if (normalized_plan(x) != normalized_plan(y) or
                int(x["active_width"]) != int(y["active_width"]) or
                x["source"] != y["source"] or x["filename"] != y["filename"]):
                raise ValueError(f"Common plan content/width/source mismatch: {key}")

        _, route_rows = csv_rows(old / "routing_results.csv", ROUTE_COLS)
        old_r = index_rows(route_rows, ("circuit_id", "sig", "seed"), f"routing rows {old}")
        _, comp_rows = csv_rows(old / "compiled_features.csv", COMP_COLS)
        old_c = index_rows(comp_rows, ("circuit_id", "sig"), f"compiled rows {old}")
        for key in sorted(common):
            c = old_c.get(key)
            if c is None or c["status"] != "ok" or int(checked_number(c["dyn_width"],integral=True)) != int(nc[key]["active_width"]):
                raise ValueError(f"Invalid compiled features for {key} in {old}")
            for f in ("logical_depth", "norm_depth"):
                checked_number(c[f])
            if key in cached_comp:
                if not equivalent(cached_comp[key], c, COMP_COLS):
                    raise ValueError(f"Conflicting compiled features between source caches: {key}")
            else:
                cached_comp[key] = c
            for seed in SEEDS:
                rk = (key[0], key[1], str(seed))
                r = old_r.get(rk)
                if r is None or r["status"] != "ok":
                    raise ValueError(f"Missing/failed cache routing {rk} in {old}")
                if r["source"] != nc[key]["source"] or r["filename"] != nc[key]["filename"]:
                    raise ValueError(f"Routing metadata mismatch for {rk}")
                for f in ("swap_count", "routed_depth"):
                    checked_number(r[f])
                if rk in cached_routes:
                    if not equivalent(cached_routes[rk], r, ("circuit_id", "source", "filename", "sig", "seed", "swap_count", "routed_depth", "status", "error")):
                        raise ValueError(f"Conflicting cache routing result for {rk}")
                else:
                    cached_routes[rk] = r
        by_source.append({"directory": str(old), "matching_plans": len(common)})

    matched = set(cached_comp)
    if set((cid, sig) for cid, sig, _ in cached_routes) != matched:
        raise ValueError("Compiled / routing plan coverage disagrees")
    if len(cached_routes) != len(matched) * len(SEEDS):
        raise ValueError("Incomplete cached seed coverage")
    total = len(nc) * len(SEEDS)
    rep = {
        "mode": "DRY_RUN" if not a.apply else "APPLY",
        "training_circuits": len(m["circuits_prepared"]),
        "training_original_candidate_rows": int(m["selected_source_candidate_rows"]),
        "training_unique_plans": len(nc),
        "training_seeds": list(SEEDS),
        "sources": by_source,
        "reusable_unique_plans": len(matched),
        "reusable_routing_jobs": len(cached_routes),
        "new_unique_plans": len(nc) - len(matched),
        "new_routing_jobs_to_run": total - len(cached_routes),
        "training_total_jobs": total,
        "checks": ["same QASM path", "Qiskit & router", "active compaction", "same plan chains", "width", "complete seed set", "cross-cache numeric equality"]
    }
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    if not a.apply:
        print("[DRY RUN] No files changed. Re-run with --apply only after reviewing counts.")
        return
    rows_r = [cached_routes[k] for k in sorted(cached_routes, key=lambda x:(x[0],x[1],int(x[2])))]
    rows_c = [cached_comp[k] for k in sorted(cached_comp)]
    write_csv_new(new / "routing_results.csv", ROUTE_COLS, rows_r)
    write_csv_new(new / "compiled_features.csv", COMP_COLS, rows_c)
    (new / "routing_setup.json").write_text(json.dumps({"seeds": list(SEEDS), **SETUP_FIXED}, indent=2), encoding="utf-8")
    (new / "training_cache_transfer_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Imported {len(cached_routes)} cached routing jobs. Remaining to route: {total - len(cached_routes)}")


if __name__ == "__main__":
    main()
