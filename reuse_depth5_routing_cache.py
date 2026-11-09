#!/usr/bin/env python3
"""Reuse already-routed fixed-width plans when comparing depth-3 and depth-5 pilots.

Use AFTER prepare, BEFORE route. Conservative: refuses to modify a new pilot
that already contains any routing/cache outputs. Never changes the old pilot.

Example:
  python reuse_depth5_routing_cache.py --old-dir pilot_six --new-dir pilot_depth5
  python reuse_depth5_routing_cache.py --old-dir pilot_six --new-dir pilot_depth5 --apply
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path

ROUTE_COLS = ("circuit_id", "source", "filename", "sig", "seed", "swap_count", "routed_depth", "routing_time_sec", "status", "error")
COMP_COLS = ("circuit_id", "sig", "logical_depth", "norm_depth", "dyn_width", "status", "error")


def csv_read(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV header: {path}")
        return tuple(reader.fieldnames), list(reader)


def csv_write_exclusive(path: Path, fields, rows):
    # 'x' refuses overwrite; consistent with the pilot's utf-8-sig CSV format.
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def only_once(rows, cols, description):
    res = {}
    for row in rows:
        key = tuple(row[k] for k in cols)
        if key in res:
            raise ValueError(f"Duplicate {description}: {key}")
        res[key] = row
    return res


def plan_content(row):
    chains = json.loads(row["plan_json"])
    # Chain order is irrelevant, but qubit order inside a chain is not.
    return tuple(sorted(tuple(int(x) for x in chain) for chain in chains))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--old-dir", type=Path, required=True)
    ap.add_argument("--new-dir", type=Path, required=True)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--apply", action="store_true", help="Actually copy matching cached rows (dry-run by default)")
    args = ap.parse_args()
    old, new = args.old_dir.resolve(), args.new_dir.resolve()
    if old == new:
        raise ValueError("Old and new directories must differ")
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("Seeds must be distinct integers")

    old_manifest, new_manifest = read_json(old / "manifest.json"), read_json(new / "manifest.json")
    old_setup = read_json(old / "routing_setup.json")
    if not set(seeds) <= set(map(int, old_setup.get("seeds", []))):
        raise ValueError("Old routing setup does not cover all required seeds")
    if old_setup.get("qiskit") != "0.45.3" or new_manifest.get("qiskit") != "0.45.3":
        raise ValueError("Qiskit 0.45.3 expected")
    if old_manifest.get("qiskit") != new_manifest.get("qiskit"):
        raise ValueError("Qiskit versions differ")
    if old_manifest.get("routing_config") != new_manifest.get("routing_config"):
        raise ValueError("Routing configurations in manifests differ")
    if old_manifest.get("original_active_compaction") != new_manifest.get("original_active_compaction"):
        raise ValueError("Active-qubit compaction settings differ")
    expected = {"router": "router_template:route", "backend": "Heavy-hex distance 7",
                "layout_method": "sabre", "routing_method": "sabre", "optimization_level": 1}
    if any(old_setup.get(k) != value for k, value in expected.items()):
        raise ValueError("Old router configuration is not the expected heavy-hex SABRE configuration")
    for cid, path in new_manifest["qasm_paths"].items():
        if old_manifest["qasm_paths"].get(cid) != path:
            raise ValueError(f"QASM path mismatch for {cid}")

    _, old_catalog_rows = csv_read(old / "plan_catalog.csv")
    _, new_catalog_rows = csv_read(new / "plan_catalog.csv")
    old_catalog = only_once(old_catalog_rows, ("circuit_id", "sig"), "old plan")
    new_catalog = only_once(new_catalog_rows, ("circuit_id", "sig"), "new plan")
    old_keys, new_keys = set(old_catalog), set(new_catalog)
    shared = old_keys & new_keys
    for key in shared:
        if plan_content(old_catalog[key]) != plan_content(new_catalog[key]):
            raise ValueError(f"Plan contents differ despite identical circuit and signature: {key}")
        if old_catalog[key]["active_width"] != new_catalog[key]["active_width"]:
            raise ValueError(f"Plan width mismatch: {key}")
    _, selected = csv_read(new / "selected_candidates.csv")
    if len(selected) != int(new_manifest["selected_source_candidate_rows"]):
        raise ValueError("New selected candidate row count mismatches manifest")
    if {(r['circuit_id'],r['sig']) for r in selected} != new_keys:
        raise ValueError("New selected candidates do not match new plan catalog")

    header_routes, raw_routes = csv_read(old / "routing_results.csv")
    if not set(ROUTE_COLS) <= set(header_routes):
        raise ValueError("Old routing results missing expected columns")
    keyed_routes = only_once(raw_routes, ("circuit_id", "sig", "seed"), "old routing job")
    selected_routes=[]
    for cid,sig in sorted(shared):
        for seed in seeds:
            key = (cid, sig, str(seed))
            if key not in keyed_routes:
                raise ValueError(f"Old routing missing shared plan/seed {key}")
            row=keyed_routes[key]
            if row['status'] != 'ok' or not all(math.isfinite(float(row[field])) for field in ('swap_count','routed_depth')):
                raise ValueError(f"Old cached routing row is not valid: {key}")
            if row["source"] != new_catalog[(cid,sig)]["source"] or row["filename"] != new_catalog[(cid,sig)]["filename"]:
                raise ValueError(f"Routing source/filename mismatch for {key}")
            selected_routes.append(row)

    header_comp, raw_comp = csv_read(old / "compiled_features.csv")
    if not set(COMP_COLS) <= set(header_comp):
        raise ValueError("Old compiled features missing expected columns")
    compiled_map = only_once(raw_comp, ("circuit_id", "sig"), "old compiled plan")
    cached_comp=[]
    for key in sorted(shared):
        if key not in compiled_map or compiled_map[key]['status'] != 'ok':
            raise ValueError(f"Missing/failed compiled features for {key}")
        if int(float(compiled_map[key]['dyn_width'])) != int(new_catalog[key]['active_width']):
            raise ValueError(f"Compiled width differs from plan width for {key}")
        cached_comp.append(compiled_map[key])

    missing_plans = new_keys - shared
    expected_more = len(missing_plans)*len(seeds)
    report = dict(old_unique_plans=len(old_keys), new_unique_plans=len(new_keys),
                  overlapping_plans=len(shared), new_plans_needing_routing=len(missing_plans),
                  reused_routing_rows=len(selected_routes), new_routing_jobs=expected_more,
                  total_new_pilot_routing_jobs=len(new_keys)*len(seeds),
                  cached_compiled_plans=len(cached_comp), seeds=seeds,
                  new_dir=str(new), old_dir=str(old),
                  strict_match=["qasm_paths","qiskit","routing_config","original_active_compaction",
                                "plan_json","active_width","source","filename","seed_set"])
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not args.apply:
        print("[DRY RUN] No files changed. If counts are correct, rerun with --apply.")
        return

    targets = [new / p for p in ('routing_results.csv','compiled_features.csv','routing_setup.json','cache_transfer_report.json')]
    blocked = [str(p) for p in targets if p.exists()]
    if blocked:
        raise FileExistsError("To protect prior work, refusing to overwrite existing new-pilot files: " + ', '.join(blocked))
    if (new / 'labels.csv').exists() or (new / 'label_summary.json').exists():
        raise FileExistsError("New pilot already has labels; refusing to backfill cache without review")
    # The routing stage reads setup and rows before resuming. Copy only the plans
    # in the new catalog, with all requested seeds, then it runs the remaining jobs.
    csv_write_exclusive(new / 'routing_results.csv', header_routes, selected_routes)
    csv_write_exclusive(new / 'compiled_features.csv', header_comp, cached_comp)
    with (new / 'routing_setup.json').open('x', encoding='utf-8') as f:
        json.dump(old_setup, f, ensure_ascii=False, indent=2)
    with (new / 'cache_transfer_report.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[OK] Cached {len(selected_routes)} routing rows and {len(cached_comp)} compiled plans.")
    print(f"[NEXT] Run normal route in {new} with seeds {','.join(map(str,seeds))}; expected {expected_more} jobs remaining.")

if __name__ == '__main__':
    main()
