#!/usr/bin/env python3
"""Validate that a training-only multi-step prepare never touches validation/final-test circuits.

No routing and no Qiskit required. Reads only csv/json files produced by prepare.
"""
import argparse, csv, json
from collections import Counter, defaultdict
from pathlib import Path


def rows(p):
    with p.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split-dir", type=Path, required=True)
    p.add_argument("--prepare-dir", type=Path, required=True)
    a=p.parse_args()
    split=rows(a.split_dir / "split_ledger.csv")
    eligible={r["circuit_id"] for r in split if r["split"]=="train" and r["audit_status"]=="screened"}
    blocked={r["circuit_id"] for r in split if r["split"]!="train"}
    manifest=json.loads((a.prepare_dir / "manifest.json").read_text(encoding="utf-8"))
    issues=json.loads((a.prepare_dir / "prepare_errors.json").read_text(encoding="utf-8"))
    selected=rows(a.prepare_dir / "selected_candidates.csv")
    summary=json.loads((a.prepare_dir / "source_summary.json").read_text(encoding="utf-8"))
    catalog=rows(a.prepare_dir / "plan_catalog.csv")
    features=rows(a.prepare_dir / "features.csv")
    pairs=rows(a.prepare_dir / "feature_pairs.csv")
    requested=set(manifest.get("circuits_requested", []))
    prepared=set(manifest.get("circuits_prepared", []))
    selected_ids={r["circuit_id"] for r in selected}
    observed = requested | prepared | selected_ids | {r["circuit_id"] for r in catalog + features + pairs + summary}
    errors=[]
    if observed & blocked: errors.append(f"LEAKAGE: nontrain in prepare: {sorted(observed & blocked)}")
    if requested != eligible: errors.append(f"Requested not identical to 28 screened train circuits: missing={sorted(eligible-requested)}, extra={sorted(requested-eligible)}")
    if prepared != eligible: errors.append(f"Prepared mismatch: missing={sorted(eligible-prepared)}, extra={sorted(prepared-eligible)}")
    if issues: errors.append(f"Prepare errors: {len(issues)}: {issues[:2]}")
    if set((r["circuit_id"],r["sig"]) for r in catalog) != set((r["circuit_id"],r["sig"]) for r in features):
        errors.append("Features keys not identical to unique plan catalog keys")
    if len(pairs)!=len(selected): errors.append("Pairs count differs from selected count")
    if {(r["circuit_id"],r["original_sig"],r["sig"]) for r in pairs} != {(r["circuit_id"],r["original_sig"],r["sig"]) for r in selected}:
        errors.append("Pair keys do not match selected candidates")
    if len({(r["circuit_id"],r["sig"]) for r in catalog}) != len(catalog):
        errors.append("Duplicate catalog keys")
    depth=Counter(int(r["discovery_move_depth"]) for r in selected)
    per_circuit=defaultdict(lambda: Counter())
    for r in selected:
        per_circuit[r["circuit_id"]]["selected"]+=1
        per_circuit[r["circuit_id"]]["candidate" if r["is_original"]=="0" else "original"]+=1
    results={
        "status":"PASS" if not errors else "FAIL",
        "train_eligible_circuits":len(eligible),
        "prepared_circuits":len(prepared),
        "selected_rows":len(selected),
        "distinct_plan_signatures_within_circuit":len(catalog),
        "source_originals":sum(c["original"] for c in per_circuit.values()),
        "candidate_rows":sum(c["candidate"] for c in per_circuit.values()),
        "selected_depth_distribution":dict(sorted(depth.items())),
        "preparation_errors":len(issues),
        "errors":errors,
        "per_circuit":{k:dict(v) for k,v in sorted(per_circuit.items())}
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if errors: raise SystemExit(2)

if __name__=="__main__":main()
