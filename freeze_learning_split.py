#!/usr/bin/env python3
"""Freeze leakage-aware circuit/family split using corrected structural audit CSVs only.

No Qiskit, routing costs, labels, or feature outcomes are read.
Usage:
 python freeze_learning_split.py plan --audit-dirs ../fixed_full ../wave2 --out-dir ./ml_split_v1
 python freeze_learning_split.py verify --audit-dirs ../fixed_full ../wave2 --out-dir ./ml_split_v1

The 5 final-test families and 5 validation families are fixed, predeclared choices;
all other families are TRAIN. Never revise the assignments using physical outcomes.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

PROTOCOL = "fixed_width_family_split_v1_2026_11_04"
# Families from wave2, without prior pilot physical-label inspection. Structural screening
# to establish eligibility (reuse exists) was allowed; routing outcomes were not used.
FINAL_GROUPS = frozenset({"qram", "squar5", "wim", "hs4", "qec_en"})
VALID_GROUPS = frozenset({"ising_model", "dnn", "four_gt", "modular_arithmetic", "ex1"})
PILOT_IDS = frozenset({
    "CaQR::cnt3-5_179.qasm", "CaQR::sym6_316.qasm",
    "CaQR::multiply_n13.qasm", "QASMBench::multiply_n13.qasm",
    "QASMBench::gcm_h6.qasm", "QASMBench::lpn_n5.qasm"
})


def group_family(family: str) -> str:
    """Conservative semantic aliasing, prevents source/name-family leakage."""
    f = str(family).strip().lower().replace("-", "_")
    if f in {"multiply_n13", "multiply", "multiplier"}:
        return "multiplication"
    if f in {"sym6", "sym9"}:
        return "symmetry"
    if f in {"system9", "sys6"}:
        return "system"
    if f in {"alu", "mini_alu"}:
        return "alu"
    if f.startswith("rd") and f[2:].isdigit():
        return "rd"
    if f.startswith("cm") and f[2:].replace("a", "").isdigit():
        return "cm"
    if f.startswith("4gt"):
        return "four_gt"
    if f in {"4mod5", "mod5mils"}:
        return "modular_arithmetic"
    if f in {"bv", "bv_n10"}:
        return "bernstein_vazirani"
    if f in {"cat_state", "ghz_state", "bell"}:
        return "entangled_states"
    if f.startswith("cnt3_5"):
        return "cnt3_5"
    return f


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_audits(dirs):
    records, inputs = [], []
    for wave, d in zip(("fixed_full", "wave2"), dirs):
        p = d / "circuit_audit.csv"
        if not p.is_file():
            raise FileNotFoundError(f"Missing corrected audit file: {p}")
        inputs.append({"wave": wave, "file": "circuit_audit.csv", "sha256": sha256(p)})
        with p.open(newline="", encoding="utf-8-sig") as h:
            for r in csv.DictReader(h):
                records.append((wave, r))
    if len(records) != 87:
        raise ValueError(f"Expected 87 structurally shortlisted records; found {len(records)}. Stop and review.")
    return records, inputs


def make_ledger(audit_entries):
    rows, ids = [], set()
    for wave, r in audit_entries:
        cid = r["source"] + "::" + r["filename"]
        if cid in ids:
            raise ValueError(f"Duplicate circuit ID across waves: {cid}")
        ids.add(cid)
        grp = group_family(r["family"])
        split = "final_test" if grp in FINAL_GROUPS else ("validation" if grp in VALID_GROUPS else "train")
        status = r["status"]
        if status == "screened":
            n = int(float(r.get("n_valid_neighbors_unique_across_originals") or 0))
            gate = "reuse_with_observed_neighbor" if n else "reuse_no_neighbor_in_bounded_screen"
        elif status == "no_reuse_observed":
            gate = "no_reuse_observed_in_bounded_generator_runs"
            n = 0
        elif status in {"skip_parser", "error"}:
            gate = "unsupported_or_error"
            n = 0
        else:
            raise ValueError(f"Unexpected status {status}: {cid}")
        rows.append({
            "circuit_id": cid, "split": split, "family_group": grp,
            "wave": wave, "source": r["source"], "filename": r["filename"],
            "audit_family": r["family"], "audit_status": status,
            "eligibility": gate, "observed_valid_neighbors": n,
            "qubits_active": r.get("qubits_active") or "",
            "best_observed_active_width": r.get("best_observed_active_width") or "",
            "qasm_path": r.get("qasm_path") or "",
            "prior_pilot_physical_results": int(cid in PILOT_IDS),
        })
    rows.sort(key=lambda r: (r["split"], r["family_group"], r["circuit_id"]))
    by_group = defaultdict(set)
    for r in rows:
        by_group[r["family_group"]].add(r["split"])
    violations = {g:sorted(v) for g,v in by_group.items() if len(v)>1}
    if violations:
        raise AssertionError(f"Family leakage: {violations}")
    if any(r["split"] != "train" for r in rows if r["prior_pilot_physical_results"]):
        raise AssertionError("Prior pilot circuits must not appear in validation/final_test")
    if any(r["wave"] != "wave2" for r in rows if r["split"]=="final_test"):
        raise AssertionError("Final-test reservations must be wave2 only")
    if {r["family_group"] for r in rows if r["split"]=="final_test"} != FINAL_GROUPS:
        raise AssertionError("Missing a designated final-test family")
    if not any(r["split"]=="validation" and r["eligibility"]=="reuse_with_observed_neighbor" for r in rows):
        raise AssertionError("Validation set has no positive examples")
    if not any(r["split"]=="final_test" and r["eligibility"]=="reuse_with_observed_neighbor" for r in rows):
        raise AssertionError("Final-test set has no positive examples")
    return rows

FIELDS = ["circuit_id","split","family_group","wave","source","filename", "audit_family",
          "audit_status","eligibility","observed_valid_neighbors","qubits_active",
          "best_observed_active_width","qasm_path","prior_pilot_physical_results"]


def jsonblob(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def lock_digest(rows):
    return hashlib.sha256(jsonblob(rows).encode("utf-8")).hexdigest()


def stats(rows):
    by = {}
    for split in ("train", "validation", "final_test"):
        rr = [r for r in rows if r["split"]==split]
        by[split] = {"all_shortlisted":len(rr),
                     "reuse_eligible":sum(r["audit_status"]=="screened" for r in rr),
                     "observed_with_neighbor":sum(r["eligibility"]=="reuse_with_observed_neighbor" for r in rr),
                     "reuse_but_no_neighbor_observed":sum(r["eligibility"]=="reuse_no_neighbor_in_bounded_screen" for r in rr),
                     "no_reuse_observed":sum(r["audit_status"]=="no_reuse_observed" for r in rr),
                     "unsupported":sum(r["eligibility"]=="unsupported_or_error" for r in rr),
                     "distinct_family_groups":len({r["family_group"] for r in rr}),
                     "prior_pilot_physical_results":sum(int(r["prior_pilot_physical_results"]) for r in rr)}
    return by


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["plan","verify"])
    p.add_argument("--audit-dirs", type=Path, nargs=2, required=True, metavar=("FIXED_FULL", "WAVE2"))
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args()
    audits, inputs=read_audits(a.audit_dirs)
    rows=make_ledger(audits)
    counts=stats(rows)
    lock = {"protocol":PROTOCOL,
            "assignment_sha256":lock_digest(rows),
            "structural_audit_sha256":inputs,
            "holdout_family_groups":{"validation":sorted(VALID_GROUPS),"final_test":sorted(FINAL_GROUPS)},
            "known_development_pilot_ids":sorted(PILOT_IDS),
            "notes":["Only corrected structural audits were used for split allocation; no routing outcomes read.",
                     "All members of one canonical family remain in the same split.",
                     "The six pilot circuits are training/development, not unbiased test circuits.",
                     "No-reuse-observed means under bounded generator runs only; not proof of impossibility.",
                     "No-neighbor-observed means under bounded two-operator structural screening only.",
                     "Final-test families are reserved and must not be routed or used to tune methods until policy is frozen.",
                     "Validation families are for hyperparameter/threshold/model selection; never fit them.",
                     "Two physically different files named multiply_n13 are always keyed by source::filename."]}
    if a.stage=="verify":
        q=a.out_dir / "split_lock.json"
        f=a.out_dir / "split_ledger.csv"
        if not q.is_file() or not f.is_file():
            raise FileNotFoundError("Split not found. Run plan first.")
        got=json.loads(q.read_text(encoding="utf-8"))
        if got != lock:
            raise RuntimeError("LOCK MISMATCH: audit content or split rules differ from the frozen version.")
        with f.open(encoding="utf-8-sig",newline="") as h:
            rr=list(csv.DictReader(h))
        if jsonblob(rr)!=jsonblob([{k:str(v) for k,v in r.items()} for r in rows]):
            raise RuntimeError("LEDGER MISMATCH: split_ledger.csv was changed after freezing.")
        print("PASS: frozen split and audit hashes match; family overlap=0; pilot leakage=0")
        print(json.dumps(counts,ensure_ascii=False,indent=2))
        return
    a.out_dir.mkdir(parents=True,exist_ok=True)
    existing=[a.out_dir / x for x in ("split_lock.json","split_ledger.csv","split_summary.json")]
    if any(x.exists() for x in existing):
        raise FileExistsError("Split already exists. Run verify; NEVER overwrite an existing allocation.")
    with (a.out_dir / "split_ledger.csv").open("w",encoding="utf-8-sig",newline="") as h:
        wr=csv.DictWriter(h,fieldnames=FIELDS,lineterminator="\n")
        wr.writeheader();wr.writerows(rows)
    (a.out_dir / "split_lock.json").write_text(json.dumps(lock,ensure_ascii=False,indent=2),encoding="utf-8")
    (a.out_dir / "split_summary.json").write_text(json.dumps({"protocol":PROTOCOL,"splits":counts,"unique_circuit_ids":len(rows)},ensure_ascii=False,indent=2),encoding="utf-8")
    for split in ("train","validation","final_test"):
        selected=[r["circuit_id"] for r in rows if r["split"]==split and r["audit_status"]=="screened"]
        (a.out_dir / f"{split}_screened_circuits.txt").write_text("\n".join(selected)+"\n",encoding="utf-8")
    print("FROZEN:",a.out_dir)
    print("sha256:",lock["assignment_sha256"])
    print(json.dumps(counts,ensure_ascii=False,indent=2))
    print("No routing, training, or QASM parsing performed.")

if __name__=="__main__":
    main()
