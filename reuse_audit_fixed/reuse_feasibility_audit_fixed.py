#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reuse feasibility screen for the existing GidNET fixed-width project.

No routing. No model fitting. No claim of semantic equivalence of compiled circuits.

Inputs (place next to this script):
  qubitreuse.py, utils.py, reuse_validity.py, reuse_adversarial.py
  benchmark_shortlist_for_reuse_audit.csv

GidNET is loaded from the supplied original qubitreuse.py. The single known
candidate-matrix bug is fixed ONLY in this audit process: classical wires are
excluded from the n-qubit reuse matrix. The source qubitreuse.py is untouched.
The rest of GidNET is unchanged. The `merge_subsets` helper is extracted from
utils.py without loading unrelated optional QNET dependencies.

Python/Qiskit: use the SAME known-working Qiskit 0.45.x environment as the
existing research experiments. GidNET uses private DAG/retworkx APIs.

Outputs:
  circuit_audit.csv: one row/circuit
  original_plans.csv: all distinct static-valid GidNET plans at best observed width
  neighbor_audit.csv: one row/(original, operator), sampled or exhaustive 1-step counts
  audit_config.json: exact settings

Important semantics:
  - Widths are ACTIVE-qubit widths. Declared idle qubits are compacted first.
  - Only known supported unitary + terminal-measurement circuits are screened.
  - GidNET is stochastic; minimum width is the best OBSERVED width, not proven optimum.
  - Neighbors are one mutation, not multi-step reachability, and may be capped.
  - 'zero neighbors' means no neighbors OBSERVED under the chosen cap unless
    all four operators (or the requested subset) were exhausted.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import random
import sys
import time
import types
from collections import Counter
from pathlib import Path


CIRCUIT_COLUMNS = [
    "source", "filename", "family", "status", "reason", "qasm_path",
    "qubits_declared", "qubits_active", "idle_removed", "gidnet_runs",
    "gidnet_candidate_count", "gidnet_distinct_all_widths", "gidnet_invalid_count",
    "gidnet_valid_count", "best_observed_active_width", "width_reduction",
    "n_originals_at_best_width", "n_originals_audited", "n_valid_neighbors_observed",
    "n_valid_neighbors_unique_across_originals", "neighbor_scan_complete",
    "no_neighbor_proved_within_tested_operators", "elapsed_sec",
]
PLAN_COLUMNS = [
    "source", "filename", "family", "plan_id", "active_width",
    "chains_json", "reuse_edges", "tested_for_neighbors"
]
NEIGHBOR_COLUMNS = [
    "source", "filename", "plan_id", "operator", "attempted", "unique_nontrivial",
    "valid", "invalid", "duplicate", "exhaustive", "valid_plan_sigs_json"
]


def _write_csv(path: Path, names: list[str], rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_merge_subsets_definition(path: Path):
    """Load exact helper function from the supplied utils.py without unrelated imports."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    hits = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "merge_subsets"]
    if len(hits) != 1:
        raise RuntimeError(f"Expected one top-level merge_subsets in {path}")
    mod = ast.Module(body=hits, type_ignores=[])
    ast.fix_missing_locations(mod)
    namespace = {}
    exec(compile(mod, str(path), "exec"), namespace)  # noqa: S102; user's local Python source
    return namespace["merge_subsets"]


def _load_gidnet(qubitreuse_path: Path, utils_path: Path):
    """Load uploaded GidNET as a private Python package, preserving .utils import."""
    package = "_reuse_audit_original_gidnet"
    pkg = types.ModuleType(package)
    pkg.__path__ = [str(qubitreuse_path.parent)]
    sys.modules[package] = pkg

    helper = types.ModuleType(package + ".utils")
    helper.merge_subsets = _read_merge_subsets_definition(utils_path)
    sys.modules[package + ".utils"] = helper

    spec = importlib.util.spec_from_file_location(package + ".qubitreuse", qubitreuse_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import GidNET source from {qubitreuse_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package + ".qubitreuse"] = module
    spec.loader.exec_module(module)

    # IMPORTANT: GidNET's original candidate-matrix builder takes every entry
    # in DAGCircuit.input_map/output_map, which includes CLASSICAL wires.  On
    # circuits that declare c[n] next to q[n], it creates a 2n x 2n matrix and
    # the resulting chains contain out-of-range "qubit" indices n..2n-1.
    # Keep the supplied original source untouched; override only this method
    # in the isolated screening process.  All stochastic plan-generation
    # logic, merging, and operators remain exactly as supplied.
    def _quantum_only_candidate_matrix(self):
        import numpy as np
        import networkx as nx
        from qiskit.converters import circuit_to_dag

        qc = self.circuit.copy()
        qc.remove_final_measurements(inplace=True)
        dag = circuit_to_dag(qc)
        graph = self._to_networkx(dag)
        qubits = list(dag.qubits)
        n = len(qubits)
        if n != self.circuit.num_qubits:
            raise AssertionError("Qubit count changed while building candidate matrix")
        roots = [dag.input_map[q] for q in qubits]
        terminals = [dag.output_map[q] for q in qubits]
        biadjacency_matrix = np.zeros((n, n), dtype=int)
        for i, root in enumerate(roots):
            reachable = nx.descendants(graph, root)
            for j, terminal in enumerate(terminals):
                if root == terminal or terminal in reachable:
                    biadjacency_matrix[i, j] = 1
        candidate_matrix = np.ones((n, n), dtype=int) - biadjacency_matrix.T
        np.fill_diagonal(candidate_matrix, 0)
        if candidate_matrix.shape != (n, n):
            raise AssertionError("GidNET candidate matrix must be n_qubits x n_qubits")
        return biadjacency_matrix, candidate_matrix

    module.GidNET.compute_intial_biadjacency_and_candidate_matrix = _quantum_only_candidate_matrix
    return module.GidNET


def _compact_active(qc):
    """Compact qubits with no gate activity; return (compacted QC, old ID list).

    Reject cases in which discarding idle/measurement-only wires could silently
    change a classical result. Barrier operations are omitted deliberately,
    matching the dependency screen's barrier semantics.
    """
    from qiskit import QuantumCircuit
    from reuse_validity import active_qubits_from_circuit

    active = sorted(active_qubits_from_circuit(qc))
    if not active:
        raise ValueError("no_active_gate_qubits")
    old_to_new = {old: i for i, old in enumerate(active)}
    compact = QuantumCircuit(len(active), qc.num_clbits)
    compact.global_phase = qc.global_phase
    for item in qc.data:
        op = item.operation
        if op.name == "barrier":
            continue
        q_idx = [qc.find_bit(q).index for q in item.qubits]
        if any(q not in old_to_new for q in q_idx):
            raise ValueError("measure_or_instruction_on_inactive_qubit")
        q_args = [compact.qubits[old_to_new[q]] for q in q_idx]
        c_args = [compact.clbits[qc.find_bit(c).index] for c in item.clbits]
        compact.append(op, q_args, c_args)
    return compact, active


def _plan_signature(plan):
    return hashlib.sha256(json.dumps(plan, separators=(",", ":")).encode()).hexdigest()[:24]


def _main_audit(shortlist: Path, out_dir: Path, gidnet_file: Path, utils_file: Path,
                iterations: int, restarts: int, max_originals: int,
                max_mutations_per_op: int, operators: list[str], seed: int,
                max_circuits: int, only_circuits: str = ""):
    from qiskit import QuantumCircuit
    from reuse_validity import ReuseValidityChecker
    from reuse_adversarial import OPERATORS, canonical

    GidNET = _load_gidnet(gidnet_file, utils_file)
    unknown = [op for op in operators if op not in OPERATORS]
    if unknown:
        raise ValueError(f"Unknown operators: {unknown}. Available: {sorted(OPERATORS)}")

    with shortlist.open("r", encoding="utf-8-sig", newline="") as f:
        items = list(csv.DictReader(f))
    if only_circuits:
        requested = {x.strip() for x in only_circuits.split(",") if x.strip()}
        present = {r["filename"] for r in items}
        missing = requested - present
        if missing:
            raise ValueError(f"Requested circuits not found in shortlist: {sorted(missing)}")
        items = [r for r in items if r["filename"] in requested]
    if max_circuits:
        items = items[:max_circuits]
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "iterations": iterations, "restarts": restarts, "max_originals": max_originals,
        "max_mutations_per_op": max_mutations_per_op,
        "operators": operators, "seed": seed, "max_circuits": max_circuits,
        "only_circuits": only_circuits,
        "gidnet_candidate_matrix_fix": "quantum_only_input_output_maps_audit_patch_v1",
        "note": "Best observed active width; first-step neighbor screen, not physical improvement or exact plan-space count",
    }
    (out_dir / "audit_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    circuit_rows, plan_rows, neighbor_rows = [], [], []
    for i, row in enumerate(items):
        t0 = time.perf_counter()
        name = row.get("filename", "").strip()
        source = row.get("source", "").strip()
        p = Path(row.get("original_path", "").strip())
        fam = row.get("family_hint_review_manually", "")
        rec = {"source": source, "filename": name, "family": fam,
               "status": "", "reason": "", "qasm_path": str(p)}
        try:
            if row.get("parser_status") != "parsed":
                rec.update(status="skip_parser", reason=row.get("parser_status", "not_previously_parsed"))
                continue
            if not p.is_file():
                rec.update(status="file_not_found", reason="Check original_path in shortlist CSV")
                continue
            if b"OPENQASM 3" in p.read_bytes()[:2048]:
                rec.update(status="skip_qasm3", reason="Requires a separate QASM3 conversion/validation")
                continue
            qc = QuantumCircuit.from_qasm_file(str(p))
            orig_n = qc.num_qubits
            qc, active_old_ids = _compact_active(qc)
            n = qc.num_qubits
            rec.update(qubits_declared=orig_n, qubits_active=n,
                       idle_removed=orig_n - n, gidnet_runs=restarts)

            checker = ReuseValidityChecker(qc)
            if checker.unsupported is not None:
                rec.update(status="skip_unsupported", reason=checker.unsupported.reason)
                continue

            # The identity (no-reuse) plan is always a valid fallback.
            # Include it so an irreducible circuit becomes no_reuse_observed,
            # rather than being confused with a GidNET failure.
            identity = canonical([[q] for q in range(n)])
            valid = {identity: n} if checker.validate(identity).is_valid else {}
            count_invalid = 0
            count_generated = 0
            for rep in range(restarts):
                # GidNET uses the global stdlib RNG; reproduce each restart exactly.
                random.seed(seed + i * 10007 + rep)
                generator = GidNET(qc)
                if generator.candidate_matrix_copy.shape != (n, n):
                    raise AssertionError(
                        f"Wrong candidate matrix size {generator.candidate_matrix_copy.shape} "
                        f"for {n} active qubits")
                # Before accepting any GidNET plan, independently verify that
                # the repaired quantum-only matrix matches the validity checker.
                for a in range(n):
                    for b in range(n):
                        expected = a != b and (a, b) not in checker._reachable_to_output
                        if bool(generator.candidate_matrix_copy[a, b]) != expected:
                            raise AssertionError(
                                f"Candidate/Checker disagreement at ({a},{b}) in {name}")
                best = generator.compute_optimized_reuse_sequences(iterations=iterations)
                candidates = list(generator.list_of_computed_reuse_sequences)
                if best is not None:
                    candidates.append(best)
                for chains in candidates:
                    if chains is None:
                        continue
                    count_generated += 1
                    try:
                        plan = canonical(chains)
                        v = checker.validate(plan)
                        if not v.is_valid:
                            count_invalid += 1
                            continue
                        valid[plan] = len(plan)
                    except (ValueError, TypeError, IndexError):
                        count_invalid += 1
            rec.update(gidnet_candidate_count=count_generated,
                       gidnet_distinct_all_widths=len(valid),
                       gidnet_invalid_count=count_invalid, gidnet_valid_count=len(valid))
            if not valid:
                rec.update(status="no_valid_gidnet_output", reason="No static-valid generated plan")
                continue
            best_width = min(valid.values())
            best_plans = sorted(p for p, w in valid.items() if w == best_width)
            rec.update(best_observed_active_width=best_width,
                       width_reduction=n - best_width,
                       n_originals_at_best_width=len(best_plans))
            if best_width == n:
                rec.update(status="no_reuse_observed", reason="Generator found only singleton chains")
                # Still record all valid original plans.
            else:
                rec.update(status="screened", reason="static_valid_only")

            to_check = best_plans[:max_originals]
            rec["n_originals_audited"] = len(to_check)
            all_valid_neighbor_sigs = set()
            all_exhaustive = True
            n_valid_summed = 0
            for j, plan in enumerate(best_plans):
                sig = _plan_signature(plan)
                plan_rows.append(dict(source=source, filename=name, family=fam,
                                      plan_id=sig, active_width=best_width,
                                      chains_json=json.dumps(plan, separators=(",", ":")),
                                      reuse_edges=n - best_width,
                                      tested_for_neighbors=(j < len(to_check))))

            for plan in to_check:
                sig = _plan_signature(plan)
                for operator in operators:
                    seen = {plan}
                    attempts = unique = good = bad = dup = 0
                    found_sigs = set()
                    exhaustive = True
                    # Generator yields lazily to avoid materializing huge neighborhoods.
                    for proposal in OPERATORS[operator](plan):
                        if attempts >= max_mutations_per_op:
                            exhaustive = False
                            break
                        attempts += 1
                        try:
                            candidate = canonical(proposal)
                        except (ValueError, TypeError):
                            bad += 1
                            continue
                        if candidate in seen:
                            dup += 1
                            continue
                        seen.add(candidate)
                        unique += 1
                        result = checker.validate(candidate, target_width=best_width)
                        if result.is_valid:
                            good += 1
                            found_sigs.add(_plan_signature(candidate))
                        else:
                            bad += 1
                    n_valid_summed += good
                    all_valid_neighbor_sigs.update(found_sigs)
                    all_exhaustive &= exhaustive
                    neighbor_rows.append(dict(
                        source=source, filename=name, plan_id=sig, operator=operator,
                        attempted=attempts, unique_nontrivial=unique, valid=good,
                        invalid=bad, duplicate=dup, exhaustive=int(exhaustive),
                        valid_plan_sigs_json=json.dumps(sorted(found_sigs))))
            rec["n_valid_neighbors_observed"] = n_valid_summed
            rec["n_valid_neighbors_unique_across_originals"] = len(all_valid_neighbor_sigs)
            rec["neighbor_scan_complete"] = int(all_exhaustive)
            rec["no_neighbor_proved_within_tested_operators"] = int(
                all_exhaustive and len(all_valid_neighbor_sigs) == 0 and len(to_check) == len(best_plans))

        except Exception as exc:
            rec.update(status="error", reason=f"{type(exc).__name__}: {str(exc)[:300]}")
        finally:
            rec["elapsed_sec"] = round(time.perf_counter() - t0, 3)
            circuit_rows.append(rec)
            print(f"[{i+1}/{len(items)}] {source}/{name}: {rec['status']}; "
                  f"width={rec.get('best_observed_active_width', '-')}, "
                  f"neighbors={rec.get('n_valid_neighbors_unique_across_originals','-')}, "
                  f"t={rec['elapsed_sec']}s", flush=True)
            # Checkpoint all results after each circuit: no loss on interruption.
            _write_csv(out_dir / "circuit_audit.csv", CIRCUIT_COLUMNS, circuit_rows)
            _write_csv(out_dir / "original_plans.csv", PLAN_COLUMNS, plan_rows)
            _write_csv(out_dir / "neighbor_audit.csv", NEIGHBOR_COLUMNS, neighbor_rows)
    print("\nCompleted. Statuses:", dict(Counter(row["status"] for row in circuit_rows)))
    print("Outputs:", out_dir.resolve())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--shortlist", type=Path,
                    default=here / "benchmark_shortlist_for_reuse_audit.csv")
    ap.add_argument("--out-dir", type=Path, default=here / "reuse_audit_out")
    ap.add_argument("--gidnet-file", type=Path, default=here / "qubitreuse.py")
    ap.add_argument("--utils-file", type=Path, default=here / "utils.py")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--max-originals", type=int, default=3)
    ap.add_argument("--max-mutations-per-op", type=int, default=300)
    ap.add_argument("--operators", default="suffix_swap,chain_transfer")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-circuits", type=int, default=0,
                    help="Use 1 or 2 for a quick smoke test; 0 for all shortlist rows")
    ap.add_argument("--circuits", default="",
                    help="Comma-separated QASM filenames for targeted diagnosis")
    a = ap.parse_args()
    if min(a.iterations, a.restarts, a.max_originals, a.max_mutations_per_op) <= 0:
        ap.error("iterations/restarts/max-originals/max-mutations-per-op must be > 0")
    for path in (a.shortlist, a.gidnet_file, a.utils_file,
                 here / "reuse_validity.py", here / "reuse_adversarial.py"):
        if not path.is_file():
            ap.error(f"Required file not found: {path}")
    _main_audit(a.shortlist, a.out_dir, a.gidnet_file, a.utils_file,
                a.iterations, a.restarts, a.max_originals, a.max_mutations_per_op,
                [s.strip() for s in a.operators.split(",") if s.strip()],
                a.seed, a.max_circuits, a.circuits)


if __name__ == "__main__":
    main()
