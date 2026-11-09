#!/usr/bin/env python3
"""Post-hoc Beam search diagnosis: search coverage vs candidate selection.

Inputs from adaptive_repair_traced.py --trace-candidates:
  adaptive_traced/candidate_trace.csv
  adaptive_traced/per_source.csv

No Qiskit, routing, mutation, fitting or model prediction is performed.
'Best seen' uses cached HELDOUT routing labels after the search: an oracle
DIAGNOSTIC, not an executable selection policy or a publishable method result.

Run from your GidNET project root:
  python diagnose_beam_selection_gap.py
or:
  python diagnose_beam_selection_gap.py --trace adaptive_traced/candidate_trace.csv --selected adaptive_traced/per_source.csv --out-dir beam_selection_gap

Use --self-test to validate independently of project files.
Requires: pandas, numpy.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["objective", "policy", "circuit", "source_plan_id", "source_sig"]
TRACE_REQUIRED = KEY + ["candidate_sig", "predicted_gain_pct", "actual_gain_pct"]
SELECTED_REQUIRED = KEY + ["pick_sig", "actual_gain_pct"]
EPS = 1e-6


def check_columns(frame: pd.DataFrame, required: list[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {', '.join(missing)}")


def read_csv(path: Path, required: list[str], label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}\n"
                                 "If candidate_trace.csv is missing, rerun the existing "
                                 "adaptive_repair_traced.py with --trace-candidates; "
                                 "it reads saved routing costs and does not reroute.")
    # Important: signatures and source IDs must remain strings.
    frame = pd.read_csv(path, dtype={k: "string" for k in KEY + ["candidate_sig", "pick_sig"]}, low_memory=False)
    check_columns(frame, required, label)
    for c in ["predicted_gain_pct", "actual_gain_pct", "n_scored", "base_held", "pick_held"]:
        if c in frame.columns:
            frame[c] = pd.to_numeric(frame[c], errors="coerce")
    for c in KEY:
        if frame[c].isna().any():
            raise ValueError(f"Missing key '{c}' in {label}")
    return frame


def make_results(selected: pd.DataFrame, trace: pd.DataFrame):
    if selected.duplicated(KEY).any():
        bad = selected.loc[selected.duplicated(KEY, keep=False), KEY].head(4)
        raise ValueError("Duplicate selected search keys: possibly merged reruns without a run_id.\n"
                         + bad.to_string(index=False))
    if trace.duplicated(KEY + ["candidate_sig"]).any():
        # Repeated rows sometimes arise after concatenation. Verify labels match.
        dup = trace[trace.duplicated(KEY + ["candidate_sig"], keep=False)]
        conflicts = dup.groupby(KEY + ["candidate_sig"], dropna=False)["actual_gain_pct"].nunique(dropna=True)
        if (conflicts > 1).any():
            raise ValueError("Repeated candidate with conflicting heldout gains")
        trace = trace.drop_duplicates(KEY + ["candidate_sig"], keep="first").copy()

    if trace["candidate_sig"].isna().any():
        raise ValueError("Candidate signature is missing in trace")

    trace_keys = trace[KEY].drop_duplicates()
    unmatched = trace_keys.merge(selected[KEY], on=KEY, how="left", indicator=True)
    n_extra = int((unmatched["_merge"] == "left_only").sum())
    if n_extra:
        raise ValueError(f"{n_extra} trace search groups have no matching row in per_source.csv. "
                         "Check these inputs came from the SAME traced run.")

    lookup = {key: g for key, g in trace.groupby(KEY, sort=False, dropna=False)}
    summary_rows = []
    rank_frames = []
    for r in selected.itertuples(index=False):
        src = {c: getattr(r, c) for c in KEY}
        key = tuple(src.values())
        cands = lookup.get(key)
        if cands is None:
            cands = trace.iloc[:0].copy()
        else:
            cands = cands.copy()
        n_scored = getattr(r, "n_scored", np.nan)
        if math.isfinite(n_scored) and int(n_scored) != len(cands):
            raise ValueError(f"Candidate trace does not match n_scored for {src}: "
                             f"n_scored={int(n_scored)}, traced_unique={len(cands)}. "
                             "Check you did not mix runs or truncate the trace.")

        base_sig = src["source_sig"]
        pick_sig = str(getattr(r, "pick_sig"))
        selected_actual = float(getattr(r, "actual_gain_pct"))
        if not math.isfinite(selected_actual):
            raise ValueError(f"Missing actual selected gain for {src}")
        if pick_sig == base_sig and abs(selected_actual) > 1e-4:
            raise ValueError(f"Original plan selected but actual gain not 0 for {src}")
        if pick_sig != base_sig:
            pick_rows = cands.loc[cands["candidate_sig"] == pick_sig]
            if len(pick_rows) != 1:
                raise ValueError(f"Selected plan {pick_sig} absent from candidate trace for {src}; "
                                 "trace and per_source are likely from different runs.")
            pick_label = float(pick_rows.iloc[0]["actual_gain_pct"])
            if not math.isfinite(pick_label) or abs(pick_label - selected_actual) > 1e-3:
                raise ValueError(f"Heldout label mismatch for selected plan {src}; "
                                 f"trace={pick_label}, selected={selected_actual}")

        valid = cands[cands["actual_gain_pct"].notna() & np.isfinite(cands["actual_gain_pct"])].copy()
        valid = valid.sort_values(["actual_gain_pct", "candidate_sig"], ascending=[False, True])
        if len(valid) and float(valid.iloc[0]["actual_gain_pct"]) > EPS:
            best_sig = str(valid.iloc[0]["candidate_sig"])
            best_seen = float(valid.iloc[0]["actual_gain_pct"])
            best_pred = float(valid.iloc[0]["predicted_gain_pct"])
        else:
            best_sig, best_seen, best_pred = base_sig, 0.0, 0.0
        gap = best_seen - selected_actual
        if gap < -1e-3:
            raise ValueError(f"Selected actual gain exceeds best observed candidate for {src}: "
                             "trace coverage or source identity is inconsistent")
        if abs(gap) < EPS:
            gap = 0.0
        n_unlabeled = int(len(cands) - len(valid))
        coverage = float(len(valid) / len(cands)) if len(cands) else 1.0
        n_better = int((valid["actual_gain_pct"] > selected_actual + EPS).sum())
        selected_rank = n_better + (1 if selected_actual >= -EPS else 2)
        # If selected has negative actual gain, original (0%) also outranks it.
        rec = dict(**src,
                   selected_sig=pick_sig,
                   selected_gain_pct=selected_actual,
                   best_seen_sig=best_sig,
                   best_seen_gain_pct=best_seen,
                   best_seen_predicted_gain_pct=best_pred,
                   selection_gap_pp=max(0.0, gap),
                   selected_is_best_seen=(gap <= EPS),
                   selected_is_original=(pick_sig == base_sig),
                   selected_is_harmful=(selected_actual < -EPS),
                   positive_candidate_found=(best_seen > EPS),
                   selected_rank_among_seen_plus_original=selected_rank,
                   n_candidates_traced=len(cands),
                   n_candidates_with_heldout=len(valid),
                   n_candidates_without_heldout=n_unlabeled,
                   label_coverage=coverage,
                   n_candidates_beating_selection=n_better,
                   n_scored_reported=n_scored,
                   selection_predicted_gain_pct=getattr(r, "predicted_gain_pct", np.nan))
        if "base_held" in selected.columns:
            rec["original_heldout_cost"] = getattr(r, "base_held")
        if "pick_held" in selected.columns:
            rec["selected_heldout_cost"] = getattr(r, "pick_held")
        summary_rows.append(rec)

        if len(cands):
            cands["is_selected"] = (cands["candidate_sig"] == pick_sig)
            cands["is_best_seen"] = (cands["candidate_sig"] == best_sig)
            cands["actual_gain_rank_in_candidates"] = cands["actual_gain_pct"].rank(
                method="min", ascending=False)
            cands["source_original_sig"] = base_sig
            rank_frames.append(cands)

    per_source = pd.DataFrame(summary_rows)
    labeled = pd.concat(rank_frames, ignore_index=True) if rank_frames else trace.iloc[:0].copy()
    if not len(per_source):
        raise ValueError("No matching searches. Check --objective flag and file contents.")
    return per_source, labeled


def summarize(per_source: pd.DataFrame):
    group_cols = ["objective", "policy", "circuit"]
    per_circuit = per_source.groupby(group_cols, dropna=False).agg(
        n_searches=("selection_gap_pp", "size"),
        selected_gain_pct=("selected_gain_pct", "mean"),
        best_seen_gain_pct=("best_seen_gain_pct", "mean"),
        selection_gap_pp=("selection_gap_pp", "mean"),
        median_gap_pp=("selection_gap_pp", "median"),
        fraction_gap_gt_1pp=("selection_gap_pp", lambda x: float((x > 1.0).mean())),
        fraction_selected_best=("selected_is_best_seen", "mean"),
        fraction_selected_harmful=("selected_is_harmful", "mean"),
        fraction_positive_found=("positive_candidate_found", "mean"),
        label_coverage=("label_coverage", "mean"),
        avg_n_candidates=("n_candidates_traced", "mean"),
    ).reset_index()
    # Important: equal weight per CIRCUIT (matching adaptive_full summary methodology).
    macro = per_circuit.groupby(["objective", "policy"], dropna=False).agg(
        n_circuits=("circuit", "nunique"),
        n_searches=("n_searches", "sum"),
        selected_gain_pct=("selected_gain_pct", "mean"),
        best_seen_gain_pct=("best_seen_gain_pct", "mean"),
        selection_gap_pp=("selection_gap_pp", "mean"),
        fraction_gap_gt_1pp=("fraction_gap_gt_1pp", "mean"),
        fraction_selected_best=("fraction_selected_best", "mean"),
        fraction_selected_harmful=("fraction_selected_harmful", "mean"),
        fraction_positive_found=("fraction_positive_found", "mean"),
        label_coverage=("label_coverage", "mean"),
        avg_n_candidates=("avg_n_candidates", "mean"),
    ).reset_index()
    return per_circuit, macro


def write_readme(output: Path, per_source: pd.DataFrame, macro: pd.DataFrame):
    lines = [
        "BEAM SEARCH DIAGNOSTICS — SEARCH COVERAGE VS SELECTION",
        "======================================================",
        "",
        "All gains are percent reductions relative to each ORIGINAL plan;",
        "selection_gap_pp is measured in PERCENTAGE POINTS.",
        "",
        "best_seen_gain_pct: hindsight oracle = max(0, actual heldout gain of all",
        "  candidates visited by that search). It is NOT an online algorithm result.",
        "selected_gain_pct: measured heldout gain of the method's actual pick.",
        "selection_gap_pp: best_seen_gain_pct - selected_gain_pct.",
        "",
        "If the gap is large: candidate selection/scoring left gains on the table",
        "  for the candidate pool already found. Limited-routing selection may help.",
        "If best_seen gain is small: this PARTICULAR search did not encounter",
        "  much improvement. This alone does not prove the operators cannot reach",
        "  better plans; inspect full-space oracle on comparable circuits/starts.",
        "If label coverage < 100%: best_seen is only a LOWER BOUND on the best",
        "  of all discovered candidates, and the measured gap can be understated.",
        "",
        "No routing, fitting, feature extraction, or candidate generation was run.",
        "All actual costs are saved heldout routing means from the prior run.",
        "The data was reused during algorithm development: not untouched final test.",
        "",
        "RESULTS — EQUAL CIRCUIT WEIGHTS",
        macro.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
        f"Searches analyzed: {len(per_source)}",
        f"Searches with any missing candidate labels: {int((per_source.n_candidates_without_heldout > 0).sum())}",
        "",
        "OUTPUT FILES",
        "per_source_oracle_gap.csv: one row per (objective,policy,circuit,source)",
        "per_circuit_summary.csv: one row per (objective,policy,circuit)",
        "macro_summary.csv: circuit-equal summary for each policy",
        "candidates_with_ranks.csv: all scored candidates, marked selected/oracle",
        "",
        "IMPORTANT: Never report best_seen as method-achieved gain in the paper.",
    ]
    (output / "README_results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(selected_path: Path, trace_path: Path, out_dir: Path, objective: str):
    chosen = read_csv(selected_path, SELECTED_REQUIRED, "per_source.csv")
    trace = read_csv(trace_path, TRACE_REQUIRED, "candidate_trace.csv")
    if objective != "all":
        chosen = chosen[chosen.objective == objective].copy()
        trace = trace[trace.objective == objective].copy()
    if chosen.empty:
        raise ValueError(f"No rows selected for objective={objective}")
    per_source, candidates = make_results(chosen, trace)
    per_circuit, macro = summarize(per_source)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_source.to_csv(out_dir / "per_source_oracle_gap.csv", index=False)
    per_circuit.to_csv(out_dir / "per_circuit_summary.csv", index=False)
    macro.to_csv(out_dir / "macro_summary.csv", index=False)
    candidates.to_csv(out_dir / "candidates_with_ranks.csv", index=False)
    write_readme(out_dir, per_source, macro)
    manifest = dict(input_selected=str(selected_path), input_trace=str(trace_path),
                    objective=objective, searches=int(len(per_source)),
                    note="Post-hoc heldout oracle for diagnosis only; no routing")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nRESULTS (equal weight per circuit):")
    print(macro.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nWrote:", out_dir.resolve())
    if (per_source.n_candidates_without_heldout > 0).any():
        print("WARNING: some candidates have no heldout labels; best-seen gain is a LOWER BOUND.")
    return per_source, macro


def self_test():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        chosen = pd.DataFrame([
            dict(objective="swap_count",policy="adaptive_beam",circuit="c1",source_plan_id="P0",
                 source_sig="origin1",pick_sig="a",actual_gain_pct=2.0,n_scored=3,
                 predicted_gain_pct=9.0,base_held=100.0,pick_held=98.0),
            dict(objective="swap_count",policy="random_beam",circuit="c1",source_plan_id="P0",
                 source_sig="origin1",pick_sig="origin1",actual_gain_pct=0.0,n_scored=2,
                 predicted_gain_pct=0.0,base_held=100.0,pick_held=100.0),
            dict(objective="swap_count",policy="adaptive_beam",circuit="c2",source_plan_id="P0",
                 source_sig="origin2",pick_sig="x",actual_gain_pct=-3.0,n_scored=1,
                 predicted_gain_pct=4.0,base_held=100.0,pick_held=103.0),
            dict(objective="routed_depth",policy="adaptive_beam",circuit="c1",source_plan_id="P0",
                 source_sig="origin1",pick_sig="a",actual_gain_pct=10.0,n_scored=1,
                 predicted_gain_pct=9.0,base_held=100.0,pick_held=90.0),
        ])
        tr = pd.DataFrame([
            dict(objective="swap_count",policy="adaptive_beam",circuit="c1",source_plan_id="P0",source_sig="origin1",candidate_sig="a",predicted_gain_pct=9.0,actual_gain_pct=2.0),
            dict(objective="swap_count",policy="adaptive_beam",circuit="c1",source_plan_id="P0",source_sig="origin1",candidate_sig="b",predicted_gain_pct=1.0,actual_gain_pct=8.0),
            dict(objective="swap_count",policy="adaptive_beam",circuit="c1",source_plan_id="P0",source_sig="origin1",candidate_sig="c",predicted_gain_pct=-1.0,actual_gain_pct=-4.0),
            dict(objective="swap_count",policy="random_beam",circuit="c1",source_plan_id="P0",source_sig="origin1",candidate_sig="a",predicted_gain_pct=-1.0,actual_gain_pct=2.0),
            dict(objective="swap_count",policy="random_beam",circuit="c1",source_plan_id="P0",source_sig="origin1",candidate_sig="b",predicted_gain_pct=-1.0,actual_gain_pct=8.0),
            dict(objective="swap_count",policy="adaptive_beam",circuit="c2",source_plan_id="P0",source_sig="origin2",candidate_sig="x",predicted_gain_pct=4.0,actual_gain_pct=-3.0),
            dict(objective="routed_depth",policy="adaptive_beam",circuit="c1",source_plan_id="P0",source_sig="origin1",candidate_sig="a",predicted_gain_pct=9.0,actual_gain_pct=10.0),
        ])
        chosen.to_csv(root / "per_source.csv", index=False)
        tr.to_csv(root / "candidate_trace.csv", index=False)
        per, macro = execute(root / "per_source.csv", root / "candidate_trace.csv", root / "outputs", "swap_count")
        a = per[(per.policy == "adaptive_beam") & (per.circuit == "c1")].iloc[0]
        b = per[(per.policy == "random_beam") & (per.circuit == "c1")].iloc[0]
        c = per[(per.policy == "adaptive_beam") & (per.circuit == "c2")].iloc[0]
        assert abs(a.best_seen_gain_pct - 8.0) < EPS and abs(a.selection_gap_pp - 6.0) < EPS
        assert abs(b.best_seen_gain_pct - 8.0) < EPS and abs(b.selection_gap_pp - 8.0) < EPS
        assert abs(c.best_seen_gain_pct) < EPS and abs(c.selection_gap_pp - 3.0) < EPS
        assert len(macro) == 2
        print("SELF-TEST PASSED")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selected", type=Path, default=Path("adaptive_traced/per_source.csv"))
    parser.add_argument("--trace", type=Path, default=Path("adaptive_traced/candidate_trace.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("beam_selection_gap"))
    parser.add_argument("--objective", choices=["swap_count", "routed_depth", "all"], default="swap_count")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
        else:
            execute(args.selected, args.trace, args.out_dir, args.objective)
    except (ValueError, FileNotFoundError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
