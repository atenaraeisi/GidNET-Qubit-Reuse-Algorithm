#!/usr/bin/env python3
"""Exact Depth-Cap Audit and Combinatorial Best-of-k Rerouting Analysis (No Bootstrap).

1. Depth-Cap Check:
   - Compares frozen_winners.csv rows between depth_cap=False and depth_cap=True.
   - If candidate_training_means.csv (and optionally source_memberships.csv) is available,
     computes exact logical_depth ratios (candidate / original) to verify whether the 5%
     cap was inactive on winners vs. whether it filtered out non-winning candidates.

2. Exact Best-of-k Rerouting (No Bootstrap / No Monte Carlo noise):
   - Uses exact combinatorial order statistics over all N=30 held-out seeds:
     * Without replacement: E[min of k draws from N] = sum_{i=1..N-k+1} x_(i) * C(N-i, k-1) / C(N, k)
     * With replacement:    E[min of k iid draws]    = sum_{i=1..N} x_(i) * (((N-i+1)/N)^k - ((N-i)/N)^k)
   - Evaluates:
     * Same-plan Rerouting on Original (Best-of-k Original vs 1-seed Original mean)
     * 1-seed Repair (A_suffix / B_suffix_chain) vs Best-of-k Original
     * Equal-k Repair (Best-of-k Selected) vs Best-of-k Original
   - Outputs both micro-averages (50 plans) and macro-averages (equal-weighted across 5 circuits).
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd


def exact_min_without_replacement(values: np.ndarray, k: int) -> float:
    """Exact expected minimum when drawing k distinct seeds out of N without replacement."""
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    if k < 1 or k > n:
        raise ValueError(f"k={k} out of bounds for n={n}")
    denom = math.comb(n, k)
    weights = np.array([math.comb(n - 1 - i, k - 1) / denom for i in range(n - k + 1)], dtype=float)
    return float(np.dot(x[: n - k + 1], weights))


def exact_min_with_replacement(values: np.ndarray, k: int) -> float:
    """Exact expected minimum when drawing k seeds with replacement (analytical limit of Bootstrap)."""
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    if k < 1:
        raise ValueError(f"k={k} must be >= 1")
    idx = np.arange(n, dtype=float)
    p_ge_curr = ((n - idx) / n) ** k
    p_ge_next = ((n - idx - 1) / n) ** k
    weights = p_ge_curr - p_ge_next
    return float(np.dot(x, weights))


def check_depth_cap(
    frozen_path: Path,
    train_means_path: Path | None = None,
    memberships_path: Path | None = None,
    cap_pct: float = 5.0,
) -> dict:
    frozen = pd.read_csv(frozen_path, dtype={"plan_id": str, "original_key": str, "selected_key": str, "best_neighbor_key": str})
    f_no = frozen[frozen["depth_cap"] == False].sort_values(["circuit", "plan_id", "objective", "pool"]).reset_index(drop=True)
    f_yes = frozen[frozen["depth_cap"] == True].sort_values(["circuit", "plan_id", "objective", "pool"]).reset_index(drop=True)

    sel_equal = bool((f_no["selected_key"].fillna("") == f_yes["selected_key"].fillna("")).all())
    nbr_equal = bool((f_no["best_neighbor_key"].fillna("") == f_yes["best_neighbor_key"].fillna("")).all())
    mean_equal = bool(np.allclose(f_no["mean_selected"], f_yes["mean_selected"], equal_nan=True))

    report = {
        "pairs_compared": len(f_no),
        "selected_key_identical_all_pairs": sel_equal,
        "best_neighbor_key_identical_all_pairs": nbr_equal,
        "mean_selected_identical_all_pairs": mean_equal,
    }

    if train_means_path and train_means_path.is_file():
        tm = pd.read_csv(train_means_path, dtype={"candidate_key": str})
        depth_map = {(str(r.circuit), str(r.candidate_key)): float(r.logical_depth) for r in tm.itertuples(index=False)}
        f_no["selected_logical_depth"] = [depth_map[(r.circuit, r.selected_key)] for r in f_no.itertuples(index=False)]
        f_no["neighbor_logical_depth"] = [
            depth_map[(r.circuit, r.best_neighbor_key)] if pd.notna(r.best_neighbor_key) and str(r.best_neighbor_key) != "" else np.nan
            for r in f_no.itertuples(index=False)
        ]
        f_no["selected_depth_ratio"] = f_no["selected_logical_depth"] / f_no["depth_original"]
        f_no["neighbor_depth_ratio"] = f_no["neighbor_logical_depth"] / f_no["depth_original"]
        limit_ratio = 1.0 + cap_pct / 100.0
        report["max_selected_depth_ratio"] = float(f_no["selected_depth_ratio"].max())
        report["max_best_neighbor_depth_ratio"] = float(f_no["neighbor_depth_ratio"].dropna().max())
        report["winners_exceeding_cap"] = int((f_no["selected_depth_ratio"] > limit_ratio + 1e-9).sum())
        report["best_neighbors_exceeding_cap"] = int((f_no["neighbor_depth_ratio"].dropna() > limit_ratio + 1e-9).sum())

        if memberships_path and memberships_path.is_file():
            mem = pd.read_csv(memberships_path, dtype={"plan_id": str, "candidate_key": str})
            orig_depths = (
                f_no[["circuit", "plan_id", "depth_original"]]
                .drop_duplicates()
                .set_index(["circuit", "plan_id"])["depth_original"]
                .to_dict()
            )
            nbrs = mem[~mem["is_original"]].copy()
            nbrs["cand_depth"] = [depth_map[(r.circuit, r.candidate_key)] for r in nbrs.itertuples(index=False)]
            nbrs["orig_depth"] = [orig_depths[(r.circuit, r.plan_id)] for r in nbrs.itertuples(index=False)]
            nbrs["ratio"] = nbrs["cand_depth"] / nbrs["orig_depth"]
            report["total_candidate_memberships"] = len(nbrs)
            report["candidates_exceeding_cap_count"] = int((nbrs["ratio"] > limit_ratio + 1e-9).sum())
            report["max_candidate_depth_ratio_overall"] = float(nbrs["ratio"].max())
            report["min_candidate_depth_ratio_overall"] = float(nbrs["ratio"].min())
    else:
        report["note"] = (
            "candidate_training_means.csv not found in path; verified exact equality in frozen_winners.csv. "
            "Pass --train-means experiment4a_results/candidate_training_means.csv to inspect raw candidate logical depths."
        )
    return report


def compute_best_of_k_tables(
    paired_path: Path,
    ks: tuple[int, ...] = (1, 2, 3, 5, 10, 15, 30),
    pools: tuple[str, ...] = ("A_suffix", "B_suffix_chain"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paired = pd.read_csv(paired_path, dtype={"plan_id": str, "original_key": str, "selected_key": str})
    sub = paired[(paired["depth_cap"] == False) & (paired["pool"].isin(pools))].copy()

    per_source_rows = []
    for (circuit, plan_id, objective, pool), grp in sub.groupby(["circuit", "plan_id", "objective", "pool"]):
        orig_vals = grp["original"].to_numpy(dtype=float)
        sel_vals = grp["selected"].to_numpy(dtype=float)
        n_seeds = len(orig_vals)
        orig_mean = float(orig_vals.mean())
        sel_mean = float(sel_vals.mean())
        orig_std = float(orig_vals.std(ddof=1))
        sel_std = float(sel_vals.std(ddof=1))

        for k in ks:
            if k > n_seeds:
                continue
            # Exact without replacement (drawing k distinct seeds from 30)
            orig_bok_wor = exact_min_without_replacement(orig_vals, k)
            sel_bok_wor = exact_min_without_replacement(sel_vals, k)
            # Exact with replacement (analytical expectation of Bootstrap)
            orig_bok_wr = exact_min_with_replacement(orig_vals, k)
            sel_bok_wr = exact_min_with_replacement(sel_vals, k)

            per_source_rows.append({
                "circuit": circuit,
                "plan_id": plan_id,
                "objective": objective,
                "pool": pool,
                "k": k,
                "orig_mean_1seed": orig_mean,
                "sel_mean_1seed": sel_mean,
                "orig_std": orig_std,
                "sel_std": sel_std,
                # Without replacement metrics
                "orig_best_of_k_wor": orig_bok_wor,
                "sel_best_of_k_wor": sel_bok_wor,
                "reroute_gain_orig_wor": orig_mean - orig_bok_wor,
                "repair_1seed_gain_vs_orig_1seed": orig_mean - sel_mean,
                "repair_1seed_vs_orig_bok_wor": orig_bok_wor - sel_mean,
                "repair_bok_gain_vs_orig_1seed_wor": orig_mean - sel_bok_wor,
                "repair_bok_vs_orig_bok_wor": orig_bok_wor - sel_bok_wor,
                # With replacement metrics (exact analytical bootstrap)
                "orig_best_of_k_wr": orig_bok_wr,
                "sel_best_of_k_wr": sel_bok_wr,
                "reroute_gain_orig_wr": orig_mean - orig_bok_wr,
                "repair_bok_vs_orig_bok_wr": orig_bok_wr - sel_bok_wr,
            })

    per_source = pd.DataFrame(per_source_rows)

    # Circuit-level summary
    by_circuit = (
        per_source.groupby(["circuit", "objective", "pool", "k"], as_index=False)
        .agg(
            n_plans=("plan_id", "size"),
            orig_mean_1seed=("orig_mean_1seed", "mean"),
            reroute_gain_orig_wor=("reroute_gain_orig_wor", "mean"),
            reroute_gain_orig_wr=("reroute_gain_orig_wr", "mean"),
            repair_1seed_gain=("repair_1seed_gain_vs_orig_1seed", "mean"),
            repair_bok_gain_vs_orig_1seed_wor=("repair_bok_gain_vs_orig_1seed_wor", "mean"),
            repair_bok_vs_orig_bok_wor=("repair_bok_vs_orig_bok_wor", "mean"),
            plans_repair_1seed_beats_orig_bok=("repair_1seed_vs_orig_bok_wor", lambda s: int((s > 1e-9).sum())),
            plans_repair_bok_beats_orig_bok=("repair_bok_vs_orig_bok_wor", lambda s: int((s > 1e-9).sum())),
        )
    )

    # Global summary (both micro across 50 plans and macro across 5 circuits)
    micro = (
        per_source.groupby(["objective", "pool", "k"], as_index=False)
        .agg(
            sources=("plan_id", "size"),
            orig_mean_1seed=("orig_mean_1seed", "mean"),
            orig_std_mean=("orig_std", "mean"),
            reroute_gain_orig_wor=("reroute_gain_orig_wor", "mean"),
            reroute_gain_orig_wr=("reroute_gain_orig_wr", "mean"),
            repair_1seed_gain=("repair_1seed_gain_vs_orig_1seed", "mean"),
            repair_1seed_vs_orig_bok_wor=("repair_1seed_vs_orig_bok_wor", "mean"),
            repair_bok_gain_vs_orig_1seed_wor=("repair_bok_gain_vs_orig_1seed_wor", "mean"),
            repair_bok_vs_orig_bok_wor=("repair_bok_vs_orig_bok_wor", "mean"),
            sources_1seed_beats_orig_bok=("repair_1seed_vs_orig_bok_wor", lambda s: int((s > 1e-9).sum())),
            sources_bok_beats_orig_bok=("repair_bok_vs_orig_bok_wor", lambda s: int((s > 1e-9).sum())),
        )
    )
    macro = (
        by_circuit.groupby(["objective", "pool", "k"], as_index=False)
        .agg(
            macro_reroute_gain_orig_wor=("reroute_gain_orig_wor", "mean"),
            macro_repair_1seed_gain=("repair_1seed_gain", "mean"),
            macro_repair_bok_vs_orig_bok_wor=("repair_bok_vs_orig_bok_wor", "mean"),
        )
    )
    summary = pd.merge(micro, macro, on=["objective", "pool", "k"])
    return per_source, by_circuit, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen", type=Path, default=Path("frozen_winners.csv"))
    p.add_argument("--paired", type=Path, default=Path("heldout_paired_seeds.csv"))
    p.add_argument("--train-means", type=Path, default=Path("candidate_training_means.csv"))
    p.add_argument("--memberships", type=Path, default=Path("source_memberships.csv"))
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 5, 10, 15, 30])
    p.add_argument("--out-dir", type=Path, default=Path("."))
    args = p.parse_args()

    cap_report = check_depth_cap(args.frozen, args.train_means, args.memberships)
    print("=== 1. DEPTH CAP CHECK ===")
    for k, v in cap_report.items():
        print(f"  {k}: {v}")

    per_source, by_circuit, summary = compute_best_of_k_tables(args.paired, tuple(args.ks))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_source.to_csv(args.out_dir / "exact_best_of_k_per_source.csv", index=False)
    by_circuit.to_csv(args.out_dir / "exact_best_of_k_by_circuit.csv", index=False)
    summary.to_csv(args.out_dir / "exact_best_of_k_summary.csv", index=False)

    print("\n=== 2. EXACT BEST-OF-K SUMMARY (B_suffix_chain) ===")
    b_sum = summary[summary["pool"] == "B_suffix_chain"]
    cols = [
        "objective", "k", "reroute_gain_orig_wor", "reroute_gain_orig_wr",
        "repair_1seed_gain", "repair_bok_gain_vs_orig_1seed_wor",
        "repair_bok_vs_orig_bok_wor", "sources_1seed_beats_orig_bok", "sources_bok_beats_orig_bok"
    ]
    print(b_sum[cols].to_string(index=False))


if __name__ == "__main__":
    main()
