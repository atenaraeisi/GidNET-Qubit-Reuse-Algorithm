#!/usr/bin/env python3
"""
Comprehensive Visualization Suite for Experiment 4A (Physical Value of Expanded Neighborhood).

Usage:
  1. Place this script inside `experiment4a_results_exp3parity` (or next to the CSV files) and run:
       python plot_experiment4a_results.py
  2. Or pass a custom directory path:
       python plot_experiment4a_results.py --dir path/to/experiment4a_results_exp3parity
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_or_reconstruct_data(data_dir: Path):
    """Load Experiment 4A CSVs; reconstruct held-out paired/comparison tables if needed."""
    cov_path = data_dir / "coverage_per_source.csv"
    fw_path = data_dir / "frozen_winners.csv"
    soc_path = data_dir / "source_opportunity_comparison.csv"
    opp_path = data_dir / "opportunity_summary.csv"
    hs_path = data_dir / "heldout_summary.csv"
    hbc_path = data_dir / "heldout_by_circuit.csv"
    rh_path = data_dir / "routes_holdout.csv"
    hoc_path = data_dir / "heldout_opportunity_comparison.csv"
    hps_path = data_dir / "heldout_paired_seeds.csv"

    for req in [cov_path, fw_path, soc_path, opp_path, hs_path, hbc_path, rh_path]:
        if not req.is_file():
            raise FileNotFoundError(f"Missing required file: {req.resolve()}")

    cov = pd.read_csv(cov_path, dtype={"plan_id": str})
    fw = pd.read_csv(fw_path, dtype={"plan_id": str, "original_key": str, "selected_key": str})
    soc = pd.read_csv(soc_path, dtype={"plan_id": str, "original_key": str, "selected_key": str})
    opp = pd.read_csv(opp_path)
    hs = pd.read_csv(hs_path)
    hbc = pd.read_csv(hbc_path)
    rh = pd.read_csv(rh_path, dtype={"candidate_key": str})

    # Reconstruct or load heldout_paired_seeds
    if hps_path.is_file():
        hps = pd.read_csv(hps_path, dtype={"plan_id": str, "original_key": str, "selected_key": str})
    else:
        look = {(r.circuit, str(r.candidate_key), int(r.seed)): r for r in rh.itertuples(index=False)}
        holdout_seeds = sorted(rh["seed"].unique())
        pairs = []
        for r in fw.itertuples(index=False):
            for seed in holdout_seeds:
                ori = look[(r.circuit, str(r.original_key), int(seed))]
                win = look[(r.circuit, str(r.selected_key), int(seed))]
                base = float(getattr(ori, r.objective))
                got = float(getattr(win, r.objective))
                pairs.append({
                    "circuit": r.circuit, "plan_id": r.plan_id, "objective": r.objective,
                    "pool": r.pool, "depth_cap": r.depth_cap, "seed": int(seed),
                    "original": base, "selected": got, "paired_improvement": base - got,
                    "original_key": r.original_key, "selected_key": r.selected_key
                })
        hps = pd.DataFrame(pairs)

    # Reconstruct or load heldout_opportunity_comparison
    if hoc_path.is_file():
        hoc = pd.read_csv(hoc_path, dtype={"plan_id": str})
    else:
        hoc = hps.groupby(["circuit", "plan_id", "objective", "pool", "depth_cap"], as_index=False).agg(
            heldout_original_mean=("original", "mean"),
            heldout_selected_mean=("selected", "mean"),
            heldout_gain_vs_original=("paired_improvement", "mean"),
            holdout_seeds=("seed", "nunique")
        )
        ref = hoc.loc[hoc.pool == "A_suffix", ["circuit", "plan_id", "objective", "depth_cap", "heldout_selected_mean"]].rename(
            columns={"heldout_selected_mean": "heldout_suffix_mean"}
        )
        hoc = hoc.merge(ref, on=["circuit", "plan_id", "objective", "depth_cap"], validate="many_to_one")
        hoc["heldout_incremental_gain_over_suffix"] = hoc.heldout_suffix_mean - hoc.heldout_selected_mean

    return {
        "cov": cov, "fw": fw, "soc": soc, "opp": opp,
        "hs": hs, "hbc": hbc, "rh": rh, "hps": hps, "hoc": hoc
    }


def plot_fig1_coverage(cov: pd.DataFrame, out_dir: Path):
    """Figure 1: Search-Space Coverage & Dead-End Elimination across Circuits and Plans."""
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), dpi=300)

    cov = cov.copy()
    cov["short_circuit"] = cov["circuit"].str.replace(".qasm", "", regex=False)
    cov["added_by_B"] = cov["B_suffix_chain"] - cov["A_suffix"]
    cov["added_by_C"] = cov["C_all"] - cov["B_suffix_chain"]

    # Panel A: Mean legal neighbors per plan by circuit (Stacked Bar: Pool A + Added by B + Added by C)
    grp = cov.groupby("short_circuit", as_index=False).agg(
        n_plans=("plan_id", "size"),
        mean_A=("A_suffix", "mean"),
        mean_add_B=("added_by_B", "mean"),
        mean_add_C=("added_by_C", "mean"),
        sum_A=("A_suffix", "sum"),
        sum_B=("B_suffix_chain", "sum"),
        sum_C=("C_all", "sum"),
        zero_A=("A_suffix", lambda s: int((s == 0).sum()))
    )

    x = np.arange(len(grp))
    w = 0.55
    axes[0].bar(x, grp["mean_A"], w, label="Pool A: suffix_swap only (Total = 130)", color="#3b82f6", edgecolor="black")
    axes[0].bar(x, grp["mean_add_B"], w, bottom=grp["mean_A"],
                label="Added in Pool B: + chain_transfer (+159 -> Total = 289)", color="#10b981", edgecolor="black")
    axes[0].bar(x, grp["mean_add_C"], w, bottom=grp["mean_A"] + grp["mean_add_B"],
                label="Added in Pool C: + node_swap / seg_rev (+2 -> Total = 291)", color="#f59e0b", edgecolor="black")

    for i, r in grp.iterrows():
        total_m = r["mean_A"] + r["mean_add_B"] + r["mean_add_C"]
        note = f"Mean: {total_m:.1f}\n(A:{int(r['sum_A'])} -> B:{int(r['sum_B'])})"
        if r["zero_A"] > 0:
            note += f"\n[{int(r['zero_A'])}/{int(r['n_plans'])} dead-ends in A!]"
        axes[0].annotate(note, xy=(x[i], total_m), xytext=(0, 4),
                         textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{r.short_circuit}\n(n={r.n_plans})" for r in grp.itertuples()], fontsize=9.5)
    axes[0].set_ylabel("Mean Legal Neighbors per Source Plan", fontsize=10, fontweight="bold")
    axes[0].set_title("A. Search-Space Expansion by Circuit (Fixed Width)", fontsize=11, fontweight="bold")
    axes[0].set_ylim(0, 14.5)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper left", fontsize=8.5)

    # Panel B: Histogram / Frequency of neighborhood sizes per plan (Pool A vs Pool B)
    bins = np.arange(-0.5, 14.5, 1.0)
    axes[1].hist(cov["A_suffix"], bins=bins, alpha=0.65, label="Pool A (suffix_swap): Median=2, Min=0 (4 dead-ends)",
                 color="#3b82f6", edgecolor="black", rwidth=0.85)
    axes[1].hist(cov["B_suffix_chain"], bins=bins, alpha=0.65, label="Pool B (+ chain_transfer): Median=4, Min=3 (0 dead-ends)",
                 color="#10b981", edgecolor="black", rwidth=0.85)
    axes[1].axvline(0, color="#dc2626", linestyle=":", linewidth=1.5, label="Dead-end threshold (0 legal neighbors)")

    axes[1].set_xticks(range(0, 14))
    axes[1].set_xlabel("Number of Distinct Legal Fixed-Width Neighbors per Plan", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Number of Source Plans (out of 50)", fontsize=10, fontweight="bold")
    axes[1].set_title("B. Distribution of Neighborhood Size per Source Plan", fontsize=11, fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right", fontsize=8.5)

    fig.tight_layout(pad=1.4)
    out_path = out_dir / "fig1_neighborhood_coverage.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_fig2_overall_and_shrinkage(opp: pd.DataFrame, hs: pd.DataFrame, hoc: pd.DataFrame, out_dir: Path):
    """Figure 2: Overall Held-Out Gain vs Original & Explore-to-Validate Generalization Gap."""
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), dpi=300)

    hs_sub = hs[hs["depth_cap"] == False].set_index(["objective", "pool"])
    opp_sub = opp[opp["depth_cap"] == False].set_index(["objective", "pool"])
    hoc_sub = hoc[hoc["depth_cap"] == False]

    objectives = ["routed_depth", "swap_count"]
    obj_labels = ["Routed Depth Reduction\n(vs. Original Plan)", "SWAP Count Reduction\n(vs. Original Plan)"]
    pools = ["A_suffix", "B_suffix_chain", "C_all"]
    pool_labels = ["Pool A (suffix_swap)", "Pool B (+ chain_transfer)", "Pool C (All 4 ops)"]
    colors = ["#64748b", "#10b981", "#f59e0b"]

    x = np.arange(len(objectives))
    w = 0.25

    for idx, (pool, plabel, col) in enumerate(zip(pools, pool_labels, colors)):
        vals = [hs_sub.loc[(obj, pool), "mean_gain_from_original"] for obj in objectives]
        wins = [int(hs_sub.loc[(obj, pool), "sources_better_than_original"]) for obj in objectives]
        rects = axes[0].bar(x + (idx - 1) * w, vals, w, label=plabel, color=col, edgecolor="black", alpha=0.9)
        for rect, win_cnt in zip(rects, wins):
            h = rect.get_height()
            axes[0].annotate(f"+{h:.2f}\n({win_cnt}/50 W)",
                             xy=(rect.get_x() + rect.get_width() / 2, h),
                             xytext=(0, 4), textcoords="offset points",
                             ha="center", va="bottom", fontsize=8, fontweight="bold")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(obj_labels, fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Held-Out Mean Reduction vs. Original (30 Seeds)", fontsize=10, fontweight="bold")
    axes[0].set_title("A. Total Held-Out Improvement over Original Plan (Pool B == Pool C)", fontsize=11, fontweight="bold")
    axes[0].set_ylim(0, 6.8)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right", fontsize=8.5)

    exp_vals = [opp_sub.loc[(obj, "B_suffix_chain"), "mean_gain_over_suffix"] for obj in objectives]
    val_vals = [hs_sub.loc[(obj, "B_suffix_chain"), "mean_extra_gain_from_suffix"] for obj in objectives]

    wtl_labels_exp = []
    wtl_labels_val = []
    for obj in objectives:
        w_exp = int(opp_sub.loc[(obj, "B_suffix_chain"), "wins_over_suffix"])
        wtl_labels_exp.append(f"{w_exp}W / {50 - w_exp}T / 0L")
        sub_o = hoc_sub[(hoc_sub["objective"] == obj) & (hoc_sub["pool"] == "B_suffix_chain")]["heldout_incremental_gain_over_suffix"]
        w_v = int((sub_o > 1e-9).sum())
        t_v = int((sub_o.abs() <= 1e-9).sum())
        l_v = int((sub_o < -1e-9).sum())
        wtl_labels_val.append(f"{w_v}W / {t_v}T / {l_v}L")

    w2 = 0.34
    r_exp = axes[1].bar(x - w2 / 2, exp_vals, w2, label="Explore Phase (5 Train Seeds: 42-46)", color="#3b82f6", edgecolor="black")
    r_val = axes[1].bar(x + w2 / 2, val_vals, w2, label="Validate Phase (30 Holdout Seeds: 100-129)", color="#10b981", edgecolor="black")

    for rect, wtl in zip(r_exp, wtl_labels_exp):
        h = rect.get_height()
        axes[1].annotate(f"+{h:.3f}\n({wtl})", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    for rect, wtl, e_val in zip(r_val, wtl_labels_val, exp_vals):
        h = rect.get_height()
        shrink = 100.0 * (h - e_val) / e_val
        axes[1].annotate(f"+{h:.3f} ({shrink:.1f}%)\n({wtl})", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Routed Depth Extra Gain\n(Pool B/C over Pool A)", "SWAP Count Extra Gain\n(Pool B/C over Pool A)"], fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Mean Extra Reduction over Pool A (suffix_swap)", fontsize=10, fontweight="bold")
    axes[1].set_title("B. Generalization Shrinkage: Explore (5 Seeds) vs. Validate (30 Seeds)", fontsize=11, fontweight="bold")
    axes[1].set_ylim(0, 2.25)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right", fontsize=8.5)

    fig.tight_layout(pad=1.4)
    out_path = out_dir / "fig2_overall_opportunity_and_shrinkage.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_fig3_circuit_breakdown(hoc: pd.DataFrame, out_dir: Path):
    """Figure 3: Circuit-Level Breakdown of Held-Out Gains (vs Original and Pool B over Pool A)."""
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.0), dpi=300)

    hoc_sub = hoc[hoc["depth_cap"] == False].copy()
    hoc_sub["short_circuit"] = hoc_sub["circuit"].str.replace(".qasm", "", regex=False)
    circuits = sorted(hoc_sub["short_circuit"].unique())
    x = np.arange(len(circuits))
    w = 0.35

    for row_idx, (obj, obj_title, unit) in enumerate([
        ("routed_depth", "Routed Depth", "depth layers"),
        ("swap_count", "SWAP Count", "SWAP gates")
    ]):
        ax_left = axes[row_idx, 0]
        sub_A = hoc_sub[(hoc_sub["objective"] == obj) & (hoc_sub["pool"] == "A_suffix")].groupby("short_circuit")["heldout_gain_vs_original"].agg(["mean", "count"]).loc[circuits]
        sub_B = hoc_sub[(hoc_sub["objective"] == obj) & (hoc_sub["pool"] == "B_suffix_chain")].groupby("short_circuit")["heldout_gain_vs_original"].agg(["mean", "count"]).loc[circuits]

        rA = ax_left.bar(x - w / 2, sub_A["mean"], w, label="Pool A (suffix_swap)", color="#64748b", edgecolor="black")
        rB = ax_left.bar(x + w / 2, sub_B["mean"], w, label="Pool B (+ chain_transfer)", color="#10b981", edgecolor="black")

        for rect in list(rA) + list(rB):
            h = rect.get_height()
            ax_left.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, max(h, 0)),
                             xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

        ax_left.axhline(0, color="black", linewidth=0.9)
        ax_left.set_xticks(x)
        ax_left.set_xticklabels([f"{c}\n(n={int(sub_A.loc[c, 'count'])})" for c in circuits], fontsize=9)
        ax_left.set_ylabel(f"Mean Reduction vs. Original ({unit})", fontsize=9.5, fontweight="bold")
        ax_left.set_title(f"{'A' if row_idx==0 else 'C'}. {obj_title}: Held-Out Gain vs. Original Plan", fontsize=10.5, fontweight="bold")
        ax_left.set_ylim(0, max(sub_B["mean"].max() * 1.25, 1.0))
        ax_left.grid(axis="y", linestyle="--", alpha=0.5)
        ax_left.legend(loc="upper left", fontsize=8.5)

        ax_right = axes[row_idx, 1]
        sub_inc = hoc_sub[(hoc_sub["objective"] == obj) & (hoc_sub["pool"] == "B_suffix_chain")]
        inc_stats = sub_inc.groupby("short_circuit")["heldout_incremental_gain_over_suffix"].agg(
            mean="mean",
            sem=lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0,
            wins=lambda s: int((s > 1e-9).sum()),
            ties=lambda s: int((s.abs() <= 1e-9).sum()),
            losses=lambda s: int((s < -1e-9).sum()),
            n="count"
        ).loc[circuits]

        bar_colors = ["#10b981" if m >= 0 else "#ef4444" for m in inc_stats["mean"]]
        ax_right.bar(x, inc_stats["mean"], 0.5, yerr=inc_stats["sem"], capsize=4,
                     color=bar_colors, edgecolor="black", alpha=0.88,
                     error_kw=dict(lw=1.1, capthick=1.1, ecolor="#1e293b"))

        for i, (c, row_s) in enumerate(inc_stats.iterrows()):
            m = row_s["mean"]
            err = row_s["sem"]
            pos_y = (m + err) if m >= 0 else (m - err)
            va = "bottom" if m >= 0 else "top"
            offset = 4 if m >= 0 else -4
            wtl_str = f"{int(row_s['wins'])}W/{int(row_s['ties'])}T/{int(row_s['losses'])}L"
            ax_right.annotate(f"{m:+.2f}\n({wtl_str})", xy=(x[i], pos_y),
                              xytext=(0, offset), textcoords="offset points",
                              ha="center", va=va, fontsize=8, fontweight="bold")

        ax_right.axhline(0, color="black", linewidth=1.0)
        ax_right.set_xticks(x)
        ax_right.set_xticklabels([f"{c}\n(n={int(inc_stats.loc[c, 'n'])})" for c in circuits], fontsize=9)
        ax_right.set_ylabel(f"Extra Gain of Pool B over A ({unit})", fontsize=9.5, fontweight="bold")
        ax_right.set_title(f"{'B' if row_idx==0 else 'D'}. {obj_title}: Extra Gain of Pool B over Pool A (±1 SEM)", fontsize=10.5, fontweight="bold")
        ymin = min(0, (inc_stats["mean"] - inc_stats["sem"]).min()) - (1.3 if row_idx == 0 else 0.7)
        ymax = (inc_stats["mean"] + inc_stats["sem"]).max() + (1.5 if row_idx == 0 else 0.8)
        ax_right.set_ylim(ymin, ymax)
        ax_right.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout(pad=1.6, h_pad=2.4)
    out_path = out_dir / "fig3_circuit_breakdown.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_fig4_plan_level_diagnostics(soc: pd.DataFrame, hoc: pd.DataFrame, hps: pd.DataFrame, out_dir: Path):
    """Figure 4: Plan-Level Diagnostics — Explore vs Validate Scatter & Waterfall for Plans where Pool B != Pool A."""
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 10.6), dpi=300)

    m = soc[(soc["depth_cap"] == False) & (soc["pool"] == "B_suffix_chain")].merge(
        hoc[(hoc["depth_cap"] == False) & (hoc["pool"] == "B_suffix_chain")],
        on=["circuit", "plan_id", "objective", "pool", "depth_cap"]
    )
    soc_A = soc[(soc["depth_cap"] == False) & (soc["pool"] == "A_suffix")][["circuit", "plan_id", "objective", "selected_key"]].rename(
        columns={"selected_key": "selected_key_A"}
    )
    m = m.merge(soc_A, on=["circuit", "plan_id", "objective"])

    hps_sub = hps[hps["depth_cap"] == False]
    hps_A = hps_sub[hps_sub["pool"] == "A_suffix"][["circuit", "plan_id", "objective", "seed", "selected"]].rename(
        columns={"selected": "selected_A"}
    )
    hps_B = hps_sub[hps_sub["pool"] == "B_suffix_chain"][["circuit", "plan_id", "objective", "seed", "selected"]].rename(
        columns={"selected": "selected_B"}
    )
    hps_ba = hps_A.merge(hps_B, on=["circuit", "plan_id", "objective", "seed"])
    hps_ba["b_over_a_seed"] = hps_ba["selected_A"] - hps_ba["selected_B"]
    sem_df = hps_ba.groupby(["circuit", "plan_id", "objective"], as_index=False).agg(
        holdout_sem=("b_over_a_seed", lambda s: s.std(ddof=1) / np.sqrt(len(s)))
    )
    m = m.merge(sem_df, on=["circuit", "plan_id", "objective"])
    m["short_circuit"] = m["circuit"].str.replace(".qasm", "", regex=False)
    m["label"] = m["short_circuit"] + ":" + m["plan_id"]

    circuit_palette = {
        "alu-bdd_288": "#8b5cf6",
        "mini_alu_305": "#3b82f6",
        "rd73_140": "#10b981",
        "sym9_146": "#f59e0b",
        "system9": "#ef4444"
    }

    for row_idx, (obj, obj_title) in enumerate([
        ("routed_depth", "Routed Depth"),
        ("swap_count", "SWAP Count")
    ]):
        diff = m[(m["objective"] == obj) & (m["selected_key"] != m["selected_key_A"])].copy()

        ax_scat = axes[row_idx, 0]
        for c_name, grp in diff.groupby("short_circuit"):
            ax_scat.errorbar(
                grp["incremental_gain_over_suffix"],
                grp["heldout_incremental_gain_over_suffix"],
                yerr=grp["holdout_sem"],
                fmt="o", markersize=7, capsize=3, alpha=0.9,
                color=circuit_palette.get(c_name, "black"),
                markeredgecolor="black", label=f"{c_name} (n={len(grp)})"
            )
            for idx_pt, r in enumerate(grp.itertuples()):
                # Slight vertical offset for overlapping points (e.g. P5 and P8 at x=4.0)
                dy = -8 if (r.short_circuit == "mini_alu_305" and r.plan_id == "P8") else 4
                ax_scat.annotate(r.plan_id, (r.incremental_gain_over_suffix, r.heldout_incremental_gain_over_suffix),
                                 xytext=(5, dy), textcoords="offset points", fontsize=7.5, fontweight="bold")

        max_v = max(diff["incremental_gain_over_suffix"].max(), diff["heldout_incremental_gain_over_suffix"].max()) + 1.5
        min_v = min(0, diff["heldout_incremental_gain_over_suffix"].min()) - 1.8
        ax_scat.plot([0, max_v], [0, max_v], linestyle="--", color="gray", linewidth=1.0, label="y = x (No shrinkage)")
        ax_scat.axhline(0, color="black", linewidth=1.0)
        ax_scat.axvline(0, color="black", linewidth=1.0)
        ax_scat.fill_between([0, max_v], 0, min_v, color="#fee2e2", alpha=0.35, label="Overfit zone (Train > 0, Holdout < 0)")

        ax_scat.set_xlim(-0.5, max_v)
        ax_scat.set_ylim(min_v, max_v * 0.85)
        ax_scat.set_xlabel("Explore Extra Gain of Pool B over A (5 Train Seeds)", fontsize=9.5, fontweight="bold")
        ax_scat.set_ylabel("Held-Out Extra Gain of Pool B over A (30 Test Seeds)", fontsize=9.5, fontweight="bold")
        ax_scat.set_title(f"{'A' if row_idx==0 else 'C'}. {obj_title}: Explore vs. Held-Out Gain (n={len(diff)} diverged plans)", fontsize=10.5, fontweight="bold")
        ax_scat.grid(True, linestyle="--", alpha=0.5)
        ax_scat.legend(loc="lower right", fontsize=8, framealpha=0.92)

        ax_bar = axes[row_idx, 1]
        diff_sorted = diff.sort_values("heldout_incremental_gain_over_suffix", ascending=True).reset_index(drop=True)
        y_pos = np.arange(len(diff_sorted))
        b_cols = [circuit_palette.get(c, "#3b82f6") for c in diff_sorted["short_circuit"]]

        ax_bar.barh(y_pos, diff_sorted["heldout_incremental_gain_over_suffix"], xerr=diff_sorted["holdout_sem"],
                    height=0.72, color=b_cols, edgecolor="black", alpha=0.9, capsize=2.5,
                    error_kw=dict(lw=0.9, capthick=0.9, ecolor="#1e293b"))
        ax_bar.axvline(0, color="black", linewidth=1.0)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(diff_sorted["label"], fontsize=8)
        ax_bar.set_xlabel("Held-Out Extra Gain of Pool B over Pool A (±1 SEM across 30 Seeds)", fontsize=9.5, fontweight="bold")
        ax_bar.set_title(f"{'B' if row_idx==0 else 'D'}. {obj_title}: Per-Plan Held-Out Extra Gain (Pool B vs. Pool A)", fontsize=10.5, fontweight="bold")
        ax_bar.grid(axis="x", linestyle="--", alpha=0.5)

    fig.tight_layout(pad=1.6, h_pad=2.4)
    out_path = out_dir / "fig4_plan_level_diagnostics.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("."),
                        help="Directory containing Experiment 4A CSV files (default: current directory)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory for generated PNG figures (default: <dir>/figures_exp4a)")
    args = parser.parse_args()

    data_dir = args.dir
    if not (data_dir / "heldout_summary.csv").is_file():
        for candidate_sub in ["experiment4a_results_exp3parity", "experiment4a_results"]:
            if (data_dir / candidate_sub / "heldout_summary.csv").is_file():
                data_dir = data_dir / candidate_sub
                break

    out_dir = args.out if args.out is not None else (data_dir / "figures_exp4a")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Experiment 4A data from: {data_dir.resolve()}")
    data = load_or_reconstruct_data(data_dir)

    f1 = plot_fig1_coverage(data["cov"], out_dir)
    print(f"Saved Figure 1: {f1}")

    f2 = plot_fig2_overall_and_shrinkage(data["opp"], data["hs"], data["hoc"], out_dir)
    print(f"Saved Figure 2: {f2}")

    f3 = plot_fig3_circuit_breakdown(data["hoc"], out_dir)
    print(f"Saved Figure 3: {f3}")

    f4 = plot_fig4_plan_level_diagnostics(data["soc"], data["hoc"], data["hps"], out_dir)
    print(f"Saved Figure 4: {f4}")

    print("\nAll 4 figures successfully generated!")


if __name__ == "__main__":
    main()
