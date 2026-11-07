#!/usr/bin/env python3
"""Exploratory Adaptive Repair Search (pre-routing search, no routing during selection).

Requires the ORIGINAL GidNET project (reuse_adversarial.py,
experiment3_pipeline_adapter.py, exp4c1_build_table.py, exp4d_features.py,
exp4d_predictability.py) and the original QASM benchmark files.

LOCO protocol: a held-out circuit never contributes routing cost to model fitting.
Training uses 4C-1 neighbour pairs and 35 seeds of *other* circuits.
Test selection uses only structural features; 30 held-out seeds are read
AFTER plan selection for evaluation.  This is a prototype, not yet a
validated hardware-aware method: the old Ridge model has no coupling-map
features and only learns 1-step changes.  Model refit uses B pool by default.

On-the-fly generation uses the real operators and independent checker,
not an oracle lookup into the enumerated solution space.  Exact-space
routing data are used ONLY to measure selected plans' held-out costs.

Example from repository root (once exp4d_features_all.csv exists):
  python adaptive_repair_prototype.py \\
    --c1-dir exp4c1_all --space-dir space_out \\
    --feature-file exp4d_features_all.csv \\
    --qasm-dir benchmarks_compact --out-dir adaptive_out \\
    --beam-width 5 --max-depth 5 --budget 120

Feature generation prerequisite:
  python exp4d_features.py --catalog exp4c1_all/catalog.csv \\
    --qasm-dir benchmarks_compact --out exp4d_features_all.csv

First run with --circuits cnt3-5_179.qasm --max-sources 1 for smoke testing.
Use --self-test to check core logic without Qiskit/QASM.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

DISC = list(range(42, 47))
HELD = list(range(100, 130))
OBJECTIVES = ("swap_count", "routed_depth")
MODEL_CACHE_VERSION = "ridge_nb_loco_v1"


def training_fingerprint(c1, feature_file, lams):
    """Hash the source data and training implementation, not routing-test data."""
    h = hashlib.sha256()
    h.update(MODEL_CACHE_VERSION.encode())
    h.update(json.dumps(list(lams), separators=(",", ":")).encode())
    for path in [c1 / "catalog.csv", c1 / "pools.csv",
                 c1 / "features.csv", c1 / "cost_table.csv", feature_file,
                 Path(__file__).with_name("exp4d_predictability.py")]:
        h.update(path.name.encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def model_filename(model_dir, objective, heldout_circuit):
    # No user-controlled characters outside alphanumeric, underscore, dot and dash.
    clean = "".join(c if c.isalnum() or c in "-_" else "_" for c in heldout_circuit)
    short_hash = hashlib.sha256(heldout_circuit.encode("utf-8")).hexdigest()[:10]
    return model_dir / (objective + "__" + clean + "__" + short_hash + ".json")


def load_cached_ridge(path, objective, heldout_circuit, columns, lams, fingerprint):
    """Read a numeric JSON model (no pickle). Fail closed on stale/incompatible models."""
    data = json.loads(path.read_text(encoding="utf-8"))
    expectations = {
        "cache_version": MODEL_CACHE_VERSION, "objective": objective,
        "heldout_circuit": heldout_circuit, "columns": list(columns),
        "lambda_candidates": list(lams), "training_fingerprint": fingerprint,
    }
    for k, value in expectations.items():
        if data.get(k) != value:
            raise ValueError(f"Cached model {path}: stale or incompatible field {k}")
    w = np.asarray(data["weights"], dtype=float)
    mu = np.asarray(data["feature_mean"], dtype=float)
    sd = np.asarray(data["feature_sd"], dtype=float)
    intercept = float(data["target_mean"])
    if any(a.shape != (len(columns),) for a in (w, mu, sd)):
        raise ValueError(f"Cached model {path}: incorrect array length")
    if not all(np.isfinite(a).all() for a in (w, mu, sd)) or (sd <= 0).any() or not np.isfinite(intercept):
        raise ValueError(f"Cached model {path}: invalid numeric values")
    if heldout_circuit in data["training_circuits"]:
        raise ValueError(f"Cached model {path}: test circuit leaked into training")
    def predict(T):
        return ((np.asarray(T, float) - mu) / sd) @ w + intercept
    return predict


def fit_and_cache_ridge(path, train, old, objective, heldout_circuit,
                        columns, lams, fingerprint):
    training = train[train.circuit != heldout_circuit]
    if training.empty:
        raise ValueError(f"No training circuits for held-out circuit {heldout_circuit}")
    dcols = ["d_" + c for c in columns]
    lam = old.pick_lambda(training, dcols, lams, "gain_all")
    X = training[dcols].to_numpy(float)
    y = training.gain_all.to_numpy(float)
    # Match exp4d_predictability.ridge_fit exactly, including ddof=0.
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    ym = y.mean()
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - ym))
    data = dict(cache_version=MODEL_CACHE_VERSION, objective=objective,
        heldout_circuit=heldout_circuit, columns=list(columns),
        lambda_candidates=list(lams), training_fingerprint=fingerprint,
        selected_lambda=float(lam), feature_mean=mu.tolist(),
        feature_sd=sd.tolist(), weights=w.tolist(), target_mean=float(ym),
        training_circuits=sorted(training.circuit.unique().tolist()),
        n_training_pairs=len(training),
        note="Ridge NB trained on other circuits' B-pool one-step pairs; fixed 35 seeds")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return load_cached_ridge(path, objective, heldout_circuit, columns, lams, fingerprint)


def get_ridge_model(model_dir, mode, train_getter, old, objective,
                    heldout_circuit, columns, lams, fingerprint):
    path = model_filename(model_dir, objective, heldout_circuit)
    if mode != "retrain" and path.exists():
        try:
            return (load_cached_ridge(path, objective, heldout_circuit,
                        columns, lams, fingerprint), "loaded", path)
        except (ValueError, KeyError, TypeError) as e:
            if mode == "load":
                raise
            print(f"WARNING: {e}; retraining this fold", flush=True)
    if mode == "load":
        raise FileNotFoundError(f"No cached model for {objective}/{heldout_circuit}: {path}")
    model = fit_and_cache_ridge(path, train_getter(), old, objective,
               heldout_circuit, columns, lams, fingerprint)
    return model, "trained", path


def cap_search(start, generate, predict, rng, *, mode="guided", beam_width=5,
               max_depth=5, budget=120, per_node_cap=50, patience=2, min_gain=0,
               trace_candidates=False):
    """Beam/Random traversal. Both use same final learned ranker and budget.

    generate(sig, rng) yields (sig, plan) pairs that are already validated.
    The adapter is responsible for accounting for *every* attempted check,
    including invalid moves, in a real wall-time comparison.
    Predict score in relative percent vs ORIGINAL; don't sum step intercepts.
    """
    assert mode in ("guided", "random")
    if min(beam_width, max_depth, budget, per_node_cap) < 1:
        raise ValueError("Search budgets, depth and beam width must be positive")
    beam = [(start, 0)]
    seen = {start}
    scored = {start: 0.0}
    discovery_depth = {start: 0}
    parents = {start: None}
    candidate_visits = []
    best, best_score = start, 0.0
    examined = 0
    max_reached_depth = 0
    for depth in range(1, max_depth + 1):
        layer = []
        for src, _ in beam:
            for sig, plan in generate(src, rng):
                if sig in seen:
                    continue
                if examined >= budget:
                    break
                seen.add(sig)
                examined += 1
                p = float(predict(sig, plan))
                if not np.isfinite(p):
                    raise ValueError("Nonfinite model prediction")
                scored[sig] = p
                discovery_depth[sig] = depth
                parents[sig] = src
                if trace_candidates:
                    candidate_visits.append((sig, depth, p))
                layer.append((sig, p))
                if p > best_score + min_gain:
                    best, best_score = sig, p
                if len(layer) >= beam_width * per_node_cap:
                    break
            if examined >= budget or len(layer) >= beam_width * per_node_cap:
                break
        if not layer:
            break
        max_reached_depth = depth
        if mode == "guided":
            layer.sort(key=lambda x: (-x[1], x[0]))
        else:
            rng.shuffle(layer)
        beam = layer[:beam_width]
        # Do not terminate merely because a step is worse: bridge moves can help.
        # Only the configured depth/budget limits search in this conservative v1.
        if examined >= budget:
            break
    chosen = best if best_score > 0 else start
    path = []
    s = chosen
    while s is not None:
        path.append(s)
        s = parents[s]
    path.reverse()
    assert len(path)-1 == discovery_depth[chosen]
    return dict(pick=chosen, predicted_gain_pct=max(0.0, best_score),
                n_scored=examined, n_seen=len(seen), max_depth_reached=max_reached_depth,
                pick_discovery_move_depth=discovery_depth[chosen], pick_path=path,
                candidate_visits=candidate_visits)


def self_test():
    # A non-monotone bridge: A -> B is bad, B -> C is good overall.
    graph = {"A": ["B", "D"], "B": ["C"], "C": [], "D": []}
    scores = {"A": 0, "B": -2, "C": 6, "D": 1}
    def gen(s, _rng):
        for t in graph[s]:
            yield t, t
    def score(sig, plan): return scores[sig]
    a = cap_search("A", gen, score, random.Random(0), mode="guided", beam_width=2,
                   max_depth=3, budget=10)
    assert a["pick"] == "C", a
    assert a["pick_discovery_move_depth"] == 2, a
    assert a["pick_path"] == ["A", "B", "C"], a
    b = cap_search("A", gen, lambda *_: -1.0, random.Random(0), mode="guided",
                   beam_width=2, max_depth=3, budget=10)
    assert b["pick"] == "A", b
    assert b["pick_discovery_move_depth"] == 0, b
    t = cap_search("A", gen, score, random.Random(0), mode="guided", beam_width=2,
                   max_depth=3, budget=10, trace_candidates=True)
    assert ("C", 2, 6) in t["candidate_visits"], t
    print("self-test passed: non-monotone bridge, zero-move acceptance, trace depths")


def prepare_features(c1, f4_file):
    import exp4d_predictability as old
    f1 = pd.read_csv(c1 / "features.csv")
    f4 = pd.read_csv(f4_file)
    full = f4.merge(f1[["circuit", "sig", "logical_depth", "norm_depth"]],
                    on=["circuit", "sig"], validate="one_to_one")
    catalog = pd.read_csv(c1 / "catalog.csv")
    if len(full) != len(catalog):
        raise ValueError(f"Feature catalog mismatch ({len(full)} vs {len(catalog)}); do not continue")
    cols0 = [c for c in full if c not in ["circuit", "sig"]]
    full[cols0] = full[cols0].astype(float)
    norm = full.copy()
    stats = {}
    for c, rows in full.groupby("circuit"):
        mu, sd = rows[cols0].mean(), rows[cols0].std(ddof=0)
        stats[c] = (mu, sd.replace(0, np.nan))
        norm.loc[rows.index, cols0] = ((rows[cols0] - mu) / sd.replace(0, np.nan)).fillna(0)
    cols = [f for f in cols0 if norm[f].abs().sum() != 0]
    Z = norm.set_index(["circuit", "sig"])[cols]
    return catalog, Z, stats, cols


def build_training(c1, Z, cols, objective):
    pools = pd.read_csv(c1 / "pools.csv")
    cost = pd.read_csv(c1 / "cost_table.csv")
    cost = cost[(cost.status == "ok") & (cost.seed.isin(DISC + HELD))]
    piv = cost.pivot_table(index=["circuit", "sig"], columns="seed", values=objective)
    piv = piv.dropna(subset=DISC + HELD)
    means = piv[DISC + HELD].mean(axis=1)
    rows = []
    for (c, source_id), g in pools.groupby(["circuit", "source_plan_id"]):
        original = g[g.is_original]
        if len(original) != 1:
            raise ValueError(f"Expected one original: {c}/{source_id}")
        osig = original.sig.iloc[0]
        if (c, osig) not in means.index or (c, osig) not in Z.index:
            continue
        orig = Z.loc[(c, osig)].to_numpy(float)
        base = means[(c, osig)]
        if base <= 0:
            continue
        for sig in g[g.in_B & ~g.is_original].sig:
            if (c, sig) not in means.index or (c, sig) not in Z.index:
                continue
            delta = Z.loc[(c, sig)].to_numpy(float) - orig
            r = {"circuit": c, "gain_all": 100 * (base - means[(c, sig)]) / base}
            r.update({"d_" + col: float(v) for col, v in zip(cols, delta)})
            rows.append(r)
    if not rows:
        raise ValueError("No valid training neighbour pairs found")
    return pd.DataFrame(rows), means


class RepositoryAdapter:
    """Real mutation + original validity check; optional *safe* allowed prefilter.
    No direct access to heldout or discovered routing scores during selection.
    """
    def __init__(self, circuit, initial_catalog, qasm_dir, cols, stats, Z):
        import exp4c1_build_table as b
        import exp4d_features as ff
        from reuse_adversarial import canonical, unique_mutations, reference_static_feasible
        from experiment3_pipeline_adapter import build_dynamic_circuit, normalize_for_routing
        self.b, self.ff = b, ff
        self.canonical, self.unique_mutations = canonical, unique_mutations
        self.feasible = reference_static_feasible
        self.build_dynamic, self.normalize = build_dynamic_circuit, normalize_for_routing
        self.circuit, self.cols = circuit, cols
        self.stats = stats[circuit]
        self.qc = b.load_qc(str(qasm_dir / circuit))
        qc_compact, mapping = ff.prepare_gidnet_input(self.qc)
        if any(old != new for old, new in mapping.items()):
            raise ValueError(f"{circuit}: nonidentity compaction {mapping}; correct indices before running")
        self.qc_compact = qc_compact
        self.profiles, self.timing = ff.build_interaction_profiles(qc_compact)
        self.plan_of = {(circuit, r.sig): canonical(b.parse_chains(r.plan_json))
                        for r in initial_catalog.itertuples() if r.circuit == circuit}
        self.Z = {sig: Z.loc[(circuit, sig)].to_numpy(float)
                  for (c, sig) in Z.index if c == circuit}
        self.validity_cache = {}
        self.counters = defaultdict(int)

    def vector(self, sig, plan):
        if sig in self.Z:
            return self.Z[sig]
        f = self.ff.plan_features(self.qc_compact, self.profiles, self.timing, plan)
        dyn = self.build_dynamic(self.qc, plan)
        norm = self.normalize(dyn)
        f["logical_depth"] = dyn.depth()
        f["norm_depth"] = norm.depth()
        mu, sd = self.stats
        z = []
        for col in self.cols:
            v = f.get(col, np.nan)
            if pd.isna(v): v = float(mu[col])
            z.append(float((v - mu[col]) / sd[col]) if not pd.isna(sd[col]) else 0.0)
        a = np.nan_to_num(np.asarray(z, float))
        self.Z[sig] = a
        return a

    def generate(self, sig, rng, ops, per_expand=60, max_attempts=800):
        plan = self.plan_of[(self.circuit, sig)]
        width = len(plan)
        operator_list = list(ops)
        rng.shuffle(operator_list)
        found = set()
        valid_count, attempted = 0, 0
        for op in operator_list:
            # Sample candidate ordering to avoid biased truncation.
            mutants = list(self.unique_mutations(plan, op))
            rng.shuffle(mutants)
            for cand in mutants:
                if attempted >= max_attempts or valid_count >= per_expand:
                    return
                attempted += 1
                self.counters["generated"] += 1
                ch = self.canonical(cand)
                s = self.b.sig_of(ch)
                if s == sig or s in found:
                    continue
                found.add(s)
                if (self.circuit, s) not in self.plan_of:
                    self.counters["checker_calls"] += 1
                    if not self.feasible(self.qc, ch, width):
                        self.counters["rejected"] += 1
                        continue
                    self.plan_of[(self.circuit, s)] = ch
                self.counters["valid"] += 1
                valid_count += 1
                yield s, ch


def load_heldout(cost_paths):
    frames = []
    for f in cost_paths:
        a = pd.read_csv(f)
        frames.append(a[(a.status == "ok") & a.seed.isin(HELD)])
    cost = pd.concat(frames, ignore_index=True).drop_duplicates(["circuit", "sig", "seed"])
    if cost.duplicated(["circuit", "sig", "seed"]).any():
        raise AssertionError("duplicate costs")
    piv = {}
    for objective in OBJECTIVES:
        p = cost.pivot_table(index=["circuit", "sig"], columns="seed", values=objective)
        m = p.dropna(subset=HELD)[HELD].mean(axis=1)
        piv[objective] = m
    return piv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--c1-dir", type=Path)
    ap.add_argument("--space-dir", type=Path)
    ap.add_argument("--feature-file", type=Path)
    ap.add_argument("--qasm-dir", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("adaptive_out"))
    ap.add_argument("--circuits", default="")
    ap.add_argument("--max-sources", type=int, default=0)
    ap.add_argument("--beam-width", type=int, default=5)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--per-expand", type=int, default=40)
    ap.add_argument("--max-attempts-per-expand", type=int, default=500)
    ap.add_argument("--ops", default="suffix_swap,chain_transfer")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lams", default="1,10,100,1000")
    ap.add_argument("--model-dir", type=Path, default=Path("adaptive_ridge_models"),
                    help="Directory for reusable per-objective/per-heldout-circuit Ridge JSON models")
    ap.add_argument("--model-mode", choices=("auto", "load", "retrain"), default="auto",
                    help="auto: reuse or fit missing/stale models; load: never train; retrain: overwrite")
    ap.add_argument("--trace-candidates", action="store_true",
                    help="Log all scored candidate predictions and lookup their heldout outcomes only AFTER search")
    ap.add_argument("--fit-only", action="store_true",
                    help="Fit/cache models without QASM files or running a search")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    required = ("c1_dir", "feature_file") if args.fit_only else (
        "c1_dir", "space_dir", "feature_file", "qasm_dir")
    for k in required:
        if getattr(args, k) is None:
            ap.error("--" + k.replace('_', '-') + " is required")
    source_files = [args.c1_dir / "catalog.csv", args.c1_dir / "pools.csv",
                    args.c1_dir / "features.csv", args.c1_dir / "cost_table.csv",
                    args.feature_file]
    if not args.fit_only:
        source_files.append(args.space_dir / "cost_table.csv")
    for p in source_files:
        if not p.exists():
            ap.error(f"Missing input: {p}")
    if not args.fit_only:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    import exp4d_predictability as old
    catalog, Z, stats, cols = prepare_features(args.c1_dir, args.feature_file)
    lams = [float(l) for l in args.lams.split(',')]
    fingerprint = training_fingerprint(args.c1_dir, args.feature_file, lams)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    orig = pd.read_csv(args.c1_dir / "pools.csv")
    orig = orig[orig.is_original][["circuit", "source_plan_id", "sig"]]
    orig = orig.drop_duplicates(["circuit", "source_plan_id"])
    if args.circuits:
        orig = orig[orig.circuit.isin(args.circuits.split(','))]
    if args.max_sources:
        orig = orig.groupby("circuit").head(args.max_sources)
    if orig.empty:
        raise ValueError("No original sources match the requested circuit filter")
    held = None if args.fit_only else load_heldout(
        [args.c1_dir / "cost_table.csv", args.space_dir / "cost_table.csv"])
    all_rows = []
    candidate_rows = []
    cache_counts = defaultdict(int)
    for obj in OBJECTIVES:
        train_holder = [None]
        def get_train():
            if train_holder[0] is None:
                train_holder[0], _ = build_training(args.c1_dir, Z, cols, obj)
            return train_holder[0]
        for c, rows in orig.groupby("circuit", sort=True):
            model, state, model_path = get_ridge_model(
                args.model_dir, args.model_mode, get_train, old,
                obj, c, cols, lams, fingerprint)
            cache_counts[state] += 1
            print(f"MODEL {state:7s} {obj:12s} {c:22s} {model_path}", flush=True)
            if args.fit_only:
                continue
            ad = RepositoryAdapter(c, catalog, args.qasm_dir, cols, stats, Z)
            ops = args.ops.split(',')
            for row in rows.itertuples(index=False):
                original = row.sig
                orig_plan = ad.plan_of[(c, original)]
                original_feature = ad.vector(original, orig_plan)
                def ranker(sig, plan):
                    # Net change vs the starting plan; don't sum per-step intercepts.
                    return model((ad.vector(sig, plan) - original_feature)[None, :])[0]
                for policy in ("ridge_one_step", "adaptive_beam", "random_beam"):
                    seed = args.seed + sum(ord(z) for z in f"{c}:{row.source_plan_id}:{policy}")
                    rng = random.Random(seed)
                    generated_before = dict(ad.counters)
                    start_time = time.perf_counter()
                    layers = 1 if policy == "ridge_one_step" else args.max_depth
                    beam = 1 if policy == "ridge_one_step" else args.beam_width
                    mode = "random" if policy == "random_beam" else "guided"
                    ans = cap_search(original,
                        lambda s, r: ad.generate(s, r, ops,
                          per_expand=args.per_expand,
                          max_attempts=args.max_attempts_per_expand),
                        ranker, rng, mode=mode, beam_width=beam,
                        max_depth=layers, budget=args.budget,
                        per_node_cap=args.per_expand,
                        trace_candidates=args.trace_candidates)
                    pick = ans["pick"]
                    lookup = held[obj]
                    b = lookup.get((c, original), np.nan)
                    v = lookup.get((c, pick), np.nan)
                    rec = dict(objective=obj, policy=policy, circuit=c,
                               source_plan_id=row.source_plan_id,
                               source_sig=original, pick_sig=pick,
                               pick_discovery_move_depth=ans["pick_discovery_move_depth"],
                               pick_path_json=json.dumps(ans["pick_path"]),
                               moved=pick != original, n_scored=ans["n_scored"],
                               max_depth_reached=ans["max_depth_reached"],
                               predicted_gain_pct=ans["predicted_gain_pct"],
                               generated=ad.counters["generated"]-generated_before.get("generated", 0),
                               checker_calls=ad.counters["checker_calls"]-generated_before.get("checker_calls", 0),
                               search_time_sec=time.perf_counter()-start_time,
                               heldout_missing=not(np.isfinite(b) and np.isfinite(v)),
                               base_held=b, pick_held=v,
                               actual_gain_pct=100*(b-v)/b if np.isfinite(b) and np.isfinite(v) and b > 0 else np.nan)
                    all_rows.append(rec)
                    if args.trace_candidates:
                        # Label lookup occurs AFTER the search has fully ended; no label
                        # reaches cap_search/ranker and can influence path selection.
                        for cand_sig, cand_depth, cand_pred in ans["candidate_visits"]:
                            cand_cost = lookup.get((c, cand_sig), np.nan)
                            candidate_rows.append(dict(
                                objective=obj, policy=policy, circuit=c,
                                source_plan_id=row.source_plan_id, source_sig=original,
                                candidate_sig=cand_sig, discovery_move_depth=cand_depth,
                                predicted_gain_pct=cand_pred,
                                actual_gain_pct=(100*(b-cand_cost)/b
                                    if np.isfinite(b) and np.isfinite(cand_cost) and b>0
                                    else np.nan)))
                    print(f"{obj:12s} {policy:16s} {c:22s} {row.source_plan_id:5s} "
                          f"pred={rec['predicted_gain_pct']:+7.2f}% actual={rec['actual_gain_pct']:+7.2f}% "
                          f"scored={rec['n_scored']}", flush=True)
    print(f"MODEL CACHE: {dict(cache_counts)} (folder={args.model_dir})", flush=True)
    if args.fit_only:
        print("Model fitting/caching complete; no search or routing evaluation performed.")
        return
    results = pd.DataFrame(all_rows)
    results.to_csv(args.out_dir / "per_source.csv", index=False)
    if args.trace_candidates:
        pd.DataFrame(candidate_rows, columns=["objective","policy","circuit",
            "source_plan_id","source_sig","candidate_sig","discovery_move_depth",
            "predicted_gain_pct","actual_gain_pct"]).to_csv(
                args.out_dir / "candidate_trace.csv", index=False)
    summary = results.groupby(["objective", "policy"]).agg(
       n_sources=("actual_gain_pct", "size"),
       n_missing=("heldout_missing", "sum"),
       mean_scored=("n_scored", "mean"),
       mean_search_time_sec=("search_time_sec", "mean")).reset_index()
    macro = results.groupby(["objective", "policy", "circuit"]).actual_gain_pct.mean().groupby(level=[0,1]).mean()
    summary["macro_gain_pct"] = [macro.get((r.objective, r.policy), float('nan')) for r in summary.itertuples()]
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    meta = dict(n_features=len(cols), columns=cols, training="LOCO, 4C-1 B neighbours, 35 train seeds",
                test="30 heldout seeds looked up only after selection",
                model="old ridge_nb (1-step), not multi-step-trained",
                caveat="on-the-fly structural features; normalization estimated from 4C-1 test-circuit catalog; topology not used",
                model_cache_dir=str(args.model_dir), model_cache_status=dict(cache_counts),
                training_fingerprint=fingerprint,
                params={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
    (args.out_dir / "manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("\n" + summary.to_string(index=False))
    if summary.n_missing.any():
        print("WARNING: Some selected plans missing 30-seed heldout data; do not report those results as evaluated")

if __name__ == "__main__":
    main()
