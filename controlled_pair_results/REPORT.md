# Controlled pair structural forensics — exploratory (Stage A)

No GidNET generation or SABRE routing was performed. These are descriptive differences, NOT causal explanations.
Plans are fixed-width from same original circuit. 30-seed outcomes come from previously saved routing results.
Gate-level reconstruction is shown ONLY when corresponding original QASM is present and cross-checked against Experiment 2.

## system9.qasm: P5 versus P4

- Routed depth (30 seeds): 298.300 → 303.167, B−A = +4.867.
- SWAP count: 90.867 → 90.300, B−A = -0.567.
- Matched seed depth comparison: 19 seeds B > A, 0 ties, 11 seeds B < A; n=30.
- Descriptive paired-depth 95% t interval: [1.743, 7.990]. Selected after inspecting routing outcomes: not a confirmatory interval.
- Normalized logical depth: 175 / 175; Max weighted degree: 63 / 63.
- Reuse edges present only in A / B: 1 / 1; reassigned logical qubits after chain alignment: [0].
- QASM status: **VERIFIED_FULL_GATE_ANALYSIS**.
- Original QASM: 330 gates (150 2q). ASAP depth A / B: 169 / 169.
- Operations with changed ASAP level: 3; earlier in B: 1; later in B: 2.
- Two-qubit gates with changed reused-wire chain pair (after alignment): 6 (4.0%).
- Gates entering/leaving the unit-gate-time critical-path set: 0.
- Detailed outputs include moved-qubit interaction history, changed 2q gate endpoints, reuse boundary gate links, pre-routing ASAP shifts, aligned chain partner sequences, and fixed-original-order load bins.

## sym9_146.qasm: P8 versus P4

- Routed depth (30 seeds): 315.133 → 320.300, B−A = +5.167.
- SWAP count: 85.300 → 87.100, B−A = +1.800.
- Matched seed depth comparison: 30 seeds B > A, 0 ties, 0 seeds B < A; n=30.
- Descriptive paired-depth 95% t interval: [3.911, 6.423]. Selected after inspecting routing outcomes: not a confirmatory interval.
- Normalized logical depth: 200 / 200; Max weighted degree: 70 / 70.
- Reuse edges present only in A / B: 1 / 1; reassigned logical qubits after chain alignment: [0].
- QASM status: **VERIFIED_FULL_GATE_ANALYSIS**.
- Original QASM: 328 gates (148 2q). ASAP depth A / B: 188 / 188.
- Operations with changed ASAP level: 4; earlier in B: 2; later in B: 2.
- Two-qubit gates with changed reused-wire chain pair (after alignment): 6 (4.1%).
- Gates entering/leaving the unit-gate-time critical-path set: 4.
- Detailed outputs include moved-qubit interaction history, changed 2q gate endpoints, reuse boundary gate links, pre-routing ASAP shifts, aligned chain partner sequences, and fixed-original-order load bins.

## What these results can and cannot say

1. A plan-level delta in routed depth is not necessarily caused by the specific changed reuse edge. Hardware layout and SABRE decisions also matter.
2. Reused-wire chain IDs are arbitrary. Chain alignment maximizes logical-qubit overlap; ambiguous symmetric matchings may have multiple equally good alignments.
3. ASAP layers assume each original gate lasts one logical time unit. These are **not** routed depth, device duration, or gate wait time.
4. Equal global logical depth / MaxWD can hide local temporal shifts and changed pairwise interactions.
5. A paired 30-seed gap quantifies conditional route variability, but the selected pairs were chosen after looking at the same data. Confirm findings on other plans/circuits.
6. Use Stage B (targeted routed-circuit inspection) only if Stage A provides a focused structural hypothesis; no routing performed here.

## Suggested next inspection

Inspect `*__moved_logical_qubits.csv`, `*__changed_two_qubit_chain_pairs.csv`, `*__operation_asap_or_critical_changes.csv`, and the figures in `figures/`. Look for a change that precedes/aligns with depth pressure, then test it on more fixed-width plans.
