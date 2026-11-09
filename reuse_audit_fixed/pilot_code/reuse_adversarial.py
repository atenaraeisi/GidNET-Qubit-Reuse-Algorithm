"""Adversarial fixed-width reuse-plan mutations and an independent reference scheduler.

This module NEVER calls ReuseValidityChecker or GidNET.  The reference builds
an operation-level precedence graph directly from QuantumCircuit.data (not
Qiskit's DAGCircuit conversion), then uses its own Kahn topological scheduler.
This is an independent implementation of STATIC dependency feasibility, NOT a
proof of quantum/classical semantic equivalence after reset insertion.

Works with terminal-measurement, otherwise unitary Qiskit circuits used by the
fixed-width experiments.  Mutation generators are evaluation inputs, not an
optimized neighborhood-search algorithm.
"""
from __future__ import annotations

from collections import defaultdict, deque
from itertools import combinations
from numbers import Integral
from typing import Iterable, Iterator, Sequence

Plan = tuple[tuple[int, ...], ...]


def canonical(chains: Sequence[Sequence[int]]) -> Plan:
    """Chain *order* is irrelevant, but order *within* a chain matters."""
    return tuple(sorted(tuple(int(q) for q in chain) for chain in chains))


def pair_boundaries(chains: Sequence[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    return tuple((int(a), int(b)) for chain in chains for a, b in zip(chain, chain[1:]))


def _as_plan(chains: Sequence[Sequence[int]]) -> Plan:
    return tuple(tuple(int(q) for q in chain) for chain in chains)


def suffix_swaps(chains: Plan) -> Iterator[Plan]:
    """Exchange suffixes of two chains, keeping both prefixes nonempty."""
    for i, j in combinations(range(len(chains)), 2):
        left, right = chains[i], chains[j]
        for a in range(1, len(left) + 1):
            for b in range(1, len(right) + 1):
                candidate = list(chains)
                candidate[i] = left[:a] + right[b:]
                candidate[j] = right[:b] + left[a:]
                yield tuple(candidate)


def transfers(chains: Plan) -> Iterator[Plan]:
    """Move one logical qubit from a nonsingleton chain to another chain."""
    for i, src in enumerate(chains):
        if len(src) <= 1:
            continue
        for pos, q in enumerate(src):
            reduced = src[:pos] + src[pos + 1:]
            for j, dst in enumerate(chains):
                if i == j:
                    continue
                for insert in range(len(dst) + 1):
                    candidate = list(chains)
                    candidate[i] = reduced
                    candidate[j] = dst[:insert] + (q,) + dst[insert:]
                    yield tuple(candidate)


def node_swap_reconnections(chains: Plan) -> Iterator[Plan]:
    """Swap individual nodes between two chains, rewiring nearby reuse edges."""
    for i, j in combinations(range(len(chains)), 2):
        for a in range(len(chains[i])):
            for b in range(len(chains[j])):
                candidate = [list(c) for c in chains]
                candidate[i][a], candidate[j][b] = candidate[j][b], candidate[i][a]
                yield tuple(tuple(c) for c in candidate)


def segment_reversals(chains: Plan) -> Iterator[Plan]:
    """Reverse a contiguous section within a chain; useful adversarial input."""
    for i, chain in enumerate(chains):
        for start in range(len(chain)):
            for stop in range(start + 2, len(chain) + 1):
                candidate = list(chains)
                candidate[i] = chain[:start] + chain[start:stop][::-1] + chain[stop:]
                yield tuple(candidate)


OPERATORS = {
    'suffix_swap': suffix_swaps,
    'chain_transfer': transfers,
    'node_swap_reconnect': node_swap_reconnections,
    'segment_reverse': segment_reversals,
}


def unique_mutations(chains: Sequence[Sequence[int]], operator: str) -> list[Plan]:
    """Generate all unique, structurally well-formed, nontrivial neighbors."""
    original = _as_plan(chains)
    key0 = canonical(original)
    seen: set[Plan] = set()
    result: list[Plan] = []
    for candidate in OPERATORS[operator](original):
        if not all(candidate):
            continue
        key = canonical(candidate)
        if key == key0 or key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _is_supported_circuit(circuit) -> bool:
    """Fail closed for unsupported Qiskit input types, independently of checker."""
    from qiskit.circuit import Gate

    measured = set()
    for inst in circuit.data:
        op = inst.operation
        qs = {circuit.find_bit(q).index for q in inst.qubits}
        if getattr(op, 'condition', None) is not None:
            return False
        if op.name == 'barrier':
            continue
        if op.name == 'measure':
            if len(qs) != 1 or len(inst.clbits) != 1:
                return False
            measured.update(qs)
        elif not isinstance(op, Gate) or measured & qs:
            return False
    return True


def reference_static_feasible(circuit, chains: Sequence[Sequence[int]], target_width: int) -> bool:
    """Independent STATIC feasibility oracle for supported circuits.

    Constructs an operation-level precedence relation directly from sequential
    instructions, respecting both quantum and classical wire order. Adds a
    dependency from terminal(a) to root(b) for each proposed reuse boundary.
    Scheduling with Kahn's algorithm must consume every event. Does not use
    NetworkX or the checker internals.

    This checks the SAME static graph-theoretic property via an independent
    construction; it is NOT independent physical-semantic equivalence testing.
    """
    if not _is_supported_circuit(circuit):
        return False
    try:
        plan = tuple(tuple(c) for c in chains)
    except (TypeError, ValueError):
        return False
    if not plan or any(not c for c in plan) or len(plan) != target_width:
        return False
    if any(not isinstance(q, Integral) or isinstance(q, bool) for c in plan for q in c):
        return False
    active = {
        circuit.find_bit(q).index
        for inst in circuit.data if inst.operation.name not in {'measure', 'barrier'}
        for q in inst.qubits
    }
    flat = [int(q) for c in plan for q in c]
    if len(flat) != len(set(flat)) or set(flat) != active:
        return False

    n = circuit.num_qubits
    succ: dict[tuple, set[tuple]] = defaultdict(set)
    indeg: dict[tuple, int] = defaultdict(int)

    def node(x):
        succ[x]
        indeg[x]

    def edge(a, b):
        node(a)
        node(b)
        if b not in succ[a]:
            succ[a].add(b)
            indeg[b] += 1

    last_q = {}
    last_c = {}
    for q in range(n):
        s, e = ('start', q), ('finish', q)
        node(s)
        node(e)
        last_q[q] = s
    for idx, inst in enumerate(circuit.data):
        op = inst.operation
        if op.name == 'barrier':
            continue
        qs = [circuit.find_bit(q).index for q in inst.qubits]
        cs = [circuit.find_bit(c).index for c in inst.clbits]
        event = ('op', idx)
        node(event)
        for q in qs:
            edge(last_q[q], event)
            last_q[q] = event
        for c in cs:
            if c in last_c:
                edge(last_c[c], event)
            last_c[c] = event
    for q in range(n):
        edge(last_q[q], ('finish', q))
    for chain in plan:
        for a, b in zip(chain, chain[1:]):
            edge(('finish', int(a)), ('start', int(b)))

    queue = deque(v for v, deg in indeg.items() if deg == 0)
    visited = 0
    while queue:
        v = queue.popleft()
        visited += 1
        for w in succ[v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                queue.append(w)
    return visited == len(indeg)


def malformed_variants(chains: Sequence[Sequence[int]], num_qubits: int):
    """Yield intentionally invalid input structures with expected checker reasons.

    These are separate from valid-shape mutations; each expected rejection is
    known from the input contract, without consulting the checker.
    """
    orig = _as_plan(chains)
    if not orig:
        return
    yield 'empty_chain', ((),) + orig, 'empty_chain', len(orig)
    yield 'width_mismatch', orig, 'width_mismatch', len(orig) + 1
    yield 'non_integer_qubit', orig[:-1] + (orig[-1] + ('not_a_qubit',),), 'non_integer_qubit', len(orig)
    if orig[0]:
        yield 'duplicate', (orig[0] + (orig[0][0],),) + orig[1:], 'duplicate_logical_qubit', len(orig)
        removed = orig[0][1:]
        reduced = (removed,) + orig[1:] if removed else orig[1:]
        # A singleton chain being removed may produce an empty/width issue;
        # check only cases where all chains remain nonempty.
        if reduced and all(reduced):
            yield 'missing_qubit', reduced, 'active_qubit_set_mismatch', len(reduced)
    yield 'out_of_range', (orig[0] + (int(num_qubits),),) + orig[1:], 'active_qubit_set_mismatch', len(orig)
