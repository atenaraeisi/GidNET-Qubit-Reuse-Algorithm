"""Pre-materialization, validity-aware fixed-width reuse mutations.

Based on PlanSpace.allowed and PlanSpace._add_edge from the user's exact
plan_space.py; this is STATIC feasibility for the supported circuit subset.

The two production operators are suffix_swap and chain_transfer.  The two
additional historical adversarial operators are supported for parity audits,
but are not automatically added to the pilot's search policy.

Input plans must be valid, cover all active qubits, and retain the target width.
Never use this as a replacement for compiler-semantic checks.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Iterator

from plan_space import PlanSpace, edges_of
from reuse_adversarial import canonical

SUPPORTED_OPERATORS = (
    'suffix_swap', 'chain_transfer', 'node_swap_reconnect', 'segment_reverse'
)


@dataclass(frozen=True)
class Mutation:
    op: str
    i: int
    j: int
    a: int
    b: int
    removed: frozenset[tuple[int, int]]
    added: frozenset[tuple[int, int]]

    def materialize(self, plan):
        """Only construct an entire candidate plan after the fast filter passes."""
        chunks = list(plan)
        if self.op == 'suffix_swap':
            left, right = plan[self.i], plan[self.j]
            chunks[self.i] = left[:self.a] + right[self.b:]
            chunks[self.j] = right[:self.b] + left[self.a:]
        elif self.op == 'chain_transfer':
            src, dst = plan[self.i], plan[self.j]
            q = src[self.a]
            chunks[self.i] = src[:self.a] + src[self.a + 1:]
            chunks[self.j] = dst[:self.b] + (q,) + dst[self.b:]
        elif self.op == 'node_swap_reconnect':
            left, right = list(plan[self.i]), list(plan[self.j])
            left[self.a], right[self.b] = right[self.b], left[self.a]
            chunks[self.i], chunks[self.j] = tuple(left), tuple(right)
        elif self.op == 'segment_reverse':
            src = plan[self.i]
            chunks[self.i] = src[:self.a] + src[self.a:self.b][::-1] + src[self.b:]
        else:
            raise ValueError(self.op)
        return canonical(chunks)


def _edge(left, right):
    return (left, right)


def iter_mutation_deltas(plan, operator: str) -> Iterator[Mutation]:
    """Same enumeration order and parameter ranges as reuse_adversarial.OPERATORS.

    Each mutation provides the edges that change, without constructing chains.
    No candidate is rejected or shuffled here.
    """
    if operator == 'suffix_swap':
        for i, j in combinations(range(len(plan)), 2):
            l, r = plan[i], plan[j]
            for a in range(1, len(l) + 1):
                for b in range(1, len(r) + 1):
                    old, new = [], []
                    if a < len(l):
                        old.append(_edge(l[a-1], l[a]))
                        new.append(_edge(r[b-1], l[a]))
                    if b < len(r):
                        old.append(_edge(r[b-1], r[b]))
                        new.append(_edge(l[a-1], r[b]))
                    yield Mutation(operator, i, j, a, b, frozenset(old), frozenset(new))
    elif operator == 'chain_transfer':
        for i, src in enumerate(plan):
            if len(src) <= 1:
                continue
            for a, q in enumerate(src):
                for j, dst in enumerate(plan):
                    if i == j:
                        continue
                    for b in range(len(dst)+1):
                        old, new = [], []
                        if a:
                            old.append(_edge(src[a-1],q))
                        if a+1 < len(src):
                            old.append(_edge(q,src[a+1]))
                        if a and a+1 < len(src):
                            new.append(_edge(src[a-1],src[a+1]))
                        if b and b < len(dst):
                            old.append(_edge(dst[b-1],dst[b]))
                        if b:
                            new.append(_edge(dst[b-1],q))
                        if b < len(dst):
                            new.append(_edge(q,dst[b]))
                        yield Mutation(operator, i, j, a, b, frozenset(old), frozenset(new))
    elif operator == 'node_swap_reconnect':
        for i, j in combinations(range(len(plan)), 2):
            l,r=plan[i],plan[j]
            for a in range(len(l)):
                for b in range(len(r)):
                    x,y=l[a],r[b]
                    old,new=[],[]
                    if a:
                        old.append(_edge(l[a-1],x));new.append(_edge(l[a-1],y))
                    if a+1 < len(l):
                        old.append(_edge(x,l[a+1]));new.append(_edge(y,l[a+1]))
                    if b:
                        old.append(_edge(r[b-1],y));new.append(_edge(r[b-1],x))
                    if b+1 < len(r):
                        old.append(_edge(y,r[b+1]));new.append(_edge(x,r[b+1]))
                    yield Mutation(operator, i, j, a, b, frozenset(old), frozenset(new))
    elif operator == 'segment_reverse':
        for i, chain in enumerate(plan):
            for a in range(len(chain)):
                for b in range(a + 2, len(chain)+1):
                    old,new=[],[]
                    if a:
                        old.append(_edge(chain[a-1],chain[a]))
                        new.append(_edge(chain[a-1],chain[b-1]))
                    if b < len(chain):
                        old.append(_edge(chain[b-1],chain[b]))
                        new.append(_edge(chain[a],chain[b]))
                    for k in range(a,b-1):
                        old.append(_edge(chain[k],chain[k+1]))
                        new.append(_edge(chain[k+1],chain[k]))
                    yield Mutation(operator, i, -1, a, b, frozenset(old), frozenset(new))
    else:
        raise ValueError(f'Unknown operator {operator!r}; supported: {SUPPORTED_OPERATORS}')


class PreparedPlan:
    def __init__(self, engine: 'ValidityAwareEngine', plan):
        self.engine = engine
        self.plan = canonical(plan)
        self.width = len(self.plan)
        self.base = edges_of(self.plan)
        # Important: original validity is checked once per parent plan, not per
        # rejected mutation.  All mutations conserve width and active membership.
        result = engine.ps.ck.validate(self.plan, target_width=self.width)
        if not result.is_valid:
            raise ValueError(f'Input plan is not statically valid: {result.reason}')
        if set(q for c in self.plan for q in c) != set(engine.ps.active):
            raise ValueError('Input plan does not cover active qubits exactly')
        self._closure_without = {}
        self.seen = set()

    def _candidate_key(self, mut: Mutation):
        if not mut.removed.issubset(self.base):
            raise AssertionError(f'Bad removed-edge delta: {mut}')
        e = (self.base - mut.removed) | mut.added
        # A mutation must not change the number of reuse edges / active width.
        if len(e) != len(self.base):
            raise AssertionError(f'Non-width-preserving edge delta: {mut}')
        return frozenset(e)

    def admissible(self, mut: Mutation) -> str:
        """Return 'valid', 'illegal_boundary', or 'dependency_cycle'."""
        ps = self.engine.ps
        for a, b in mut.added:
            if a not in ps.pos or b not in ps.pos or not ps.allowed[ps.pos[a]][ps.pos[b]]:
                return 'illegal_boundary'
        # Start from the unchanged portion of the *current* plan, then add only
        # the new boundaries with the bitset transitive-closure update.
        removed = mut.removed
        if removed not in self._closure_without:
            remaining = sorted(self.base - removed)
            self._closure_without[removed] = ps.closure_of(
                [(ps.pos[a], ps.pos[b]) for a,b in remaining])
        reach = self._closure_without[removed]
        if reach is None:
            raise AssertionError('A subset of a statically valid plan created a cycle')
        for a, b in sorted(mut.added):
            reach = ps._add_edge(reach, ps.pos[a], ps.pos[b])
            if reach is None:
                return 'dependency_cycle'
        return 'valid'

    def scan(self, operator: str, *, max_attempts: int = 0,
             verify_survivors: bool = False):
        """Return (valid mutation plans, counters), exhaustively or with a raw cap.

        If max_attempts > 0, the cap matches the legacy pilot's raw enumeration
        cap, NOT a count of valid neighbors. No candidate shuffling occurs here.
        The caller may shuffle the valid list and keep a subset as before.
        """
        result = []
        stats = Counter()
        for index, mut in enumerate(iter_mutation_deltas(self.plan, operator)):
            if max_attempts and index >= max_attempts:
                break
            stats['attempted'] += 1
            key = self._candidate_key(mut)
            if key == self.base:
                stats['same_plan'] += 1
                continue
            if key in self.seen:
                stats['duplicate'] += 1
                continue
            self.seen.add(key)
            status = self.admissible(mut)
            if status != 'valid':
                stats['fast_reject_' + status] += 1
                continue
            plan = mut.materialize(self.plan)
            if edges_of(plan) != key:
                raise AssertionError('Operator edge delta disagrees with materialized plan')
            if verify_survivors and not self.engine.ps.ck.validate(plan, target_width=self.width).is_valid:
                raise AssertionError('Fast filter accepted a plan rejected by independent checker')
            result.append(plan)
            stats['valid'] += 1
        return result, stats


class ValidityAwareEngine:
    """Precompute allowed matrix and base reachability once per circuit."""
    def __init__(self, checker):
        if checker.unsupported is not None:
            raise ValueError(f'Unsupported source: {checker.unsupported.reason}')
        self.ps = PlanSpace(checker.source_circuit, checker=checker)

    def prepare(self, plan):
        return PreparedPlan(self, plan)
