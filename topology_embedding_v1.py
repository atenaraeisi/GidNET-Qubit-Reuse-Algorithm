#!/usr/bin/env python3
"""Routing-free topology-conditioned quotient-graph placement surrogate.

No Qiskit import is needed for scoring. Qiskit is required only to obtain the
EXACT heavy-hex edge set or for the full search's circuit validity checking.

Important: the score is not an estimate or lower bound of actual SABRE SWAPs.
It is a static communication-pressure heuristic, to be falsified on held-out
routing results. Lower is better. No physical routing or label learning occurs.
"""
from __future__ import annotations

import functools
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np


def canonical(plan):
    return tuple(sorted(tuple(int(x) for x in chain) for chain in plan))


def graph_from_coupling_map(cm):
    """Use exactly the hardware coupling supplied by the project's Qiskit version."""
    g = nx.Graph()
    g.add_nodes_from(range(int(cm.size())))
    g.add_edges_from((int(u), int(v)) for u, v in cm.get_edges())
    return g


def graph_for_topology(name: str, *, grid='10x12', line=115):
    """Offline-exact graph for grid/line, Qiskit-derived graph for hh7."""
    name = name.lower().strip()
    if name == 'grid':
        rows, cols = [int(x) for x in grid.lower().split('x')]
        if min(rows, cols) < 1:
            raise ValueError('Positive grid dimensions required')
        g = nx.Graph();g.add_nodes_from(range(rows * cols))
        # Vertex labels and undirected edges follow row-major grid Qiskit semantics.
        for r in range(rows):
            for c in range(cols):
                u = r * cols + c
                if r+1 < rows:g.add_edge(u, (r+1)*cols+c)
                if c+1 < cols:g.add_edge(u, r*cols+c+1)
        return g
    if name == 'line':
        return nx.path_graph(int(line))
    if name == 'hh7':
        try:
            from router_topo import coupling
        except ImportError as exc:
            raise RuntimeError('hh7 requires Qiskit 0.45.3 + router_topo.py in your project. '
                               'Do not approximate the heavy-hex graph with a grid.') from exc
        return graph_from_coupling_map(coupling('hh7'))
    raise ValueError(f'Unsupported topology: {name}')


def read_twoq_qasm2_compact(path):
    """Strict, tiny OPENQASM2 reader for offline score audit only.

    It assumes primitive qreg-indexed gates in supplied benchmark QASMs. It does
    NOT compile custom gate macros or reset semantics. Real search uses Qiskit
    and precisely the same prepare_gidnet_input as prior experiments.
    """
    source = Path(path).read_text(encoding='utf-8')
    if not re.search(r'\bOPENQASM\s+2\.0\s*;', source):
        raise ValueError(f'Not OPENQASM 2: {path}')
    source = re.sub(r'//[^\n]*', '', source)
    regs = re.findall(r'\bqreg\s+(\w+)\s*\[\s*(\d+)\s*\]\s*;', source)
    if not regs:
        raise ValueError('No qreg found')
    offsets = {};total = 0
    for name, count in regs:
        offsets[name] = total; total += int(count)
    seen = set();pairs=[]
    for stmt in source.split(';'):
        s = stmt.strip()
        if not s or re.match(r'^(OPENQASM|include|qreg|creg)\b', s):continue
        if re.search(r'\b(gate|opaque|if|reset)\b', s):
            raise ValueError(f'Unsupported QASM for offline-only audit: {s[:80]}')
        name = s.split('(',1)[0].split()[0].lower()
        if name in ('measure','barrier'):
            continue
        qs = re.findall(r'(\w+)\[(\d+)\]', s)
        if not qs:continue
        indices=[]
        for reg, bit in qs:
            if reg not in offsets:raise ValueError(f'Unknown qreg {reg}')
            indices.append(offsets[reg]+int(bit))
        seen.update(indices)
        if len(indices)==2:
            if indices[0]==indices[1]:raise ValueError('Invalid 2q gate with same operands')
            pairs.append(tuple(indices))
    lut={q:i for i,q in enumerate(sorted(seen))}
    return [tuple(lut[q] for q in p) for p in pairs],len(lut)


@dataclass(frozen=True)
class EmbeddingResult:
    score: float
    placement: tuple[int,...]
    cross_gates: int
    unique_edges: int
    max_degree: int
    hhi: float


class TopologyScorer:
    """Greedy multi-start placement + bounded local swaps, cached by Plan.

    Complexity (after distance precomputation) is roughly
    O(starts * (width**2 * hardware_nodes + local_steps*(width*hardware_nodes+width**2))),
    plus O(two_qubit_gates) for quotient construction. This is an implementation
    estimate, not an exact asymptotic bound for arbitrary graph topology.
    """
    def __init__(self, pairs, physical_graph, *, starts=4, local_steps=1):
        self.pairs=tuple((int(a),int(b)) for a,b in pairs)
        if starts<1 or local_steps<0:raise ValueError('Invalid embedding budgets')
        self.starts=int(starts);self.local_steps=int(local_steps)
        g=physical_graph.to_undirected() if physical_graph.is_directed() else physical_graph.copy()
        if g.number_of_nodes()==0:raise ValueError('Empty physical graph')
        nodes=sorted(g.nodes())
        self.nodes=tuple(nodes)
        self.node_idx={u:i for i,u in enumerate(nodes)}
        self.size=len(nodes)
        # Disconnected devices are rejected until placements are component-aware.
        # This avoids silently inventing SWAP paths between islands.
        if not nx.is_connected(g):raise ValueError('Physical coupling must be connected for v1')
        self.dist=np.zeros((self.size,self.size),dtype=np.float64)
        for u,lens in nx.all_pairs_shortest_path_length(g):
            for v,steps in lens.items():self.dist[self.node_idx[u],self.node_idx[v]]=steps
        self.excess=np.maximum(self.dist-1.0,0.0)
        degree=np.array([g.degree(u) for u in nodes],dtype=float)
        closeness=self.dist.sum(axis=1)
        self.rank=np.lexsort((np.arange(self.size),-degree,closeness)).tolist()
        self._cache={}

    def _weights(self,plan):
        owner={q:i for i,c in enumerate(plan) for q in c}
        W=np.zeros((len(plan),len(plan)),dtype=np.float64)
        for a,b in self.pairs:
            if a not in owner or b not in owner:
                raise ValueError(f'Plan missing logical qubit from circuit 2q interactions: {(a,b)}')
            x,y=owner[a],owner[b]
            if x==y:
                # Static reuse validity forbids an interacting pair in one chain.
                raise ValueError('Invalid plan: logically interacting qubits in same reuse chain')
            W[x,y]+=1;W[y,x]+=1
        return W

    def _objective(self,W,pidx):
        if len(pidx)<2:return 0.0
        return float(np.sum(np.triu(W,1)*self.excess[np.ix_(pidx,pidx)]))

    def _greedy(self,W,anchor):
        w=W.shape[0]
        strength=W.sum(axis=1)
        first=int(np.argmax(strength))
        pidx=np.full(w,-1,dtype=int)
        pidx[first]=anchor
        used=np.zeros(self.size,dtype=bool);used[anchor]=True
        placed=[first]
        remaining=set(range(w))-{first}
        while remaining:
            # Add the logical chain most connected to the currently placed chains.
            q=min(remaining,key=lambda x:(-float(W[x,placed].sum()),-float(strength[x]),x))
            if np.any(W[q,placed]>0):
                costs=self.excess[:,pidx[placed]] @ W[q,placed]
            else:
                # No placed interactions: pack into region near existing mapping.
                costs=self.dist[:,anchor]*0.001
            # Deterministic tie breaks; priorities never override nontrivial distance cost.
            costs=np.asarray(costs,dtype=float)+self.dist.sum(axis=1)*1e-9
            costs[used]=np.inf
            ph=int(np.argmin(costs))
            pidx[q]=ph;used[ph]=True;placed.append(q);remaining.remove(q)
        return pidx

    def _improve(self,W,placement):
        """Bounded local search: single-chain moves and pairwise site swaps."""
        pidx=np.array(placement,copy=True,dtype=int)
        for _ in range(self.local_steps):
            base=self._objective(W,pidx)
            best=base; bestmove=None
            occupied=set(int(t) for t in pidx)
            # For one-chain moves, change only the incident edge costs.
            for q,old in enumerate(pidx):
                incident=W[q]
                if not np.any(incident):continue
                new_cost=self.excess[:,pidx]@incident
                old_cost=float(new_cost[old])
                change=new_cost-old_cost
                for u in occupied:
                    change[u]=np.inf
                ix=int(np.argmin(change));value=base+float(change[ix])
                if value<best-1e-9:best=value;bestmove=('move',q,ix)
            for a in range(len(pidx)):
                for b in range(a+1,len(pidx)):
                    pidx[a],pidx[b]=pidx[b],pidx[a]
                    value=self._objective(W,pidx)
                    pidx[a],pidx[b]=pidx[b],pidx[a]
                    if value<best-1e-9:best=value;bestmove=('swap',a,b)
            if bestmove is None:break
            mode,a,b=bestmove
            if mode=='move':pidx[a]=b
            else:pidx[a],pidx[b]=pidx[b],pidx[a]
        return pidx

    def score(self,plan):
        key=canonical(plan)
        if key in self._cache:return self._cache[key]
        if len(key)>self.size:raise ValueError('Plan wider than physical device')
        W=self._weights(key)
        degree=(W>0).sum(axis=1)
        edge_weights=W[np.triu_indices(len(key),1)]
        edge_weights=edge_weights[edge_weights>0]
        total=edge_weights.sum()
        hhi=float(np.sum((edge_weights/total)**2)) if total>0 else 0.0
        # Spread anchor seeds across center, middle and periphery of the same graph.
        pos=sorted(set(np.linspace(0,max(0,self.size-1),self.starts,dtype=int).tolist()))
        anchors=[self.rank[i] for i in pos]
        best=None
        for anchor in anchors:
            pidx=self._greedy(W,anchor)
            if self.local_steps:pidx=self._improve(W,pidx)
            val=self._objective(W,pidx)
            if best is None or val<best[0]-1e-9 or (abs(val-best[0])<1e-9 and tuple(pidx)<best[1]):
                best=(val,tuple(map(int,pidx)))
        assert best is not None
        result=EmbeddingResult(score=float(best[0]),placement=tuple(self.nodes[i] for i in best[1]),
                cross_gates=int(total),unique_edges=len(edge_weights),max_degree=int(max(degree,default=0)),hhi=hhi)
        self._cache[key]=result
        return result

    def relative_score(self,plan):
        r=self.score(plan)
        return r.score/max(1,r.cross_gates)

