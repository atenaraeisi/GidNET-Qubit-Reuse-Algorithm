"""Routers for the topology experiment (Experiment 8).

Same configuration as Experiment 4A (gid_vs_caqr.get_fair_metrics):
    transpile(qc, coupling_map=..., routing_method="sabre", layout_method="sabre",
              optimization_level=1, seed_transpiler=seed)
    swap  = tp.count_ops().get("swap", 0)       depth = tp.depth()
ONLY the coupling map changes.  Run check_router_parity.py first: route_hh7 must reproduce your
existing cost_table rows exactly, otherwise nothing measured on the other topologies is comparable.

Topologies (physical qubit counts are kept similar so that 'topology', not 'device size', is what changes):
    hh7  : CouplingMap.from_heavy_hex(7)     115 qubits, degree <= 3   (the baseline of 4A)
    grid : CouplingMap.from_grid(10, 12)     120 qubits, degree <= 4   (richer connectivity)
    line : CouplingMap.from_line(115)        115 qubits, degree <= 2   (poorest connectivity)
Override with env vars, e.g.  TOPO_GRID=8x8  TOPO_LINE=64   (must be >= the largest circuit's width).
"""
import os
from functools import lru_cache

from qiskit import transpile
from qiskit.transpiler import CouplingMap

OPT_LEVEL = int(os.environ.get('TOPO_OPT_LEVEL', '1'))


@lru_cache(maxsize=None)
def coupling(name: str) -> CouplingMap:
    if name == 'hh7':
        return CouplingMap.from_heavy_hex(7)
    if name == 'grid':
        r, c = (int(x) for x in os.environ.get('TOPO_GRID', '10x12').lower().split('x'))
        return CouplingMap.from_grid(r, c)
    if name == 'line':
        return CouplingMap.from_line(int(os.environ.get('TOPO_LINE', '115')))
    raise ValueError(name)


def _route(qc, seed: int, name: str):
    cm = coupling(name)
    if qc.num_qubits > cm.size():
        raise ValueError(f'circuit has {qc.num_qubits} qubits > topology {name} ({cm.size()})')
    tp = transpile(qc, coupling_map=cm, routing_method='sabre', layout_method='sabre',
                   optimization_level=OPT_LEVEL, seed_transpiler=int(seed))
    return dict(swap_count=tp.count_ops().get('swap', 0), routed_depth=tp.depth())


def route_hh7(qc, seed: int):
    return _route(qc, seed, 'hh7')


def route_grid(qc, seed: int):
    return _route(qc, seed, 'grid')


def route_line(qc, seed: int):
    return _route(qc, seed, 'line')
