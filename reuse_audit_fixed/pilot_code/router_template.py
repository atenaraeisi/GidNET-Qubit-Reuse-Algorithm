"""Routing callback for exp4c1_build_table.py.

Matches the routing configuration in experiment4A_physical_value.py:
- Qiskit transpile
- heavy_hex coupling map with distance 7
- SABRE layout + SABRE routing
- optimization_level=1
- seed_transpiler set from the supplied seed

The input circuit is expected to have already been built and normalized by
exp4c1_build_table.py, just as the prepared circuit is routed in Experiment 4A.
"""

from functools import lru_cache

from qiskit import transpile
from qiskit.transpiler import CouplingMap


@lru_cache(maxsize=1)
def _coupling_map():
    """Build the shared heavy-hex topology once per Python process."""
    return CouplingMap.from_heavy_hex(7)


def route(qc, seed):
    """Route one already-prepared circuit and return metrics expected by 4C1.

    Parameters
    ----------
    qc : qiskit.QuantumCircuit
        Circuit after the pipeline's reset/measurement preparation step.
    seed : int
        Transpiler seed for this routing trial.

    Returns
    -------
    dict
        Contains `swap_count` and `routed_depth`, matching Experiment 4A.
    """
    if qc is None:
        raise ValueError("route() received qc=None")

    routed = transpile(
        qc,
        coupling_map=_coupling_map(),
        layout_method="sabre",
        routing_method="sabre",
        optimization_level=1,
        seed_transpiler=int(seed),
    )

    return {
        "swap_count": int(routed.count_ops().get("swap", 0)),
        "routed_depth": int(routed.depth()),
    }
