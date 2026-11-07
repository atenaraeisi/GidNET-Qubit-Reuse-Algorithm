"""Router hook for exp4c1_build_table.py.

Copy the routing step EXACTLY from experiment4A_physical_value.py / Experiment 3
(backend, coupling map, basis gates, layout and routing method, optimization level,
the way SWAP count and depth are measured).  If this differs from 4A, the cost table
is not comparable with the 4A results.

Input : the circuit returned by normalize_for_routing()  (resets already lowered)
Output: dict(swap_count=..., routed_depth=...)
"""


def route(qc, seed: int):
    raise NotImplementedError(
        'Fill in route() with the routing/measurement code used in experiment4A_physical_value.py'
    )
    # Sketch of the shape (values here are placeholders, NOT the 4A configuration):
    #
    # from qiskit import transpile
    # out = transpile(qc, coupling_map=COUPLING_MAP, basis_gates=BASIS_GATES,
    #                 layout_method=..., routing_method=..., optimization_level=...,
    #                 seed_transpiler=seed)
    # return dict(swap_count=<same counting as 4A>, routed_depth=out.depth())
