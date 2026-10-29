# GidNET — Qubit Reuse Research Fork

This repository is a research fork of [GidNET](https://github.com/gideonuchehara/GidNET-Qubit-Reuse-Algorithm), a graph-based qubit-reuse compiler for dynamic quantum circuits.

The original GidNET implementation and algorithm are by **Gideon Uchehara et al.** This fork keeps the original implementation and adds fixes, validation code, and experimental tooling that I use while studying qubit reuse and its physical compilation cost.

## What I changed

### Reset insertion at reuse boundaries

In the original dynamic-circuit construction, reuse edges were represented using DAG output nodes. The compilation loop skips `DAGOutNode` objects, so matching reset locations against those nodes could leave the reset-insertion branch unreachable.

The fork tracks the last actual `DAGOpNode` before each reused qubit boundary and inserts the reset after that operation instead.

This makes the reuse boundary explicit in the compiled circuit and allows later compilation and routing stages to see the physical reset operation.

Relevant code:

- `gidnet/qubitreuse.py`

### Reset lowering for Qiskit experiments

I also added an experimental reset-lowering helper for Qiskit 0.45.x:

```text
reset q
```

can be represented as an active reset:

```text
measure q -> c
x q if c == 1
```

When a valid measurement already immediately precedes the reset boundary, its classical result is reused instead of inserting a duplicate measurement.

Relevant code:

- `gidnet/reset_lowering.py`
- `test_reset_lowering.py`
- `test_reset_lowering_semantics.py`

The tests cover structural behavior, reuse of existing measurements, stale-measurement handling, GidNET integration, input immutability, and ideal-simulator semantic checks.

### Reset-noise diagnostic

`inspect_reset_noise.py` examines how a backend-derived Aer `NoiseModel` treats reset, measurement, and conditional-X operations. It also compares native reset with the measurement-based lowering and includes an explicit reset-noise sanity check.

This script is diagnostic rather than part of the GidNET algorithm itself.

### Benchmark runner

`run_gidnet_on_all_95.py` runs the modified GidNET compiler across the benchmark set used in my local experiments and records width, depth, measurement, reset, and CX counts.

The script currently contains local experiment paths, so those paths need to be adjusted before running it on another machine.

## Current focus

I am using this fork to study a broader question:

> How much can physically relevant compilation cost vary between qubit-reuse solutions that achieve the same logical width?

My current experiments look at fixed-width reuse plans, routing behavior, SWAP overhead, routed depth, and small bounded changes to reuse plans.

The research experiments themselves are still in progress, so this repository will continue to change.

## Original GidNET

GidNET (**Graph-Based Identification of Qubit Network**) is a qubit-reuse algorithm that reduces the number of physical qubits required to execute a quantum circuit. It identifies valid reuse relationships from circuit dependencies and compiles the circuit into a dynamic form.

For the original implementation, experiments, and full algorithm description, see:

- [Original GitHub repository](https://github.com/gideonuchehara/GidNET-Qubit-Reuse-Algorithm)
- [GidNET preprint](https://arxiv.org/abs/2410.08817)
- [IEEE publication](https://ieeexplore.ieee.org/document/10821360)

## Installation

This fork has been used with **Qiskit 0.45.x** for the current experiments.

```bash
git clone https://github.com/atenaraeisi/GidNET-Qubit-Reuse-Algorithm.git
cd GidNET-Qubit-Reuse-Algorithm
pip install -r requirements.txt
```

Some experiment scripts also use Qiskit Aer, pandas, and NetworkX.

## Basic usage

```python
from qiskit import QuantumCircuit
from gidnet.qubitreuse import GidNET

circuit = QuantumCircuit(5)

circuit.cx(1, 2)
circuit.cx(0, 3)
circuit.cx(1, 4)
circuit.cx(2, 4)
circuit.cx(3, 4)

compiler = GidNET(circuit)
dynamic_circuit = compiler.compile_to_dynamic_circuit(
    iterations=20,
    draw=False,
)

print("Original width:", circuit.num_qubits)
print("Dynamic width:", dynamic_circuit.num_qubits)
print(dynamic_circuit.count_ops())
```

## Reset lowering

To lower native reset operations into measurement plus conditional-X:

```python
from gidnet.reset_lowering import lower_reset_to_measure_x

lowered = lower_reset_to_measure_x(dynamic_circuit)

print(dynamic_circuit.count_ops())
print(lowered.count_ops())
```

## Repository notes

The repository contains the original GidNET code and experiment material together with my fork-specific changes. The most relevant additions for my current work are:

```text
gidnet/
├── qubitreuse.py
└── reset_lowering.py

test_reset_lowering.py
test_reset_lowering_semantics.py
inspect_reset_noise.py
run_gidnet_on_all_95.py
```

Additional research code and experiments may be added as the project develops.

## Attribution

The **GidNET algorithm and original codebase are not my work**. They were developed by Gideon Uchehara and collaborators.

My contributions in this fork are limited to the fixes, validation code, diagnostics, and experimental tooling described above.

If you use GidNET in academic work, please cite the original authors and paper.

## License

This fork follows the license of the original GidNET repository.
