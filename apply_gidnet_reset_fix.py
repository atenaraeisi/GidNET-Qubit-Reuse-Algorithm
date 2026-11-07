"""Safely fix reset placement after a multi-qubit terminal gate in GidNET.

Run once from the repository root:
    python apply_gidnet_reset_fix.py

Target: gidnet/qubitreuse.py (research fork snapshot, Oct 2026).
This patch never touches experiment data. A backup is created first.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import sys

TARGET = Path(__file__).resolve().parent / "gidnet" / "qubitreuse.py"
BACKUP = TARGET.with_suffix(".py.before_reset_wire_fix.bak")
MARKER = "self.reset_after_node_qubits"

PATCHES = [
    (
        "        self.reset_after_nodes = []\n        self.circuit_dag_with_reuse_edges = None",
        "        self.reset_after_nodes = []\n"
        "        # Maps a terminal operation to the logical qubit(s) that end at it.\n"
        "        self.reset_after_node_qubits = defaultdict(set)\n"
        "        self.circuit_dag_with_reuse_edges = None",
    ),
    (
        "        reset_after_hashes = [n.__hash__() for n in self.reset_after_nodes]\n",
        "        # Reset targets are tracked separately from the operation's other qargs.\n",
    ),
    (
        "            if node.__hash__() in reset_after_hashes:\n"
        "                # This was the last operation on a qubit that is about to be\n"
        "                # reused by a different logical qubit -- insert an explicit\n"
        "                # reset so the transpiler sees a real physical boundary here\n"
        "                # (previously this branch was unreachable; see BUGFIX above).\n"
        "                reset_node = self._add_reset_op(new_qargs)\n"
        "                self.dynamic_circuit_dag.apply_operation_back(reset_node.op, qargs=reset_node.qargs)",
        "            # One final multi-qubit operation may end the lifetime of just\n"
        "            # one of its operands (or of several operands). Reset *only*\n"
        "            # the retired wire(s), once per reuse boundary.\n"
        "            for old_qubit in sorted(self.reset_after_node_qubits.get(node.__hash__(), ())):\n"
        "                physical_wire = old_to_new_qubit_remap_dict[old_qubit]\n"
        "                reset_qargs = (qubits_indices[physical_wire],)\n"
        "                reset_node = self._add_reset_op(reset_qargs)\n"
        "                self.dynamic_circuit_dag.apply_operation_back(\n"
        "                    reset_node.op, qargs=reset_node.qargs\n"
        "                )",
    ),
    (
        "        self.reset_after_nodes = []  # Actual last op-node on each qubit being reused (for M/R insertion)",
        "        self.reset_after_nodes = []  # Actual last op-node on each reused qubit\n"
        "        self.reset_after_node_qubits = defaultdict(set)  # op-node hash -> retired old qubits",
    ),
    (
        "                    if last_op_node is not None:\n"
        "                        self.reset_after_nodes.append(last_op_node)",
        "                    if last_op_node is not None:\n"
        "                        self.reset_after_nodes.append(last_op_node)\n"
        "                        self.reset_after_node_qubits[last_op_node.__hash__()].add(root)",
    ),
]


def main():
    if not TARGET.is_file():
        raise SystemExit(f"Cannot find {TARGET}; place this script in the repository root.")
    original = TARGET.read_text(encoding="utf-8")
    if MARKER in original:
        print("This exact fix seems to be installed already; no changes made.")
        return

    updated = original
    for index, (before, after) in enumerate(PATCHES, 1):
        occurrences = updated.count(before)
        if occurrences != 1:
            raise SystemExit(
                f"Patch stopped safely: expected exactly one match for segment {index}, "
                f"found {occurrences}. File not changed."
            )
        updated = updated.replace(before, after, 1)

    compile(updated, str(TARGET), "exec")  # Syntax check before touching original.
    if BACKUP.exists():
        raise SystemExit(f"Backup {BACKUP} already exists. File not changed; inspect backup first.")
    shutil.copy2(TARGET, BACKUP)
    TARGET.write_text(updated, encoding="utf-8")
    print(f"Applied fix to: {TARGET}")
    print(f"Saved original backup: {BACKUP}")
    print("Now run: python -m pytest -q test_semantic_correctness.py test_reset_boundary_regression.py")


if __name__ == "__main__":
    main()
