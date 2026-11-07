"""Fast smoke/regression tests for Stage A, no Qiskit needed."""
from pathlib import Path
import tempfile
import unittest
import controlled_pair_forensics as cpf
import structural_forensics as sf


class Tests(unittest.TestCase):
    def test_reuse_edge_diff_and_chain_alignment(self):
        a=((0,3),(1,),(2,))
        b=((2,),(0,1),(3,))
        self.assertEqual(cpf.edges_from_chains(a),{(0,3)})
        self.assertEqual(cpf.edges_from_chains(b),{(0,1)})
        perm,_=cpf.aligned_chains(a,b)
        self.assertEqual(set(perm.values()),{0,1,2})

    def test_op_dag_changes_at_boundary_and_is_acyclic(self):
        ops=[sf.Operation(0,'h',(0,)),sf.Operation(1,'x',(1,)),sf.Operation(2,'cx',(1,2)),sf.Operation(3,'x',(2,))]
        # Safe independent links q0->q1 and q0->q2.
        a=cpf.all_operation_dependencies(ops,((0,1),(2,)))
        b=cpf.all_operation_dependencies(ops,((0,2),(1,)))
        self.assertEqual(len(a['level']),4)
        self.assertEqual(a['level'][1],2)
        self.assertEqual(b['level'][3],3)
        self.assertEqual(a['depth'],4)
        self.assertEqual(b['depth'],3)

    def test_wrong_cycle_rejected(self):
        ops=[sf.Operation(0,'h',(0,)),sf.Operation(1,'cx',(0,1)),sf.Operation(2,'x',(1,))]
        with self.assertRaises(ValueError):
            cpf.all_operation_dependencies(ops,((1,0),))

if __name__=='__main__':unittest.main()
