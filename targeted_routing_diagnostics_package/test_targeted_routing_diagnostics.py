"""Pure-Python smoke checks runnable without Qiskit; full Qiskit test on user PC."""
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from targeted_routing_diagnostics import (
    parse_chains,match_chains,reposition_layout,
    read_input_plan_data,summarise_runs,write_csv)

class PureTests(unittest.TestCase):
    def test_parse_preserves_order(self):
        self.assertEqual(parse_chains('((0, 3), (1,), (2, 4, 6))'),((0,3),(1,),(2,4,6)))
        for x in ['((0, 1), (1,))','((0, True),)','[]']:
            with self.assertRaises(ValueError):parse_chains(x)
    def test_layout_matching_system9(self):
        a=parse_chains('((0, 3), (1, 7, 8), (2, 4, 6), (5,), (9,), (10,), (11,))')
        b=parse_chains('((0, 11), (1, 7, 8), (2, 4, 6), (3,), (5,), (9,), (10,))')
        m,score=match_chains(a,b)
        self.assertEqual(score,11)
        anchor=[23,31,40,11,14,53,76]
        counterpart=reposition_layout(anchor,m,'better')
        self.assertEqual(anchor,reposition_layout(counterpart,m,'worse'))
        self.assertEqual(counterpart[m[0]],anchor[0])
    def test_layout_matching_sym9(self):
        a=parse_chains('((0, 10), (1, 11), (2, 4, 6), (3, 5, 7, 8), (9,))')
        b=parse_chains('((0, 2, 4, 6), (1, 11), (3, 5, 7, 8), (9,), (10,))')
        m,score=match_chains(a,b)
        self.assertEqual(score,11)
        v=[12,22,44,66,88]
        self.assertEqual(reposition_layout(reposition_layout(v,m,'better'),m,'worse'),v)
    def test_data_from_real_csvs(self):
        data=Path('/mnt/data/GPT.zip')
        if not data.exists():self.skipTest('Source GPT.zip not mounted')
        with tempfile.TemporaryDirectory() as td:
            z=zipfile.ZipFile(data)
            for n in z.namelist():
                if n.endswith('.csv'):
                    f=Path(td)/Path(n).name;f.write_bytes(z.read(n))
            plans,old,paths=read_input_plan_data(td)
            self.assertEqual(plans['system9.qasm']['P5']['normalized_logical_depth'],'175')
            self.assertEqual(len(old[('sym9_146.qasm','P8')]),30)
    def test_summarise_pairing(self):
        rows=[{'circuit':'system9.qasm','mode':'free','seed':100,'plan_id':'P5',
               'routed_depth':300,'swap_count':100,'swap_critical_count':3,'swap_layers':30,
               'routed_overhead':125,'cx_critical_count':15,'max_consecutive_swap_layers':4},
              {'circuit':'system9.qasm','mode':'free','seed':100,'plan_id':'P4',
               'routed_depth':310,'swap_count':95,'swap_critical_count':6,'swap_layers':33,
               'routed_overhead':135,'cx_critical_count':17,'max_consecutive_swap_layers':7}]
        pair,sums=summarise_runs(rows)
        self.assertEqual(pair[0]['delta_routed_depth'],10)
        self.assertEqual(pair[0]['delta_swap_count'],-5)
        self.assertEqual(sums[0]['mean_delta_swap_critical_count'],3)
    def test_csv_writer(self):
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'output.csv'
            write_csv(f,[{'x':1,'y':2}])
            self.assertIn('x,y',f.read_text(encoding='utf-8-sig'))

if __name__=='__main__':unittest.main()
