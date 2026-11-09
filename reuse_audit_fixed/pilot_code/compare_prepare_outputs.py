#!/usr/bin/env python3
"""Check that a new prepare retained exactly the prior selected/discovered plans.

No Qiskit, no routing. Compare IDs, plans, step depths, parents, and operators.
Use only when BOTH prepare runs used the same inputs, seeds and budgets.
"""
import argparse,csv,json,sys
from pathlib import Path


def load(path, fields):
    with path.open(encoding='utf-8-sig',newline='') as f:
        rr=list(csv.DictReader(f))
    return sorted(tuple(row[k] for k in fields) for row in rr)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--old-dir',type=Path,required=True)
    ap.add_argument('--new-dir',type=Path,required=True)
    a=ap.parse_args()
    specs={
      'discovery.csv':('circuit_id','original_sig','sig','parent_sig','discovery_move_depth','operator'),
      'selected_candidates.csv':('circuit_id','original_sig','sig','parent_sig','discovery_move_depth','operator','is_original','active_width'),
      'plan_catalog.csv':('circuit_id','sig','active_width','plan_json'),
    }
    bad=0
    for file,fields in specs.items():
        try:
            l,r=load(a.old_dir/file,fields),load(a.new_dir/file,fields)
            ok=l==r
            print(f'{file}: {"PASS" if ok else "FAIL"} (old={len(l)}, new={len(r)})')
            if not ok:
                sl,sr=set(l),set(r)
                print('  legacy-only:',list(sorted(sl-sr))[:2])
                print('  fast-only:',list(sorted(sr-sl))[:2])
                bad+=1
        except Exception as e:
            print(f'{file}: ERROR {e}')
            bad+=1
    if bad:sys.exit(1)
    print('All candidate identity/path checks PASS (no Routing involved).')

if __name__=='__main__':main()
