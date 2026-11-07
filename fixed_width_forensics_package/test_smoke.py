"""Fast smoke checks against the supplied ZIP; run from extracted work dir."""
from pathlib import Path
import numpy as np
import pandas as pd
from structural_forensics import parse_qasm, parse_chains, compute_features, read_inputs, audit_data

def test_zip_reconstruction():
    p=Path(__file__).parent/'GPT'
    d=read_inputs(p);audit_data(d)
    n,ops=parse_qasm(p/'system9.qasm')
    source=d['fresh'].query('circuit == "system9.qasm"')
    existing=d['features'].query('circuit == "system9.qasm"').set_index('plan_id')
    assert len(source)==13
    for row in source.itertuples(index=False):
        f,_,_=compute_features(ops,parse_chains(row.plan_signature),n,row.normalized_logical_depth)
        ref=existing.loc[row.plan_id]
        assert f['max_weighted_degree_rebuilt']==ref.max_weighted_degree
        assert f['quotient_unique_edges_rebuilt']==ref.quotient_unique_edges
        assert np.isclose(f['weighted_degree_cv_rebuilt'],ref.weighted_degree_cv)
    a=source.sort_values('fresh_mean_routed_depth')
    assert a.iloc[0].plan_id == 'P11' and a.iloc[-1].plan_id == 'P1'
    print('PASS: 13 plans, quotient features match Experiment 2, best/worst identities verified.')

if __name__=='__main__':test_zip_reconstruction()
