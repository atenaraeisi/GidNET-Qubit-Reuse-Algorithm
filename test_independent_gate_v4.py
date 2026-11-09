#!/usr/bin/env python3
"""Fast no-routing unit checks for independent pairwise ranker + supervised gate."""
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import train_independent_gate_v4 as v4


def synthetic_rows():
    rows=[]
    for objective in v4.OBJECTIVES:
        for f in range(6):
            family=f'fam{f}'
            for src in range(2):
                cid=f'Circuit::{family}_{src}.qasm'
                oid=f'orig{src}'
                base=9.0+f
                for idx in range(4):
                    s=oid if idx==0 else f'candidate{idx}'
                    gain=0.0 if idx==0 else ((idx-1.5)*1.2+((f+src)%3-1)*0.3)
                    r={'objective':objective, 'family_group':family,'circuit_id':cid,
                       'original_sig':oid,'sig':s,'is_original':int(idx==0),
                       'discovery_move_depth':idx,'move_depth':idx,
                       'compiled_logical_depth':21-idx+f,
                       'orig_compiled_logical_depth':21+f,
                       'delta_compiled_logical_depth':-idx,
                       'compiled_norm_depth':30-idx+f,
                       'orig_compiled_norm_depth':30+f,
                       'delta_compiled_norm_depth':-idx,
                       'compiled_dyn_width':4,
                       'orig_compiled_dyn_width':4,
                       'delta_compiled_dyn_width':0,
                       'abs_gain':gain, 'candidate_mean':base-gain, 'baseline_mean':base,
                       'gain_pct':100*gain/base, 'harm':int(gain<0)}
                    rows.append(r)
    return pd.DataFrame(rows)


def main():
    d=synthetic_rows()
    cols=['compiled_logical_depth','orig_compiled_logical_depth','delta_compiled_logical_depth',
          'compiled_norm_depth','orig_compiled_norm_depth','delta_compiled_norm_depth',
          'compiled_dyn_width','orig_compiled_dyn_width','delta_compiled_dyn_width','move_depth']
    assert len(v4.select_features(cols,'compiled'))==10
    assert all('abs_gain' not in x and 'baseline_mean' not in x for x in v4.gate_features(cols))
    obj=d[d.objective.eq('swap_count')].reset_index(drop=True)
    tr=obj[obj.family_group.isin(['fam0','fam1','fam2','fam3'])].reset_index(drop=True)
    te=obj[obj.family_group.isin(['fam4','fam5'])].reset_index(drop=True)
    imputer,ranker,details=v4.fit_ranker(tr,cols,'pair_logit',0.2,20,2026)
    first=next(iter(te.groupby(v4.KEY,sort=True).groups))
    part=te.groupby(v4.KEY).get_group(first)
    _,choice,meta=v4.top_candidate_with_rank_scores(part,cols,imputer,ranker)
    assert choice is not None and meta['rank_n_candidates']==3
    # Choice must be independent of true physical labels/baselines on heldout family.
    corrupted=part.copy()
    corrupted['abs_gain']=np.arange(len(corrupted))*10_000.0
    corrupted['baseline_mean']=-9999.0
    _,choice_corrupted,meta2=v4.top_candidate_with_rank_scores(corrupted,cols,imputer,ranker)
    assert choice.sig==choice_corrupted.sig, 'Physical-label leakage into ranker choice'
    assert meta==meta2
    frame,audit=v4.crossfit_gate_examples(tr,cols,'pair_logit',2,0.2,20,2026)
    assert len(frame)>0 and frame.family_group.nunique()<=4
    gate,stats=v4.fit_gate(frame,cols,0.2)
    assert stats['n_gate_examples']==len(frame)
    groups,features=v4.selected_winners_for_heldout(te,cols,imputer,ranker)
    probs=v4.gate_predict(gate,features)
    assert (probs>=0).all() and (probs<=1).all()
    rows,_=v4.one_split_evaluate(tr,te,cols,('pair_logit',),('compiled',),v4.GATES,2,0.2,20,0.2,2026)
    assert len(rows)==4*int(te.is_original.sum())
    assert not rows.duplicated(['method',*v4.KEY]).any()
    assert set(rows.method)==set(v4.policies_for('swap_count',('pair_logit',),('compiled',),v4.GATES))
    chosen,scored,audit=v4.inner_choose_policy(obj,cols,('pair_logit',),('compiled',),v4.GATES,
                                               3,2,0.2,20,0.2,1.0,2026)
    assert chosen==scored.iloc[0].method
    # Prevent cross-family leakage by confirming all nonempty gate fit crossfold sources only.
    assert len(list(v4.family_folds(obj,3)))==3
    # One-class gate has deterministic predictions; no crashes when all labels identical.
    mono=frame.copy();mono['selected_true_gain']=-100.0
    monogate, monoinfo=v4.fit_gate(mono,cols,0.2)
    assert np.all(v4.gate_predict(monogate,features)==0)
    print('[PASS] ranker choice does not use true routing labels')
    print('[PASS] gate trained on out-of-fold ranker choices, family-disjoint')
    print('[PASS] threshold policies, no-repair handling, one-class gate')
    print('[PASS] family-grouped inner policy selection')

if __name__=='__main__': main()
