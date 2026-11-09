"""Fast unit tests; no Qiskit or user data needed."""
import numpy as np
import pandas as pd
from train_pairwise_repair_v2 import (
    weighted_pairs, choose_pairwise, symmetric_win_prob, decision_row, summarize
)

class ToyModel:
    def predict_proba(self,x):
        # Clearly order by the first difference coordinate.
        p=1/(1+np.exp(-np.clip(x[:,0],-30,30)))
        return np.column_stack([1-p,p])

def test_pair_orientation_symmetry():
    x=np.array([[2.0,9.0], [1.0,7.0], [-2.0,4.0]],dtype=np.float32)
    xi=x[:,None,:]-x[None,:,:]
    av=0.5*(x[:,None,:]+x[None,:,:])
    pair=np.concatenate([xi,av],axis=-1).reshape(-1,4)
    pred=symmetric_win_prob(ToyModel(),pair).reshape(3,3)
    assert np.allclose(pred+pred.T,1)
    assert np.allclose(np.diag(pred),0.5)
    assert pred[0,2]>pred[2,0]

def test_train_pairs_reversed_and_within_group():
    d=pd.DataFrame({'circuit_id':['A','A','A','B','B'],
        'original_sig':['o1','o1','o1','o2','o2'],
        'family_group':['fam1','fam1','fam1','fam2','fam2'],
        'is_original':[1,0,0,1,0],
        'abs_gain':[0,3,-2,0,1]})
    x=np.array([[0,1],[3,1],[-2,1],[0,9],[1,9]],dtype=np.float32)
    xx,yy,w,info=weighted_pairs(d,x,margin=0.2,max_nonanchor=10,seed=123)
    assert info['originals_with_pairs']==2
    assert len(yy)%2==0 and len(yy)>=6
    assert np.isclose(w.mean(),1)
    assert yy[::2].tolist()==[1-v for v in yy[1::2]]
    # first half of x is difference; swapped orientation negates it
    assert np.allclose(xx[::2,:2],-xx[1::2,:2])
    # second half is context; the reverse pair retains it
    assert np.allclose(xx[::2,2:],xx[1::2,2:])

def test_gate_and_no_repair():
    ranking=pd.DataFrame([
        dict(objective='swap_count',circuit_id='A',original_sig='o',sig='o',
             is_original=1,baseline_mean=5,abs_gain=0,discovery_move_depth=0,
             borda=0.40,p_vs_original=0.50,delta_compiled_norm_depth=0),
        dict(objective='swap_count',circuit_id='A',original_sig='o',sig='a',
             is_original=0,baseline_mean=5,abs_gain=3,discovery_move_depth=2,
             borda=0.60,p_vs_original=0.70,delta_compiled_norm_depth=-1),
    ])
    assert choose_pairwise(ranking,0,0.65)['sig']=='a'
    assert choose_pairwise(ranking,0,0.75) is None
    ranking['baseline_mean']=0
    assert choose_pairwise(ranking,0,0.50) is None

if __name__=='__main__':
    test_pair_orientation_symmetry()
    test_train_pairs_reversed_and_within_group()
    test_gate_and_no_repair()
    print('[PASS] pair orientation/symmetry, within-original pairs, conservative gate')
