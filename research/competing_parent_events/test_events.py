from __future__ import annotations
import unittest,copy,inspect,hashlib,json,tempfile
from pathlib import Path
from dataclasses import replace
import numpy as np
import division_events as e

def fixture():
 n={0:{'t':0,'z':0.,'y':0.,'x':0.},1:{'t':1,'z':0.,'y':0.,'x':2.},
    2:{'t':1,'z':0.,'y':0.,'x':5.},3:{'t':0,'z':0.,'y':0.,'x':5.},
    4:{'t':2,'z':0.,'y':0.,'x':3.},5:{'t':2,'z':0.,'y':0.,'x':6.}}
 edges=[{'source_id':0,'target_id':1},{'source_id':3,'target_id':2},
        {'source_id':1,'target_id':4},{'source_id':2,'target_id':5}]
 return n,edges

def window():return {'gt_divider_id':100,'parent_side_prediction_ids':[0],
                     'daughter_side_prediction_ids':[[1,4],[2,5]]}

class Tests(unittest.TestCase):
 def test_empty(self):
  ids,x,c=e.propose({},[]);self.assertEqual(ids.shape,(0,5));self.assertEqual(x.shape,(0,len(e.FEATURES)))
 def test_expected_feature_width(self):self.assertEqual(len(e.FEATURES),29)
 def test_competing_parent_proposal_present(self):
  n,es=fixture();ids,x,c=e.propose(n,es);self.assertIn([0,1,2,3,0],ids.tolist());self.assertGreater(c['competing_parent_proposals'],0)
 def test_no_annotations_in_api(self):
  self.assertEqual(list(inspect.signature(e.propose).parameters),['nodes','edges','config','heartbeat'])
 def test_features_finite(self):
  ids,x,c=e.propose(*fixture());self.assertTrue(np.isfinite(x).all())
 def test_inputs_not_mutated(self):
  n,es=fixture();old=copy.deepcopy((n,es));e.propose(n,es);self.assertEqual((n,es),old)
 def test_order_invariance(self):
  n,es=fixture();a=e.propose(n,es);b=e.propose(dict(reversed(list(n.items()))),list(reversed(es)))
  np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
 def test_translation_invariance(self):
  n,es=fixture();a=e.propose(n,es);n2=copy.deepcopy(n)
  for d in n2.values():
   for k in ('x','y','z'):d[k]+=100
  b=e.propose(n2,es);np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_allclose(a[1],b[1],atol=1e-6)
 def test_no_absolute_time_feature(self):
  n,es=fixture();a=e.propose(n,es)
  for d in n.values():d['t']+=100
  b=e.propose(n,es);np.testing.assert_array_equal(a[1],b[1])
 def test_missing_temporal_context_does_not_forbid_candidate(self):
  ids,x,_=e.propose(*fixture());i=ids.tolist().index([0,1,2,3,0]);self.assertEqual(x[i,e.FEATURES.index('parent_history')],0)
 def test_duplicate_rejected(self):
  n,es=fixture()
  with self.assertRaisesRegex(RuntimeError,'DUPLICATE'):e.structure(n,es+es[:1])
 def test_nan_rejected(self):
  n,es=fixture();n[0]['x']=float('nan')
  with self.assertRaisesRegex(RuntimeError,'NONFINITE'):e.structure(n,es)
 def test_merge_rejected(self):
  n,es=fixture()
  with self.assertRaisesRegex(RuntimeError,'MERGED'):e.structure(n,es+[{'source_id':3,'target_id':1}])
 def test_time_gap_rejected(self):
  n,es=fixture()
  with self.assertRaisesRegex(RuntimeError,'NONADJACENT'):e.structure(n,[{'source_id':0,'target_id':4}])
 def test_missing_node_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'MISSING'):e.structure(fixture()[0],[{'source_id':0,'target_id':999}])
 def test_event_count_limit(self):
  with self.assertRaisesRegex(RuntimeError,'COUNT_LIMIT'):e.propose(*fixture(),config=replace(e.CONFIG,max_events_per_movie=1))
 def test_radius_limit(self):
  n,es=fixture();ids,_,_=e.propose(n,es,config=replace(e.CONFIG,radius_um=.1));self.assertEqual(len(ids),0)
 def test_invalid_configuration(self):
  with self.assertRaisesRegex(RuntimeError,'CONFIG'):e.propose(*fixture(),config=replace(e.CONFIG,radius_um=-1))
 def test_label_positive(self):
  n,es=fixture();inc,out=e.structure(n,es);ids=np.array([[0,1,2,3,0]])
  y,event=e.labels_for_events(ids,inc,out,[window()],{0:100,1:101,2:102},{100:[101,102]},{100:0,101:0,102:0})
  self.assertEqual(y.tolist(),[1]);self.assertEqual(event.tolist(),[100])
 def test_unknown_ignored(self):
  n,es=fixture();inc,out=e.structure(n,es);y,_=e.labels_for_events(np.array([[0,1,2,3,0]]),inc,out,[],{},{},{})
  self.assertEqual(y.tolist(),[-1])
 def test_annotated_nondivision_negative(self):
  n,es=fixture();inc,out=e.structure(n,es);y,_=e.labels_for_events(np.array([[0,1,2,3,0]]),inc,out,[],{0:100},{100:[101]},{100:0})
  self.assertEqual(y.tolist(),[0])
 def test_conflicting_components_negative(self):
  n,es=fixture();inc,out=e.structure(n,es)
  y,_=e.labels_for_events(np.array([[0,1,2,3,0]]),inc,out,[window()],{1:101,2:102},{},{101:0,102:1})
  self.assertEqual(y.tolist(),[0])
 def test_daughter_distinctness(self):
  n,es=fixture();inc,out=e.structure(n,es);w=window();w['daughter_side_prediction_ids']=[[1,4],[1,4]]
  self.assertFalse(e.event_support([0,1,2,3,0],inc,out,w))
 def test_parent_anchor_required(self):
  n,es=fixture();inc,out=e.structure(n,es);w=window();w['parent_side_prediction_ids']=[99]
  self.assertFalse(e.event_support([0,1,2,3,0],inc,out,w))
 def test_transaction_reassigns_claimed_daughter(self):
  n,es=fixture();new,a=e.apply_events(n,es,np.array([[0,1,2,3,0]]),np.array([.99]),.9)
  self.assertEqual(a['accepted_new_forks'],1);self.assertEqual(len(es),len(new));self.assertNotIn((3,2),[(r['source_id'],r['target_id']) for r in new])
  self.assertIn((0,2),[(r['source_id'],r['target_id']) for r in new])
 def test_transaction_nodes_unchanged(self):
  n,es=fixture();old=copy.deepcopy(n);e.apply_events(n,es,np.array([[0,1,2,3,0]]),np.array([.99]),.9);self.assertEqual(n,old)
 def test_no_change_below_threshold(self):
  n,es=fixture();new,a=e.apply_events(n,es,np.array([[0,1,2,3,0]]),np.array([.1]),.9);self.assertEqual(a['accepted_new_forks'],0);self.assertEqual(len(new),len(es))
 def test_existing_fork_protected(self):
  n,es=fixture();es[1]={'source_id':0,'target_id':2};ids,x,c=e.propose(n,es)
  self.assertTrue(any(r[0]==0 and r[4]==1 for r in ids))
  new,a=e.apply_events(n,es,ids,np.ones(len(ids)),.5);inc,out=e.structure(n,new);self.assertEqual(out[0],[1,2])
 def test_conflicting_events_rejected(self):
  n,es=fixture();ids=np.array([[0,1,2,3,0],[3,2,1,0,0]]);new,a=e.apply_events(n,es,ids,np.array([.95,.96]),.9)
  self.assertEqual(a['accepted_new_forks'],1)
 def test_transaction_budget(self):
  new,a=e.apply_events(*fixture(),np.array([[0,1,2,3,0]]),np.array([.99]),.9,replace(e.CONFIG,maximum_new_forks_per_movie=0))
  self.assertEqual(a['accepted_new_forks'],0)
 def test_nonfinite_score_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'SCORE_INVALID'):e.apply_events(*fixture(),np.array([[0,1,2,3,0]]),np.array([np.nan]),.9)
 def test_insufficient_divisions_stops_fit(self):
  with self.assertRaisesRegex(RuntimeError,'INSUFFICIENT_DISTINCT'):e.fit_model(np.ones((30,29)),np.r_[1,np.zeros(29)],np.r_[0,np.full(29,-1)])
 def test_missing_negative_labels_stop_fit(self):
  with self.assertRaisesRegex(RuntimeError,'NEGATIVE'):e.fit_model(np.ones((2,29)),np.ones(2),np.arange(2))
 def test_fit_and_training_only_threshold(self):
  rng=np.random.default_rng(7);x=rng.normal(size=(100,29));y=np.r_[np.ones(6),np.zeros(94)];x[:6,0]+=6;events=np.r_[np.repeat([0,1],3),np.full(94,-1)]
  m=e.fit_model(x,y,events);self.assertTrue(m['optimizer_success']);self.assertEqual(m['training_fp_above_threshold'],0)
  np.testing.assert_allclose(m['mean'],x.mean(0));self.assertTrue(np.isfinite(e.predict(m,x)).all())
 def test_unknown_excluded_from_scaling(self):
  rng=np.random.default_rng(8);x=rng.normal(size=(102,29));x[100:]=1000;y=np.r_[1,1,np.zeros(98),-1,-1];events=np.r_[0,1,np.full(100,-1)]
  m=e.fit_model(x,y,events);np.testing.assert_allclose(m['mean'],x[:100].mean(0))
 def test_scores_not_label_dependent(self):self.assertEqual(list(inspect.signature(e.predict).parameters),['model','x'])
 def test_event_apply_has_no_label_parameter(self):self.assertNotIn('labels',inspect.signature(e.apply_events).parameters)
 def test_fit_nan_failure(self):
  x=np.zeros((30,29));x[0,0]=np.nan
  with self.assertRaisesRegex(RuntimeError,'BAD_TRAINING'):e.fit_model(x,np.r_[1,1,np.zeros(28)],np.r_[0,1,np.full(28,-1)])

if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests);r=unittest.TextTestRunner(verbosity=2).run(suite)
 print('COMPETING_PARENT_EVENT_TESTS',r.testsRun,flush=True)
 raise SystemExit(0 if r.wasSuccessful() else 1)
