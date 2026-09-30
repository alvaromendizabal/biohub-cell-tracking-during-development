from __future__ import annotations
import unittest,copy,tempfile,math,json,os,sys,gzip
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import appearance_edges as ae


def simple_tracks(n=3,frames=8,cross=False):
    nodes={};edges=[]
    for t in range(frames):
        for c in range(n):
            i=t*n+c;nodes[i]={'node_id':i,'t':t,'z':4.,'y':5.+c*8.,'x':4.+t}
            if t:edges.append({'source_id':(t-1)*n+c,'target_id':i})
    if cross:
        for e in edges:
            if e['source_id']==3*n:e['target_id']=4*n+1
            elif e['source_id']==3*n+1:e['target_id']=4*n
    return nodes,edges

class GeometryTests(unittest.TestCase):
 def test_feature_shapes(self):
  n,e=simple_tracks();p,x,i,c=ae.candidates(n,e);self.assertEqual(x.shape,(len(p),20));self.assertEqual(i.sum(),len(e));self.assertGreater(len(p),len(e))
 def test_finite(self):
  n,e=simple_tracks();p,x,i,c=ae.candidates(n,e);self.assertTrue(np.isfinite(x).all())
 def test_label_free_api(self):
  import inspect
  self.assertNotIn('labels',inspect.signature(ae.candidates).parameters);self.assertNotIn('gt',inspect.signature(ae.frame_profiles).parameters)
 def test_deterministic_order(self):
  n,e=simple_tracks();a=ae.candidates(n,e);b=ae.candidates(dict(reversed(list(n.items()))),list(reversed(e)))
  for i in range(3):np.testing.assert_array_equal(a[i],b[i])
 def test_coordinate_translation_invariance(self):
  n,e=simple_tracks();m=copy.deepcopy(n)
  for v in m.values():v['x']+=100;v['y']+=200;v['z']+=50
  a,b=ae.candidates(n,e),ae.candidates(m,e)
  np.testing.assert_allclose(a[1],b[1],atol=1e-5)
 def test_input_unchanged(self):
  n,e=simple_tracks();old=copy.deepcopy((n,e));ae.candidates(n,e);self.assertEqual((n,e),old)
 def test_long_incumbent_kept(self):
  n,e=simple_tracks(1);n[4]['x']+=100
  p,x,i,c=ae.candidates(n,e);self.assertEqual(int(i.sum()),len(e))
 def test_division_locked(self):
  n,e=simple_tracks();e=[v for v in e if v['target_id']!=13];e.append({'source_id':9,'target_id':13})
  locked=ae.protected_nodes(n,e);p,x,i,c=ae.candidates(n,e)
  self.assertIn(9,locked);self.assertTrue(all(a not in locked and b not in locked for a,b,t in p))
 def test_no_merge(self):
  n,e=simple_tracks();e.append({'source_id':1,'target_id':3})
  with self.assertRaises(RuntimeError):ae.candidates(n,e)
 def test_candidate_bound(self):
  n,e=simple_tracks()
  with self.assertRaises(RuntimeError):ae.candidates(n,e,replace(ae.CONFIG,max_pairs_per_movie=1))
 def test_empty(self):
  p,x,i,c=ae.candidates({},[]);self.assertEqual(p.shape,(0,3))

class AppearanceTests(unittest.TestCase):
 def frame(self):
  z,y,x=np.indices((16,32,32));return (100+1000*np.exp(-((z-8)**2+(y-16)**2+(x-16)**2)/12)).astype(np.float32)
 def test_profile_shape(self):
  p,s=ae.frame_profiles(self.frame(),[[8,16,16],[7,15,15]]);self.assertEqual(p.shape,(2,125));self.assertEqual(s.shape,(2,6))
 def test_flat_finite(self):
  p,s=ae.frame_profiles(np.ones((8,16,16)),[[4,8,8]]);self.assertTrue(np.isfinite(p).all());self.assertEqual(float(p.sum()),0)
 def test_boundary_recorded(self):
  p,s=ae.frame_profiles(self.frame(),[[0,0,0],[8,16,16]]);self.assertLess(s[0,5],s[1,5])
 def test_nonfinite_rejected(self):
  a=self.frame();a[0,0,0]=np.nan
  with self.assertRaises(RuntimeError):ae.frame_profiles(a,[[8,16,16]])
 def test_bad_scale_rejected(self):
  with self.assertRaises(RuntimeError):ae.frame_profiles(self.frame(),[[8,16,16]],scale=[1,0,1])
 def test_identity_pair(self):
  p,s=ae.frame_profiles(self.frame(),[[8,16,16]]);a=ae.appearance_features(np.array([[9,9,0]]),np.array([9]),p,s)
  self.assertAlmostEqual(float(a[0,0]),1,places=6);self.assertEqual(float(a[0,1]),0)
 def test_missing_endpoint(self):
  p,s=ae.frame_profiles(self.frame(),[[8,16,16]])
  with self.assertRaises(RuntimeError):ae.appearance_features(np.array([[0,99,0]]),np.array([0]),p,s)
 def test_order_match(self):
  p,s=ae.frame_profiles(self.frame(),[[8,16,16],[4,9,9]]);pairs=np.array([[1,2,0]])
  a=ae.appearance_features(pairs,np.array([1,2]),p,s);b=ae.appearance_features(pairs,np.array([2,1]),p[::-1],s[::-1]);np.testing.assert_array_equal(a,b)
 def test_intensity_linear_invariance(self):
  f=self.frame();p,s=ae.frame_profiles(f,[[8,16,16]]);q,t=ae.frame_profiles(f*3+20,[[8,16,16]]);np.testing.assert_allclose(p,q,atol=2e-6)
 def test_physical_stencil(self):self.assertEqual(ae.OFFSETS.shape,(125,3));self.assertEqual(set(ae.OFFSETS[:,0]),{-3,-1.5,0,1.5,3})
 def test_duplicate_node_id(self):
  with self.assertRaises(RuntimeError):ae.appearance_features(np.array([[1,1,0]]),np.array([1,1]),np.zeros((2,125)),np.zeros((2,6)))

class LabelTests(unittest.TestCase):
 def test_true_edge(self):self.assertEqual(ae.sparse_labels([[1,2,0]],{1:10,2:20},{(10,20)}).tolist(),[1])
 def test_known_source_wrong_target(self):self.assertEqual(ae.sparse_labels([[1,3,0]],{1:10},{(10,20)}).tolist(),[0])
 def test_known_target_wrong_source(self):self.assertEqual(ae.sparse_labels([[3,2,0]],{2:20},{(10,20)}).tolist(),[0])
 def test_unknown_ignored(self):self.assertEqual(ae.sparse_labels([[3,4,0]],{1:10},{(10,20)}).tolist(),[-1])
 def test_unannotated_terminal_ignored(self):self.assertEqual(ae.sparse_labels([[2,3,0]],{2:20},{(10,20)}).tolist(),[-1])

class DecodeTests(unittest.TestCase):
 def fixture(self):
  n={0:dict(t=0,z=0,y=0,x=0),1:dict(t=0,z=0,y=0,x=1),2:dict(t=1,z=0,y=0,x=0),3:dict(t=1,z=0,y=0,x=1)}
  e=[dict(source_id=0,target_id=2),dict(source_id=1,target_id=3)]
  pairs=np.array([[0,2,0],[0,3,0],[1,2,0],[1,3,0]])
  return n,e,pairs,np.array([True,False,False,True])
 def test_real_assignment_exchange(self):
  n,e,p,i=self.fixture();out,r=ae.decode(n,e,p,np.array([.1,.9,.9,.1]),i)
  self.assertEqual(r['changed_edges'],2);self.assertEqual({(e['source_id'],e['target_id']) for e in out},{(0,3),(1,2)})
 def test_incumbent_bonus_abstains(self):
  n,e,p,i=self.fixture();out,r=ae.decode(n,e,p,np.array([.5,.51,.51,.5]),i);self.assertEqual(r['changed_edges'],0)
 def test_no_uncalibrated_weak_edges(self):
  n,e,p,i=self.fixture();out,r=ae.decode(n,e,p,np.array([.01,.3,.3,.01]),i);self.assertEqual(r['changed_edges'],0)
 def test_missing_incumbent(self):
  n,e,p,i=self.fixture()
  with self.assertRaises(RuntimeError):ae.decode(n,e,p[1:],np.array([.9,.9,.1]),i[1:])
 def test_invalid_scores(self):
  n,e,p,i=self.fixture()
  with self.assertRaises(RuntimeError):ae.decode(n,e,p,np.array([.1,np.nan,.9,.1]),i)
 def test_input_immutable(self):
  n,e,p,i=self.fixture();old=copy.deepcopy((n,e));ae.decode(n,e,p,np.array([.1,.9,.9,.1]),i);self.assertEqual((n,e),old)
 def test_global_three_cycle(self):
  n,e=simple_tracks(3,2);p=np.array([[a,3+b,0] for a in range(3) for b in range(3)]);inc=p[:,0]+3==p[:,1]
  scores=np.array([.95 if b==(a+1)%3 else .1 for a in range(3) for b in range(3)])
  out,r=ae.decode(n,e,p,scores,inc);self.assertEqual(r['changed_edges'],3);self.assertEqual(r['accepted_cycles'],1)
 def test_no_new_nodes_or_degrees(self):
  n,e,p,i=self.fixture();out,r=ae.decode(n,e,p,np.array([.1,.9,.9,.1]),i)
  oldin,oldout=ae.structure(n,e);newin,newout=ae.structure(n,out);self.assertEqual(set(oldin),set(newin));self.assertEqual(len(out),len(e))
 def test_pair_order_invariance(self):
  n,e,p,i=self.fixture();s=np.array([.1,.9,.9,.1]);out,r=ae.decode(n,e,p,s,i);q=[3,2,1,0];out2,r2=ae.decode(n,e,p[q],s[q],i[q]);self.assertEqual(out,out2)
 def test_wrong_mask(self):
  n,e,p,i=self.fixture()
  with self.assertRaises(RuntimeError):ae.decode(n,e,p,np.array([.1,.9,.9,.1]),~i)

class ModelTests(unittest.TestCase):
 def test_four_runs_configuration_fixed(self):self.assertEqual(ae.CONFIG.max_trees,100);self.assertEqual(ae.CONFIG.max_fit_seconds,90)
 def test_insufficient_labels(self):
  with self.assertRaises(RuntimeError):ae.train(np.zeros((10,20)),np.zeros(10))
 def test_real_histogram_model_and_roundtrip(self):
  import pickle
  rng=np.random.default_rng(9);x=rng.normal(size=(800,4)).astype(np.float32);y=(x[:,0]>.3).astype(np.int8)
  m,r=ae.train(x,y);self.assertEqual(r['boosting_iterations'],100);self.assertFalse(m.early_stopping)
  n=pickle.loads(pickle.dumps(m));np.testing.assert_array_equal(m.predict_proba(x),n.predict_proba(x))
 def test_no_unknown_labels(self):
  rng=np.random.default_rng(10);x=rng.normal(size=(800,4));y=(x[:,0]>.4).astype(np.int8);y[:30]=-1
  m,r=ae.train(x,y);self.assertEqual(r['positive_edges']+r['negative_edges'],770)
 def test_bad_features(self):
  x=np.ones((800,4));x[0,0]=np.nan
  with self.assertRaises(RuntimeError):ae.train(x,np.r_[np.zeros(400),np.ones(400)])

class WorkerCacheTests(unittest.TestCase):
 def test_cache_roundtrip_and_corruption(self):
  import association_worker as w
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);w.np_save(p/'a.npz',x=np.arange(4));w.seal(p,'c',['a.npz']);self.assertTrue(w.reused(p,'c',['a.npz']))
   (p/'a.npz').write_text('broken')
   with self.assertRaises(RuntimeError):w.reused(p,'c',['a.npz'])
 def test_contract_change(self):
  import association_worker as w
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);w.save(p/'x.json',{});w.seal(p,'first',['x.json'])
   with self.assertRaises(RuntimeError):w.reused(p,'other',['x.json'])
 def test_actual_return_rejection_arithmetic(self):
  import association_worker as w
  r=w.summary([dict(edge_tp=2391,edge_fp=111,edge_fn=112,division_tp=1,division_fp=1,division_fn=5,adj_edge_jaccard=.9187599591432132)])
  self.assertAlmostEqual(r['score'],.9330456734289274,places=12)
 def test_report_code_compiles(self):
  import association_report as r
  for smoke in (True,False):
   for c in r.build_cells(Path('/tmp/test'),smoke):
    if c.cell_type=='code':compile(c.source,'notebook','exec')
 def test_feature_names_exclude_identifiers(self):
  for name in (*ae.GEO_NAMES,*ae.APP_NAMES):self.assertNotIn(name,('t','x','y','z','embryo','node_id','dataset'))

class ImageCacheTests(unittest.TestCase):
 def fixture(self,tmp):
  import association_worker as w
  root=Path(tmp);stem='movie';keys=['train/movie.zarr/zarr.json','train/movie.zarr/0/zarr.json','train/movie.zarr/0/c/0/0/0/0']
  entries={}
  for key in keys:
   p=root/key;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'cache bytes '+key.encode())
   entries[key]={'status':'complete','bytes':p.stat().st_size,'sha256':w.sha(p)}
  state={'plans':{stem:{'train/movie.zarr/0':{'shape':[1,2,2,2],'keys':['c/0/0/0/0']}}},'entries':entries}
  return w,root,state,stem,keys
 def test_real_file_checksums(self):
  with tempfile.TemporaryDirectory() as d:
   w,root,state,stem,keys=self.fixture(d)
   with patch.object(w,'CACHE',root):
    h,p=w.image_contract(state,stem);self.assertEqual(len(h),64)
 def test_corrupt_chunk_blocks_before_extraction(self):
  with tempfile.TemporaryDirectory() as d:
   w,root,state,stem,keys=self.fixture(d);(root/keys[-1]).write_bytes(b'bad')
   with patch.object(w,'CACHE',root):
    with self.assertRaises(RuntimeError):w.image_contract(state,stem)
 def test_receipted_fill_value_only(self):
  with tempfile.TemporaryDirectory() as d:
   w,root,state,stem,keys=self.fixture(d);(root/keys[-1]).unlink();state['entries'][keys[-1]]={'status':'absent','meaning':'zarr_chunk_absent_fill_value'}
   with patch.object(w,'CACHE',root):
    h,p=w.image_contract(state,stem);self.assertEqual(len(h),64)

if __name__=='__main__':unittest.main(verbosity=2)
