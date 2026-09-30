from __future__ import annotations
import copy,inspect,json,math,random,sys,unittest
from dataclasses import replace
from temporal_reassignment import CONFIG,SCALE,reassign,validate

def fixture(switched=True,offset=0):
    nodes={};edges=[]
    for track in (0,1):
        for t in range(6):
            i=offset+track*10+t
            x=(t-2) if track==0 else 4-t
            y=track*2
            nodes[i]={'node_id':i,'t':t,'z':0.,'y':y/SCALE[1],'x':(x+10)/SCALE[2]}
        for t in range(5):
            a=offset+track*10+t;b=a+1
            if switched and t==2:b=offset+(1-track)*10+t+1
            edges.append({'source_id':a,'target_id':b,'edge_prob':.2})
    return nodes,edges,copy.deepcopy(nodes)

def pairs(e):return {(x['source_id'],x['target_id']) for x in e}

class Tests(unittest.TestCase):
 def test_corrects_crossing(self):
    n,e,r=fixture();o,a=reassign(n,e,r);self.assertEqual(a['accepted_swaps'],1);self.assertIn((2,3),pairs(o));self.assertIn((12,13),pairs(o))
 def test_keeps_correct_links(self):
    n,e,r=fixture(False);o,a=reassign(n,e,r);self.assertEqual(pairs(e),pairs(o));self.assertEqual(a['accepted_swaps'],0)
 def test_no_mutation(self):
    n,e,r=fixture();before=copy.deepcopy((n,e,r));reassign(n,e,r);self.assertEqual((n,e,r),before)
 def test_degrees_preserved(self):
    n,e,r=fixture();o,a=reassign(n,e,r);before=validate(n,e);after=validate(n,o)
    for b,c in zip(before,after):self.assertEqual({i:len(x) for i,x in b.items()},{i:len(x) for i,x in c.items()})
 def test_missing_raw_context_skips(self):
    n,e,r=fixture();del r[0];o,a=reassign(n,e,r);self.assertEqual(a['accepted_swaps'],0)
 def test_unsmoothed_coordinates_drive_decision(self):
    n,e,r=fixture()
    for d in n.values():d['x']+=.07
    o,a=reassign(n,e,r);self.assertEqual(a['accepted_swaps'],1)
 def test_time_mismatch_raw_skips(self):
    n,e,r=fixture();r[0]['t']=20;o,a=reassign(n,e,r);self.assertEqual(a['accepted_swaps'],0)
 def test_nan_nodes_rejected(self):
    n,e,r=fixture();n[0]['x']=float('nan')
    with self.assertRaises(ValueError):reassign(n,e,r)
 def test_nan_raw_rejected(self):
    n,e,r=fixture();r[0]['x']=float('nan')
    with self.assertRaises(ValueError):reassign(n,e,r)
 def test_duplicate_rejected(self):
    n,e,r=fixture();e.append(e[0].copy())
    with self.assertRaises(ValueError):reassign(n,e,r)
 def test_missing_endpoint_rejected(self):
    n,e,r=fixture();del n[0]
    with self.assertRaises(ValueError):reassign(n,e,r)
 def test_nonconsecutive_rejected(self):
    n,e,r=fixture();e[0]['target_id']=2
    with self.assertRaises(ValueError):reassign(n,e,r)
 def test_node_id_rejected(self):
    n,e,r=fixture();n[0]['node_id']=99
    with self.assertRaises(ValueError):reassign(n,e,r)
 def test_negative_time_rejected(self):
    n,e,r=fixture();n[0]['t']=-1
    with self.assertRaises(ValueError):reassign(n,e,r)
 def test_untrusted_kwargs_no_gt(self):
    params=inspect.signature(reassign).parameters
    self.assertEqual(list(params),['nodes','edges','raw_nodes','config'])
 def test_empty_graph(self):self.assertEqual(reassign({},[],{})[1]['accepted_swaps'],0)
 def test_context_acceleration_gate(self):
    n,e,r=fixture();r[0]['x']+=10/SCALE[2];self.assertEqual(reassign(n,e,r)[1]['accepted_swaps'],0)
 def test_total_gain_gate(self):
    n,e,r=fixture();self.assertEqual(reassign(n,e,r,replace(CONFIG,min_total_gain_um=10))[1]['accepted_swaps'],0)
 def test_individual_gain_gate(self):
    n,e,r=fixture();self.assertEqual(reassign(n,e,r,replace(CONFIG,min_each_link_gain_um=10))[1]['accepted_swaps'],0)
 def test_distance_gate(self):
    n,e,r=fixture();self.assertEqual(reassign(n,e,r,replace(CONFIG,max_new_step_um=.1))[1]['accepted_swaps'],0)
 def test_search_radius_gate(self):
    n,e,r=fixture();self.assertEqual(reassign(n,e,r,replace(CONFIG,source_neighbor_radius_um=.1))[1]['accepted_swaps'],0)
 def test_invalid_config(self):
    n,e,r=fixture()
    for c in [replace(CONFIG,max_cost_ratio=1),replace(CONFIG,max_new_step_um=-1),replace(CONFIG,max_seconds=float('nan'))]:
     with self.assertRaises(ValueError):reassign(n,e,r,c)
 def test_division_context_protected(self):
    n,e,r=fixture();n[99]=dict(n[1],node_id=99);r[99]=dict(n[99]);e.append({'source_id':0,'target_id':99})
    o,a=reassign(n,e,r);self.assertEqual(a['accepted_swaps'],0);self.assertEqual(pairs(o),pairs(e))
 def test_merge_rejected(self):
    n,e,r=fixture();e.append({'source_id':10,'target_id':1})
    with self.assertRaises(ValueError):reassign(n,e,r)
 def test_order_determinism(self):
    n,e,r=fixture();o,a=reassign(n,e,r);random.Random(1).shuffle(e);n=dict(reversed(list(n.items())))
    oo,aa=reassign(n,e,r);self.assertEqual(o,oo);self.assertEqual(a['swaps'],aa['swaps'])
 def test_repeat_does_not_reverse_correction(self):
    n,e,r=fixture();o,a=reassign(n,e,r);oo,aa=reassign(n,o,r);self.assertEqual(pairs(o),pairs(oo))
 def test_no_fabricated_learned_probability(self):
    n,e,r=fixture();o,a=reassign(n,e,r)
    for row in o:
     if row.get('bidirectional_temporal_reassignment'):self.assertIsNone(row['edge_prob'])
 def test_invariant_edge_count(self):
    n,e,r=fixture();o,a=reassign(n,e,r);self.assertEqual(len(o),len(e))
 def test_static_disjoint_chains_no_intervention(self):
    n,e,r=fixture(False)
    for i,x in r.items():x['x']=float(i//10)*50
    self.assertEqual(reassign(n,e,r)[1]['accepted_swaps'],0)
 def test_large_fixture_bounded(self):
    n={};e=[];r={}
    for k in range(100):
     nn,ee,rr=fixture(True,100*k)
     for d in nn.values():d['z']=k*25
     for d in rr.values():d['z']=k*25
     n.update(nn);e+=ee;r.update(rr)
    o,a=reassign(n,e,r);self.assertEqual(a['accepted_swaps'],100)
 def test_two_time_directions_required(self):
    n,e,r=fixture();r[5]['x']+=6/SCALE[2];r[4]['x']+=3/SCALE[2]
    self.assertEqual(reassign(n,e,r)[1]['accepted_swaps'],0)

if __name__=='__main__':unittest.main(verbosity=2)
