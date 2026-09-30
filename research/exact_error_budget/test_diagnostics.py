from __future__ import annotations
import ast,copy,json,math,unittest
from pathlib import Path
from diagnostics import *

def nodes(ids_times):return {i:{'node_id':i,'t':t,'z':0.,'y':0.,'x':float(i)} for i,t in ids_times}
def edges(pairs):return [{'source_id':a,'target_id':b} for a,b in pairs]
def run_case(pn,pe,gn,ge,mp,ev,expected,**kw):return partition_edges(pn,pe,gn,ge,mp,ev,expected,**kw)
class Tests(unittest.TestCase):
 def setUp(self):
  self.gn=nodes([(1,0),(2,1)]);self.ge=edges([(1,2)]);self.pn=nodes([(10,0),(20,1)]);self.mp={10:1,20:2}
 def case(self,mp=None,pn=None,pe=None,ev=None,expected=None,**kw):
  return run_case(self.pn if pn is None else pn,[] if pe is None else pe,self.gn,self.ge,self.mp if mp is None else mp,[] if ev is None else ev,expected or {'edge_tp':0,'edge_fp':0,'edge_fn':1},**kw)
 def test_free_endpoints(self):self.assertEqual(self.case()['fn_categories'],{'free_endpoint_fragmentation':1})
 def test_missing_source(self):self.assertEqual(self.case(mp={20:2})['fn_categories'],{'source_endpoint_unmatched':1})
 def test_missing_target(self):self.assertEqual(self.case(mp={10:1})['fn_categories'],{'target_endpoint_unmatched':1})
 def test_both_missing(self):self.assertEqual(self.case(mp={})['fn_categories'],{'both_endpoints_unmatched':1})
 def test_true_edge(self):
  r=self.case(pe=edges([(10,20)]),ev=[dict(source_id=10,target_id=20,matched=True,valid=True)],expected={'edge_tp':1,'edge_fp':0,'edge_fn':0});self.assertEqual(r['edge_tp'],1)
 def test_target_assigned_elsewhere(self):
  pn=dict(self.pn,**{}) ;pn[30]=nodes([(30,0)])[30]
  r=self.case(pn=pn,pe=edges([(30,20)]));self.assertEqual(r['fn_categories'],{'target_assigned_elsewhere':1})
 def test_source_continues_elsewhere(self):
  pn=dict(self.pn);pn[30]=nodes([(30,1)])[30]
  self.assertEqual(self.case(pn=pn,pe=edges([(10,30)]))['fn_categories'],{'source_continues_elsewhere':1})
 def test_source_capacity(self):
  pn=dict(self.pn);pn.update(nodes([(30,1),(40,1)]))
  self.assertEqual(self.case(pn=pn,pe=edges([(10,30),(10,40)]))['fn_categories'],{'source_at_two_child_capacity':1})
 def test_missing_second_daughter(self):
  gn=nodes([(1,0),(2,1),(3,1)]);ge=edges([(1,2),(1,3)]);pn=nodes([(10,0),(20,1),(30,1)])
  r=run_case(pn,edges([(10,30)]),gn,ge,{10:1,20:2,30:3},[dict(source_id=10,target_id=30,matched=True,valid=True)],{'edge_tp':1,'edge_fp':0,'edge_fn':1})
  self.assertEqual(r['fn_categories'],{'missing_second_daughter_edge':1})
 def test_unmatched_predictions_are_ignored(self):
  pn=nodes([(10,0),(20,1)])
  r=self.case(pn=pn,mp={},pe=edges([(10,20)]),ev=[dict(source_id=10,target_id=20,matched=False,valid=False)])
  self.assertEqual(r['edge_fp'],0);self.assertEqual(r['ignored_prediction_edges'],1)
 def test_invalid_sparse_fp_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'UNMATCHED_EDGE'):
   self.case(mp={},pe=edges([(10,20)]),ev=[dict(source_id=10,target_id=20,matched=False,valid=True)])
 def test_fp_source_only(self):
  r=self.case(mp={10:1},pe=edges([(10,20)]),ev=[dict(source_id=10,target_id=20,matched=False,valid=True)],expected={'edge_tp':0,'edge_fp':1,'edge_fn':1})
  self.assertEqual(r['fp_categories'],{'source_matched_target_unmatched':1})
 def test_fp_target_only(self):
  r=self.case(mp={20:2},pe=edges([(10,20)]),ev=[dict(source_id=10,target_id=20,matched=False,valid=True)],expected={'edge_tp':0,'edge_fp':1,'edge_fn':1})
  self.assertEqual(r['fp_categories'],{'source_unmatched_target_matched':1})
 def test_raw_match_removed(self):
  r=self.case(mp={},raw_to_gt={100:1,200:2},raw_nodes=nodes([(100,0),(200,1)]))
  self.assertEqual(r['unmatched_gt_node_categories'],{'raw_matched_node_removed_from_final':2})
 def test_raw_match_retained(self):
  r=self.case(mp={},raw_to_gt=self.mp,raw_nodes=self.pn)
  self.assertEqual(r['unmatched_gt_node_categories'],{'raw_matched_node_retained_but_final_matching_changed':2})
 def test_missing_raw_and_final(self):self.assertEqual(self.case(mp={})['unmatched_gt_node_categories'],{'unmatched_in_raw_and_final':2})
 def test_no_prediction_mutation(self):
  before=copy.deepcopy((self.pn,self.gn,self.ge,self.mp));self.case();self.assertEqual(before,(self.pn,self.gn,self.ge,self.mp))
 def test_nonunique_matching_rejected(self):
  pn=nodes([(10,0),(30,0)])
  with self.assertRaisesRegex(RuntimeError,'NONUNIQUE'):inverse_matching({10:1,30:1},pn,self.gn)
 def test_wrong_time_matching(self):
  with self.assertRaisesRegex(RuntimeError,'TIME_MISMATCH'):inverse_matching({20:1},self.pn,self.gn)
 def test_unknown_match(self):
  with self.assertRaisesRegex(RuntimeError,'UNKNOWN'):inverse_matching({999:1},self.pn,self.gn)
 def test_negative_match_sentinel(self):self.assertEqual(inverse_matching({10:-1},self.pn,self.gn),{})
 def test_count_disagreement_stops(self):
  with self.assertRaisesRegex(RuntimeError,'FN_ATTRIBUTION'):self.case(expected={'edge_tp':0,'edge_fp':0,'edge_fn':2})
 def test_duplicate_edge_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'DUPLICATE'):adjacency(self.pn,edges([(10,20),(10,20)]))
 def test_nonconsecutive_edge_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'NONCONSECUTIVE'):adjacency(self.pn,edges([(20,10)]))
 def test_scale(self):self.assertAlmostEqual(physical_distance({'z':1,'y':0,'x':0},{'z':0,'y':0,'x':0}),1.625)
 def test_division_recovered(self):self.assertEqual(self.reason(recovered=True),'recovered')
 def reason(self,**kw):
  d=dict(recovered=False,parent_support={1},daughter_support=[{2},{3}],local_forks={1},valid_topology_forks={1},invalid_forks=set());d.update(kw);return division_reason(**d)
 def test_division_parent(self):self.assertEqual(self.reason(parent_support=set()),'parent_window_unmatched')
 def test_division_daughter(self):self.assertEqual(self.reason(daughter_support=[{2},set()]),'daughter_window_unmatched')
 def test_division_no_fork(self):self.assertEqual(self.reason(local_forks=set()),'parent_and_daughters_present_no_local_fork')
 def test_division_invalid_fork(self):self.assertEqual(self.reason(invalid_forks={1}),'local_forks_invalid_by_component_or_merge_rules')
 def test_division_wrong_topology(self):self.assertEqual(self.reason(valid_topology_forks=set()),'local_forks_do_not_connect_both_daughters')
 def test_division_assignment(self):self.assertEqual(self.reason(),'valid_local_evidence_not_selected_by_event_matching')
 def test_actual_return_aggregate(self):
  rows=[x['metric'] for x in json.loads((Path(__file__).parent/'expected_baselines.json').read_text()).values()]
  self.assertAlmostEqual(scalar_score(rows),.9330456734289274,places=12)
 def test_sensitivity_not_a_measured_result(self):
  row={'edge_tp':8,'edge_fp':1,'edge_fn':1,'division_tp':1,'division_fp':0,'division_fn':1,'total_node_ratio':0.}
  result=sensitivity_table([{'metric':row,'edges':{'fn_categories':{'free_endpoint_fragmentation':1}}}])
  self.assertTrue(all(not x['achieved_result'] for x in result));self.assertEqual(row['edge_fn'],1)
 def test_perfect_division_score(self):
  self.assertAlmostEqual(scalar_score([dict(edge_tp=4,edge_fp=0,edge_fn=0,division_tp=1,division_fp=0,division_fn=0,total_node_ratio=0)]),1.1)
 def test_zero_divisions(self):
  self.assertAlmostEqual(scalar_score([dict(edge_tp=4,edge_fp=0,edge_fn=0,division_tp=0,division_fp=0,division_fn=0,total_node_ratio=0)]),1.)
 def test_invalid_counts(self):
  with self.assertRaisesRegex(RuntimeError,'INVALID_COUNT'):scalar_score([dict(edge_tp=1,edge_fp=-1,edge_fn=0,division_tp=0,division_fp=0,division_fn=0,total_node_ratio=0)])
 def test_notebook_sources_compile(self):
  from error_report import build_cells
  for smoke in (True,False):
   for c in build_cells(Path('/tmp/test'),smoke):
    if c.cell_type=='code':compile(c.source,'notebook-test','exec')

if __name__=='__main__':unittest.main(verbosity=2)
