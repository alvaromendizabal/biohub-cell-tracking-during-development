import copy, math, unittest
from division_retention import retain_learned_divisions as retain, validate_graph,score,CONFIG

def fixture():
    n={0:{'node_id':0,'t':0,'z':0,'y':0,'x':0},
       1:{'node_id':1,'t':1,'z':0,'y':-8,'x':0},
       2:{'node_id':2,'t':1,'z':0,'y':8,'x':0},
       3:{'node_id':3,'t':2,'z':0,'y':-10,'x':0},
       4:{'node_id':4,'t':2,'z':0,'y':10,'x':0}}
    e=[{'source_id':a,'target_id':b,'edge_prob':.95} for a,b in [(0,1),(1,3),(2,4)]]
    return n,e,{(0,1):.96,(0,2):.94,(1,3):.95,(2,4):.95}

class Tests(unittest.TestCase):
    def test_restores_exact_orphan_fork(self):
        n,e,p=fixture();r,a=retain(n,e,p);self.assertEqual(len(r),4);self.assertEqual(a['added_forks'],1)
    def test_no_mutation(self):
        n,e,p=fixture();old=copy.deepcopy((n,e,p));retain(n,e,p);self.assertEqual((n,e,p),old)
    def test_existing_child_may_be_other_daughter(self):
        n,e,p=fixture();e[0]['target_id']=2;r,a=retain(n,e,p);self.assertEqual(a['added_forks'],1)
    def test_low_probability_rejected(self):
        n,e,p=fixture();p[0,2]=.79;self.assertEqual(retain(n,e,p)[1]['added_forks'],0)
    def test_missing_probability_rejected(self):
        n,e,p=fixture();p[0,2]=None;self.assertEqual(retain(n,e,p)[1]['added_forks'],0)
    def test_nan_probability_rejected(self):
        n,e,p=fixture();p[0,2]=float('nan');self.assertEqual(retain(n,e,p)[1]['added_forks'],0)
    def test_inf_probability_rejected(self):
        n,e,p=fixture();p[0,2]=float('inf');self.assertEqual(retain(n,e,p)[1]['added_forks'],0)
    def test_never_steals_a_child(self):
        n,e,p=fixture();n[5]=dict(n[0],node_id=5);e+=[{'source_id':5,'target_id':2}]
        self.assertEqual(retain(n,e,p)[1]['added_forks'],0)
    def test_no_existing_continuation(self):
        n,e,p=fixture();e=e[1:];self.assertEqual(retain(n,e,p)[1]['added_forks'],0)
    def test_no_forward_context(self):
        n,e,p=fixture();e=e[:-1];self.assertEqual(retain(n,e,p)[1]['added_forks'],0)
    def test_not_diverging(self):
        n,e,p=fixture();n[3]['y']=-7;n[4]['y']=7;self.assertEqual(retain(n,e,p)[1]['added_forks'],0)
    def test_outside_geometry(self):
        n,e,p=fixture();n[2]['y']=80;self.assertEqual(retain(n,e,p)[1]['added_forks'],0)
    def test_nonsymmetric(self):
        n,e,p=fixture();n[1]['y']=7;self.assertEqual(retain(n,e,p)[1]['added_forks'],0)
    def test_already_divided(self):
        n,e,p=fixture();e+=[{'source_id':0,'target_id':2}];self.assertEqual(retain(n,e,p)[1]['added_forks'],0)
    def test_wrong_time(self):
        n,e,p=fixture();n[2]['t']=2
        with self.assertRaises(ValueError):retain(n,e,p)
    def test_duplicates_rejected(self):
        n,e,p=fixture();e+=e[:1]
        with self.assertRaises(ValueError):retain(n,e,p)
    def test_invalid_nodes_rejected(self):
        n,e,p=fixture();n[2]['x']=float('nan')
        with self.assertRaises(ValueError):retain(n,e,p)
    def test_repeat_is_idempotent(self):
        n,e,p=fixture();r,_=retain(n,e,p);r2,a=retain(n,r,p);self.assertEqual(r,r2);self.assertEqual(a['added_forks'],0)
    def test_order_independent(self):
        n,e,p=fixture();r,a=retain(n,e,p);r2,b=retain(dict(reversed(list(n.items()))),list(reversed(e)),dict(reversed(list(p.items()))))
        self.assertEqual(a['accepted'],b['accepted']);self.assertEqual({(x['source_id'],x['target_id']) for x in r},{(x['source_id'],x['target_id']) for x in r2})
    def test_empty(self): self.assertEqual(retain({},[],{})[1]['added_forks'],0)
    def test_bad_configuration_rejected(self):
        n,e,p=fixture();cfg=dict(CONFIG,min_learned_score=2)
        with self.assertRaises(ValueError):retain(n,e,p,cfg)
    def test_probability_conversion_matches_reference(self):
        self.assertAlmostEqual(score(2),1/(1+math.exp(-2)));self.assertEqual(score(.8),.8);self.assertIsNone(score('a'))
    def test_no_ground_truth_parameter(self):
        import inspect
        self.assertEqual(list(inspect.signature(retain).parameters),['nodes','motion_edges','learned_edge_probs','config'])

if __name__=='__main__':unittest.main(verbosity=2)
