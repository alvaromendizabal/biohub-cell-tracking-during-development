"""Ground-truth-assisted diagnostics only. Never used to alter a prediction.

Partitions counts from the pinned sparse-annotation evaluator. Unmatched
predictions are not automatically false positives. Algebraic score scenarios
hold other terms fixed, are non-additive, and are not achieved model results.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import math

SCALE=(1.625,.40625,.40625)

def require(ok: bool, message: str) -> None:
    if not ok: raise RuntimeError(message)

def adjacency(nodes: dict, edges: list) -> tuple[dict,dict]:
    inc=defaultdict(list);out=defaultdict(list);seen=set()
    for e in edges:
        a,b=int(e['source_id']),int(e['target_id'])
        require((a,b) not in seen,'DUPLICATE_EDGE');seen.add((a,b))
        require(a in nodes and b in nodes,'UNKNOWN_ENDPOINT')
        require(int(nodes[b]['t'])==int(nodes[a]['t'])+1,'NONCONSECUTIVE_EDGE')
        inc[b].append(a);out[a].append(b)
    return dict(inc),dict(out)

def inverse_matching(pred_to_gt: dict, pred_nodes: dict, gt_nodes: dict) -> dict:
    inv={}
    for p,g in pred_to_gt.items():
        if g is None or int(g)<0:continue
        p,g=int(p),int(g)
        require(p in pred_nodes and g in gt_nodes,'MATCH_UNKNOWN_NODE')
        require(int(pred_nodes[p]['t'])==int(gt_nodes[g]['t']),'MATCH_TIME_MISMATCH')
        require(g not in inv,'NONUNIQUE_GT_MATCH')
        inv[g]=p
    return inv

def physical_distance(a,b):
    return math.sqrt(sum(((float(a[k])-float(b[k]))*s)**2 for k,s in zip(('z','y','x'),SCALE)))

def partition_edges(pred_nodes, pred_edges, gt_nodes, gt_edges, pred_to_gt,
                    evaluated_edges, expected, raw_to_gt=None, raw_nodes=None):
    """Partition every counted FP/FN; masks/validity come from the real evaluator.

    evaluated_edges contains original prediction IDs with bool matched/valid.
    Raw matching is a separately recomputed one-to-one geometric assignment;
    changes from raw to final may reflect competition or smoothing, not removal.
    """
    inc,out=adjacency(pred_nodes,pred_edges);gi,go=adjacency(gt_nodes,gt_edges)
    inv=inverse_matching(pred_to_gt,pred_nodes,gt_nodes)
    raw_inv=inverse_matching(raw_to_gt or {},raw_nodes or {},gt_nodes)
    gt_pairs={(int(e['source_id']),int(e['target_id'])) for e in gt_edges}
    pred_pairs={(int(e['source_id']),int(e['target_id'])) for e in pred_edges}
    recovered=set();false_rows=[];ignored=0;tp=0
    for e in evaluated_edges:
        a,b=int(e['source_id']),int(e['target_id']);ma=pred_to_gt.get(a);mb=pred_to_gt.get(b)
        if ma is not None and int(ma)<0:ma=None
        if mb is not None and int(mb)<0:mb=None
        if e['matched']:
            require(e['valid'],'MATCHED_EDGE_NOT_VALID')
            require(ma is not None and mb is not None and (ma,mb) in gt_pairs,'TRUE_EDGE_NOT_GT_PAIR')
            require((ma,mb) not in recovered,'DUPLICATE_MATCHED_GT_EDGE')
            recovered.add((ma,mb));tp+=1
        elif e['valid']:
            require(ma is not None or mb is not None,'UNMATCHED_EDGE_CANNOT_BE_SPARSE_FP')
            category=('both_endpoints_matched_wrong_link' if ma is not None and mb is not None
                      else 'source_matched_target_unmatched' if ma is not None else 'source_unmatched_target_matched')
            false_rows.append({'source_id':a,'target_id':b,'matched_gt_source':ma,'matched_gt_target':mb,
                               'category':category,'t':int(pred_nodes[a]['t']),
                               'distance_um':physical_distance(pred_nodes[a],pred_nodes[b]),
                               'source_outdegree':len(out.get(a,[])), 'target_indegree':len(inc.get(b,[]))})
        else: ignored+=1
    missed=[]
    for a,b in sorted(gt_pairs-recovered):
        pa,pb=inv.get(a),inv.get(b)
        if pa is None and pb is None:cat='both_endpoints_unmatched'
        elif pa is None:cat='source_endpoint_unmatched'
        elif pb is None:cat='target_endpoint_unmatched'
        elif len(inc.get(pb,[]))>0:cat='target_assigned_elsewhere'
        elif len(out.get(pa,[]))>=2:cat='source_at_two_child_capacity'
        elif len(out.get(pa,[]))==1 and len(go.get(a,[]))==2:cat='missing_second_daughter_edge'
        elif len(out.get(pa,[]))==0:cat='free_endpoint_fragmentation'
        else:cat='source_continues_elsewhere'
        if pa is not None and pb is not None:require((pa,pb) not in pred_pairs,'EXISTING_TRUE_PAIR_NOT_COUNTED')
        row={'gt_source_id':a,'gt_target_id':b,'pred_source_id':pa,'pred_target_id':pb,'category':cat,
             'is_gt_division_edge':len(go.get(a,[]))==2,'t':int(gt_nodes[a]['t']),
             'source_outdegree':None if pa is None else len(out.get(pa,[])),
             'target_indegree':None if pb is None else len(inc.get(pb,[])),
             'pred_distance_um':None if pa is None or pb is None else physical_distance(pred_nodes[pa],pred_nodes[pb]),
             'raw_source_matched':a in raw_inv,'raw_target_matched':b in raw_inv}
        missed.append(row)
    require(tp==int(expected['edge_tp']),'EDGE_TP_ATTRIBUTION_MISMATCH')
    require(len(false_rows)==int(expected['edge_fp']),'EDGE_FP_ATTRIBUTION_MISMATCH')
    require(len(missed)==int(expected['edge_fn']),'EDGE_FN_ATTRIBUTION_MISMATCH')
    node_missing=[]
    for g in sorted(set(gt_nodes)-set(inv)):
        raw_id=raw_inv.get(g)
        # An ID removed from final is direct evidence of pruning. Remaining IDs
        # can lose assignment after coordinate changes or matching competition.
        kind=('unmatched_in_raw_and_final' if raw_id is None else
              'raw_matched_node_removed_from_final' if raw_id not in pred_nodes else
              'raw_matched_node_retained_but_final_matching_changed')
        node_missing.append({'gt_node_id':g,'t':int(gt_nodes[g]['t']),'category':kind,'raw_matched_node_id':raw_id})
    counts=lambda rows:dict(sorted(Counter(x['category'] for x in rows).items()))
    return {'edge_tp':tp,'edge_fp':len(false_rows),'edge_fn':len(missed),'ignored_prediction_edges':ignored,
            'fn_categories':counts(missed),'fp_categories':counts(false_rows),'unmatched_gt_node_categories':counts(node_missing),
            'false_negative_edges':missed,'false_positive_edges':false_rows,'unmatched_gt_nodes':node_missing,
            'gt_nodes':len(gt_nodes),'gt_edges':len(gt_pairs),'matched_gt_nodes':len(inv),
            'raw_matched_gt_nodes':len(raw_inv),'note':'Annotation-assisted diagnostic labels. No prediction modifications.'}

def division_reason(*,recovered,parent_support,daughter_support,local_forks,valid_topology_forks,invalid_forks):
    if recovered:return 'recovered'
    if not parent_support:return 'parent_window_unmatched'
    if sum(bool(x) for x in daughter_support)<2:return 'daughter_window_unmatched'
    if not local_forks:return 'parent_and_daughters_present_no_local_fork'
    if set(local_forks).issubset(invalid_forks):return 'local_forks_invalid_by_component_or_merge_rules'
    if not valid_topology_forks:return 'local_forks_do_not_connect_both_daughters'
    return 'valid_local_evidence_not_selected_by_event_matching'

def scalar_score(rows):
    require(bool(rows),'NO_ROWS')
    for r in rows:
        for k in ('edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn'):
            require(math.isfinite(r[k]) and r[k]>=0,'INVALID_COUNT')
    weights=[r['edge_tp']+r['edge_fp']+r['edge_fn'] for r in rows];total=sum(weights)
    require(total>0,'NO_EDGE_EVALUATION')
    adj=sum(max(0.,r['edge_tp']/w*(1-.1*r['total_node_ratio']))*w for r,w in zip(rows,weights) if w)/total
    dt=sum(r['division_tp'] for r in rows);dfp=sum(r['division_fp'] for r in rows);dfn=sum(r['division_fn'] for r in rows)
    return adj+(.1*dt/(dt+dfp+dfn) if dt+dfp+dfn else 0.)

def sensitivity_table(movie_audits):
    """Counterfactual count sensitivities only; NOT achievable model estimates."""
    import copy
    rows=[m['metric'] for m in movie_audits];base=scalar_score(rows);scenarios=[]
    cats=sorted({k for m in movie_audits for k in m['edges']['fn_categories']})
    cases=[('correct_all_counted_false_edges','fp',None),('recover_all_missed_divisions','div_fn',None)]+[
        ('recover_missed_edges__'+c,'fn',c) for c in cats]
    for name,kind,cat in cases:
        mod=copy.deepcopy(rows);events=0
        for r,m in zip(mod,movie_audits):
            if kind=='fp':n=r['edge_fp'];r['edge_fp']=0
            elif kind=='div_fn':n=r['division_fn'];r['division_tp']+=n;r['division_fn']=0
            else:
                n=m['edges']['fn_categories'].get(cat,0);r['edge_tp']+=n;r['edge_fn']-=n
            events+=n
        score=scalar_score(mod)
        scenarios.append({'scenario':name,'event_count':events,'count_sensitivity_score':score,
                          'delta_from_local_baseline':score-base,
                          'achieved_result':False,'assumptions':'Other counts and predicted node count fixed; scenarios overlap and must not be summed.'})
    return sorted(scenarios,key=lambda x:(-x['delta_from_local_baseline'],x['scenario']))
