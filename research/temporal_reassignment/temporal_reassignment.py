"""One-pass, label-free six-frame correction of isolated identity exchanges.

The frozen graph determines candidate neighborhoods. Unsmooth coordinates from
cached detections determine costs. Outputs keep every node, coordinate, edge
count and node degree unchanged. No learned scores are invented for new edges.
This is a conservative mechanistic experiment, not a Trackastra implementation.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import asdict, dataclass
import copy
import math
import time
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

SCALE = np.array([1.625, .40625, .40625], dtype=np.float64)

@dataclass(frozen=True)
class Settings:
    source_neighbor_radius_um: float = 12.0
    max_new_step_um: float = 10.0
    max_context_step_um: float = 10.0
    max_context_acceleration_um: float = 1.5
    max_new_directional_residual_um: float = 2.0
    min_each_link_gain_um: float = .75
    min_total_gain_um: float = 2.0
    max_cost_ratio: float = .5
    max_pair_checks: int = 300000
    max_seconds: float = 90.0

CONFIG = Settings()

def validate(nodes: dict[int, dict], edges: list[dict]) -> tuple[dict, dict]:
    incoming, outgoing = defaultdict(list), defaultdict(list)
    for i,n in nodes.items():
        if not isinstance(i,int) or isinstance(i,bool) or int(n['node_id'])!=i:
            raise ValueError('NODE_ID_MISMATCH')
        t=float(n['t'])
        if not math.isfinite(t) or t<0 or t!=int(t):raise ValueError('INVALID_NODE_TIME')
        if not all(math.isfinite(float(n[k])) for k in ('z','y','x')):raise ValueError('NONFINITE_NODE')
    seen=set()
    for e in edges:
        a,b=int(e['source_id']),int(e['target_id'])
        if (a,b) in seen:raise ValueError('DUPLICATE_EDGE')
        seen.add((a,b))
        if a not in nodes or b not in nodes:raise ValueError('EDGE_ENDPOINT_MISSING')
        if nodes[b]['t']!=nodes[a]['t']+1:raise ValueError('NONCONSECUTIVE_EDGE')
        incoming[b].append(a);outgoing[a].append(b)
    if any(len(x)>1 for x in incoming.values()):raise ValueError('MERGE_IN_INPUT')
    if any(len(x)>2 for x in outgoing.values()):raise ValueError('INVALID_FORK_IN_INPUT')
    return dict(incoming),dict(outgoing)

def configuration_check(c: Settings) -> None:
    for k,v in asdict(c).items():
        if not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:raise ValueError('INVALID_CONFIG '+k)
    if c.max_cost_ratio>=1:raise ValueError('COST_RATIO_MUST_BE_BELOW_ONE')

def _context_positions(chain, nodes, raw):
    # No interpolation: skip neighborhoods containing added synthetic detections.
    if any(i not in raw or raw[i]['t']!=nodes[i]['t'] for i in chain):return None
    result=np.array([[raw[i][k] for k in ('z','y','x')] for i in chain],dtype=np.float64)*SCALE
    if not np.isfinite(result).all():raise ValueError('NONFINITE_RAW_CONTEXT')
    return result

def _link_cost(left, right):
    lv=(left[2]-left[0])/2.0
    rv=(right[2]-right[0])/2.0
    forward=float(np.linalg.norm(right[0]-(left[2]+lv)))
    backward=float(np.linalg.norm(left[2]-(right[0]-rv)))
    return .5*(forward+backward),forward,backward

def reassign(nodes: dict[int,dict], edges: list[dict], raw_nodes: dict[int,dict],
             config: Settings=CONFIG) -> tuple[list[dict],dict[str,Any]]:
    """Reassign two t->t+1 links only with strong agreement from both time directions.

    `raw_nodes` is a prediction-only coordinate table, never annotations. Three
    detections on either side of the cut establish two-step velocities. The
    entire six-frame context is reserved after each accepted exchange, avoiding
    interacting corrections or reuse of context another accepted swap changed.
    """
    configuration_check(config);start=time.monotonic();inc,out=validate(nodes,edges)
    contexts={};by_time=defaultdict(list)
    for e in edges:
        a,b=int(e['source_id']),int(e['target_id'])
        if len(out.get(a,[]))!=1 or len(inc.get(a,[]))!=1 or len(inc.get(b,[]))!=1 or len(out.get(b,[]))!=1:continue
        a1=inc[a][0];b1=out[b][0]
        if len(inc.get(a1,[]))!=1 or len(out.get(a1,[]))!=1 or len(inc.get(b1,[]))!=1 or len(out.get(b1,[]))!=1:continue
        a2=inc[a1][0];b2=out[b1][0]
        if len(out.get(a2,[]))!=1 or len(inc.get(b2,[]))!=1:continue
        # Exclude a nearby division even on the outer frame of the context.
        if len(inc.get(a2,[]))>1 or len(out.get(b2,[]))>1:continue
        chain=[a2,a1,a,b,b1,b2]
        if len(set(chain))!=6:continue
        pos=_context_positions(chain,nodes,raw_nodes)
        if pos is None:continue
        left,right=pos[:3],pos[3:]
        if any(float(np.linalg.norm(v))>config.max_context_step_um for v in (*np.diff(left,axis=0),*np.diff(right,axis=0))):continue
        if max(float(np.linalg.norm(left[2]-2*left[1]+left[0])),float(np.linalg.norm(right[2]-2*right[1]+right[0])))>config.max_context_acceleration_um:continue
        key=(a,b);contexts[key]=(chain,left,right,e);by_time[int(nodes[a]['t'])].append(key)
    candidates=[];checked=0
    for t,keys in sorted(by_time.items()):
        keys=sorted(keys)
        if len(keys)<2:continue
        xyz=np.array([contexts[k][1][2] for k in keys]);tree=cKDTree(xyz)
        # Query one source at a time: no dense N*N distance or all-pairs allocation.
        for i,key in enumerate(keys):
            for j in sorted(tree.query_ball_point(xyz[i],config.source_neighbor_radius_um)):
                if j<=i:continue
                checked+=1
                if checked>config.max_pair_checks:raise RuntimeError('PAIR_CHECK_BUDGET_EXCEEDED')
                if checked%1000==0 and time.monotonic()-start>config.max_seconds:raise TimeoutError('CANDIDATE_TIME_BUDGET_EXCEEDED')
                other=keys[j];chain,l1,r1,_=contexts[key];chain2,l2,r2,_=contexts[other]
                if set(chain)&set(chain2):continue
                if max(float(np.linalg.norm(r2[0]-l1[2])),float(np.linalg.norm(r1[0]-l2[2])))>config.max_new_step_um:continue
                old1,_,_=_link_cost(l1,r1);old2,_,_=_link_cost(l2,r2)
                new1,f1,b1=_link_cost(l1,r2);new2,f2,b2=_link_cost(l2,r1)
                if max(f1,b1,f2,b2)>config.max_new_directional_residual_um:continue
                gain1,gain2=old1-new1,old2-new2
                if min(gain1,gain2)<config.min_each_link_gain_um:continue
                old,new=old1+old2,new1+new2;gain=old-new
                if gain<config.min_total_gain_um or new>config.max_cost_ratio*old:continue
                candidates.append({'t':t,'original_edges':[list(key),list(other)],
                    'replacement_edges':[[key[0],other[1]],[other[0],key[1]]],
                    'old_cost_um':old,'new_cost_um':new,'gain_um':gain,
                    'forward_backward_residuals_um':[f1,b1,f2,b2],
                    'context_node_ids':sorted(chain+chain2)})
    reserved=set();removed=set();new_edges=[];accepted=[]
    for p in sorted(candidates,key=lambda x:(-x['gain_um'],x['t'],x['original_edges'])):
        if reserved.intersection(p['context_node_ids']):continue
        reserved.update(p['context_node_ids']);removed.update(tuple(x) for x in p['original_edges'])
        for a,b in p['replacement_edges']:
            distance=float(np.linalg.norm((np.array([raw_nodes[b][k] for k in ('z','y','x')])-np.array([raw_nodes[a][k] for k in ('z','y','x')]))*SCALE))
            new_edges.append({'source_id':a,'target_id':b,'edge_prob':None,'distance_um':distance,
                              'bidirectional_temporal_reassignment':True})
        accepted.append(p)
    result=[copy.deepcopy(e) for e in edges if (int(e['source_id']),int(e['target_id'])) not in removed]+new_edges
    result.sort(key=lambda e:(int(e['source_id']),int(e['target_id'])))
    ni,no=validate(nodes,result)
    if len(result)!=len(edges):raise AssertionError('EDGE_COUNT_CHANGED')
    for i in nodes:
        if len(inc.get(i,[]))!=len(ni.get(i,[])) or len(out.get(i,[]))!=len(no.get(i,[])):raise AssertionError('NODE_DEGREE_CHANGED')
    for i,children in out.items():
        if len(children)==2 and set(no[i])!=set(children):raise AssertionError('EXISTING_DIVISION_EDGES_CHANGED')
    return result,{'config':asdict(config),'eligible_edges_with_six_frame_context':len(contexts),
                   'neighbor_pairs_checked':checked,'admissible_swaps':len(candidates),
                   'accepted_swaps':len(accepted),'removed_edge_count':len(removed),
                   'added_edge_count':len(new_edges),'node_coordinates_changed':False,
                   'edge_count_preserved':True,'node_degrees_preserved':True,
                   'existing_division_edges_preserved':True,'swaps':accepted,
                   'elapsed_seconds':time.monotonic()-start}
