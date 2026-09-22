"""Conservative restoration of learned forks erased by one-to-one relinking.

No labels, training, model fitting, identifiers-as-features or future GT are used.
Full observed movie context is used at inference, as in the frozen reference.
"""
from __future__ import annotations
import math
from collections import defaultdict

CONFIG = {
    'min_learned_score': 0.80,
    'max_parent_um': 9.0,
    'max_sister_um': 14.0,
    'max_midpoint_ratio': 0.60,
    'min_separation_growth_um': 0.50,
    'max_added_fraction': 0.00375,
}
SCALE = (1.625, 0.40625, 0.40625)

def score(value):
    try: value = float(value)
    except (ValueError, TypeError): return None
    if not math.isfinite(value): return None
    if value < 0.0 or value > 1.0:
        value = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, value))))
    return value

def pos(node):
    p = tuple(float(node[k]) * s for k,s in zip(('z','y','x'), SCALE))
    if not all(math.isfinite(v) for v in p): raise ValueError('NONFINITE_NODE_POSITION')
    return p

def distance(a,b): return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

def validate_graph(nodes, edges):
    incoming, outgoing, seen = defaultdict(int), defaultdict(int), set()
    for key,n in nodes.items():
        if int(n['node_id']) != int(key): raise ValueError('NODE_ID_MISMATCH')
        if int(n['t']) != n['t'] or int(n['t']) < 0: raise ValueError('INVALID_NODE_TIME')
        pos(n)
    for e in edges:
        a,b = int(e['source_id']),int(e['target_id'])
        if a not in nodes or b not in nodes or a==b: raise ValueError('EDGE_ENDPOINT_INVALID')
        if (a,b) in seen: raise ValueError('DUPLICATE_EDGE')
        if int(nodes[b]['t']) != int(nodes[a]['t'])+1: raise ValueError('EDGE_TIME_INVALID')
        incoming[b]+=1; outgoing[a]+=1; seen.add((a,b))
        if incoming[b]>1 or outgoing[a]>2: raise ValueError('LINEAGE_DEGREE_INVALID')
    return {'nodes':len(nodes),'edges':len(edges),'forks':sum(v==2 for v in outgoing.values())}

def retain_learned_divisions(nodes, motion_edges, learned_edge_probs, config=None):
    """Add only an orphan second branch of an exact two-daughter learned fork.

    The existing continuation must be one proposed daughter. Both daughters
    must continue for another observed frame and separate, rather than merge.
    Already assigned children are never stolen. No existing edge is deleted.
    All gates are predeclared; this function never accepts ground-truth input.
    """
    cfg=dict(CONFIG if config is None else config)
    if set(cfg)!=set(CONFIG): raise ValueError('CONFIG_KEYS_MISMATCH')
    if not all(math.isfinite(float(v)) for v in cfg.values()): raise ValueError('CONFIG_NONFINITE')
    if not (0<=cfg['min_learned_score']<=1 and 0<cfg['max_added_fraction']<=0.01):
        raise ValueError('CONFIG_RANGE_INVALID')
    if any(cfg[k]<0 for k in cfg if k not in ('min_learned_score','max_added_fraction')):
        raise ValueError('CONFIG_RANGE_INVALID')
    validate_graph(nodes,motion_edges)
    out=defaultdict(list); incoming={}; learned=defaultdict(dict)
    for e in motion_edges:
        a,b=int(e['source_id']),int(e['target_id']);out[a].append(b);incoming[b]=a
    audit={'raw_scored_edges':0,'raw_two_child_parents':0,'eligible_forks':0,'added_forks':0,
           'skipped_probability':0,'skipped_assignment':0,'skipped_geometry':0,
           'skipped_context':0,'skipped_cap':0,'config':cfg,'accepted':[]}
    for (a,b),value in (learned_edge_probs or {}).items():
        a,b=int(a),int(b);p=score(value)
        if p is not None and a in nodes and b in nodes and int(nodes[b]['t'])==int(nodes[a]['t'])+1:
            learned[a][b]=p; audit['raw_scored_edges']+=1
    positions={k:pos(n) for k,n in nodes.items()}; proposals=[]
    for a, children in sorted(learned.items()):
        if len(children)!=2: continue
        audit['raw_two_child_parents']+=1
        b,c=sorted(children); pb,pc=children[b],children[c]
        if min(pb,pc)<cfg['min_learned_score']:
            audit['skipped_probability']+=1;continue
        existing=out.get(a,[])
        if len(existing)!=1 or existing[0] not in (b,c):
            audit['skipped_assignment']+=1;continue
        orphan=c if existing[0]==b else b
        if orphan in incoming:
            audit['skipped_assignment']+=1;continue
        pa,pbpos,pcpos=positions[a],positions[b],positions[c]
        db,dc=distance(pa,pbpos),distance(pa,pcpos);sep=distance(pbpos,pcpos)
        midpoint=tuple((x+y)/2 for x,y in zip(pbpos,pcpos))
        symmetry=distance(pa,midpoint)/max((db+dc)/2,1e-9)
        if max(db,dc)>cfg['max_parent_um'] or sep>cfg['max_sister_um'] or symmetry>cfg['max_midpoint_ratio']:
            audit['skipped_geometry']+=1;continue
        nb,nc=out.get(b,[]),out.get(c,[])
        if len(nb)!=1 or len(nc)!=1 or nb[0]==nc[0]:
            audit['skipped_context']+=1;continue
        growth=distance(positions[nb[0]],positions[nc[0]])-sep
        if growth<cfg['min_separation_growth_um']:
            audit['skipped_context']+=1;continue
        proposals.append((-min(pb,pc),a,orphan,db if orphan==b else dc,growth))
    audit['eligible_forks']=len(proposals)
    # Explicit small global budget; no removal of previously selected forks.
    cap=max(1,int(len(nodes)*cfg['max_added_fraction'])) if nodes else 0
    result=[dict(e) for e in motion_edges]
    for neg,a,b,d,growth in sorted(proposals):
        if audit['added_forks']>=cap:
            audit['skipped_cap']+=1;continue
        if b in incoming or len(out.get(a,[]))!=1:
            audit['skipped_assignment']+=1;continue
        result.append({'source_id':a,'target_id':b,'edge_prob':learned[a][b],
            'distance_um':d,'retained_learned_division':1})
        incoming[b]=a;out[a].append(b);audit['added_forks']+=1
        audit['accepted'].append({'parent':a,'new_daughter':b,'min_learned_score':-neg,
                                 'distance_um':d,'sister_growth_um':growth})
    validate_graph(nodes,result)
    assert {(int(e['source_id']),int(e['target_id'])) for e in motion_edges}.issubset(
        {(int(e['source_id']),int(e['target_id'])) for e in result})
    return result,audit
