"""Label-free competing-parent division proposals and a small sparse-supervision scorer.

This is a new, experimental graph-feature model, not the published HOCT model.
No annotation, embryo ID, absolute time, or spatial ID enters the feature function.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import defaultdict
import math, time
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit
from scipy.optimize import minimize

@dataclass(frozen=True)
class Config:
    radius_um: float = 15.0
    alternatives_per_parent: int = 4
    history_steps: int = 3
    ridge: float = 0.1
    minimum_score: float = 0.5
    maximum_new_forks_per_movie: int = 32
    max_events_per_movie: int = 250000
    max_fit_seconds: int = 90
    minimum_positive_events_per_training_embryo: int = 2
CONFIG=Config()
FEATURES=('distance_existing','distance_alternative','distance_sum','distance_max','distance_min',
 'daughter_separation','parent_midpoint_distance','distance_ratio','daughter_direction_cosine',
 'parent_speed','parent_history','parent_motion_noise',
 'forward_existing_error','forward_alternative_error','forward_midpoint_error',
 'existing_daughter_speed','alternative_daughter_speed','daughter_velocity_difference',
 'daughter_midpoint_velocity_error','daughter_separation_change',
 'existing_daughter_future','alternative_daughter_future',
 'alternative_has_other_parent','incumbent_distance','incumbent_speed','incumbent_history',
 'incumbent_forward_error','incumbent_vs_proposed_distance','incumbent_motion_noise')
SCALE=np.array([1.625,.40625,.40625],dtype=float)

def require(ok,message):
    if not ok: raise RuntimeError(message)

def structure(nodes,edges):
    incoming={};outgoing=defaultdict(list);seen=set()
    for i,n in nodes.items():
        require(int(i)==i and i>=0,'INVALID_NODE_ID')
        require(int(n['t'])==n['t'] and n['t']>=0,'INVALID_TIME')
        require(all(math.isfinite(float(n[k])) for k in ('z','y','x')),'NONFINITE_COORDINATE')
    for e in edges:
        a,b=int(e['source_id']),int(e['target_id'])
        require(a in nodes and b in nodes,'EDGE_ENDPOINT_MISSING')
        require((a,b) not in seen,'DUPLICATE_EDGE');seen.add((a,b))
        require(nodes[b]['t']==nodes[a]['t']+1,'NONADJACENT_TIME_EDGE')
        require(b not in incoming,'MERGED_CHILD');incoming[b]=a;outgoing[a].append(b)
        require(len(outgoing[a])<=2,'TOO_MANY_DAUGHTERS')
    for v in outgoing.values():v.sort()
    return incoming,outgoing

def motion(points,nodes,incoming,outgoing,forward=False,steps=3):
    result={}
    for i in nodes:
        ids=[i];j=i
        for _ in range(steps):
            if forward:
                nxt=outgoing.get(j,[])
                if len(nxt)!=1:break
                j=nxt[0]
            else:
                if j not in incoming:break
                q=incoming[j]
                if len(outgoing.get(q,[]))!=1:break
                j=q
            ids.append(j)
        if len(ids)==1:result[i]=(np.zeros(3),0,0.);continue
        seq=np.array([points[k] for k in ids])
        velocities=np.diff(seq,axis=0)*(1 if forward else -1)
        vel=np.median(velocities,axis=0)
        noise=float(np.sqrt(np.mean(np.sum((velocities-vel)**2,axis=1))))
        result[i]=(vel,len(ids)-1,noise)
    return result

def features(p,a,b,q,points,back,future):
    pp,pa,pb=points[p],points[a],points[b]
    da,db=pa-pp,pb-pp;ra=float(np.linalg.norm(da));rb=float(np.linalg.norm(db))
    vp,npast,noise=back[p];va,na,_=future[a];vb,nb,_=future[b]
    midpoint=(pa+pb)/2;sep=float(np.linalg.norm(pa-pb))
    inc_d=inc_s=inc_h=inc_e=inc_n=0.
    if q>=0:
        vq,inc_h,inc_n=back[q];inc_d=float(np.linalg.norm(pb-points[q]))
        inc_s=float(np.linalg.norm(vq));inc_e=float(np.linalg.norm(pb-points[q]-vq))
    row=[ra,rb,ra+rb,max(ra,rb),min(ra,rb),sep,float(np.linalg.norm(midpoint-pp)),
         min(ra,rb)/max(ra,rb,1e-9),float(np.dot(da,db)/max(ra*rb,1e-9)),
         float(np.linalg.norm(vp)),npast,noise,float(np.linalg.norm(da-vp)),float(np.linalg.norm(db-vp)),
         float(np.linalg.norm(midpoint-pp-vp)),float(np.linalg.norm(va)),float(np.linalg.norm(vb)),
         float(np.linalg.norm(va-vb)),float(np.linalg.norm((va+vb)/2-vp)),
         float(np.linalg.norm((pa+va)-(pb+vb)))-sep,na,nb,int(q>=0),inc_d,inc_s,inc_h,inc_e,inc_d-rb,inc_n]
    require(len(row)==len(FEATURES) and np.isfinite(row).all(),'FEATURE_INVALID')
    return row

def propose(nodes,edges,config=CONFIG,heartbeat=None):
    """Enumerate proposals on every frame; never receives labels or diagnostic IDs.

    Columns of ids: parent, existing daughter, proposed daughter, displaced parent,
    existing_fork (training/control example, never modified).
    """
    require(config.radius_um>0 and config.alternatives_per_parent>0,'INVALID_CONFIG')
    incoming,outgoing=structure(nodes,edges)
    points={i:np.array([n[k] for k in ('z','y','x')],dtype=float)*SCALE for i,n in nodes.items()}
    back=motion(points,nodes,incoming,outgoing,False,config.history_steps)
    future=motion(points,nodes,incoming,outgoing,True,config.history_steps)
    frames=defaultdict(list)
    for i,n in nodes.items():frames[int(n['t'])].append(i)
    rows=[];ids=[];counts=defaultdict(int)
    for t in sorted(frames):
        targets=sorted(frames.get(t+1,[]))
        if not targets:continue
        xyz=np.array([points[i] for i in targets]);tree=cKDTree(xyz)
        for p in sorted(frames[t]):
            children=outgoing.get(p,[])
            if len(children)==2:
                a,b=children
                ids.append((p,a,b,-1,1));rows.append(features(p,a,b,-1,points,back,future));counts['existing_forks']+=1
                continue
            if len(children)!=1:continue
            a=children[0]
            if float(np.linalg.norm(points[a]-points[p]))>config.radius_um:continue
            distances,positions=tree.query(points[p],k=min(len(targets),config.alternatives_per_parent+3),distance_upper_bound=config.radius_um)
            offered=[]
            for d,j in zip(np.atleast_1d(distances),np.atleast_1d(positions)):
                if not np.isfinite(d) or int(j)>=len(targets):continue
                b=targets[int(j)]
                if b==a:continue
                q=incoming.get(b,-1)
                if q==p or (q>=0 and len(outgoing.get(q,[]))!=1):continue
                offered.append((float(d),b,q))
            for _,b,q in sorted(offered)[:config.alternatives_per_parent]:
                ids.append((p,a,b,q,0));rows.append(features(p,a,b,q,points,back,future))
                counts['new_proposals']+=1;counts['competing_parent_proposals' if q>=0 else 'unassigned_child_proposals']+=1
            require(len(ids)<=config.max_events_per_movie,'EVENT_COUNT_LIMIT')
        if heartbeat:heartbeat(t,len(ids))
    xx=np.asarray(rows,dtype=np.float32).reshape(-1,len(FEATURES));ii=np.asarray(ids,dtype=np.int64).reshape(-1,5)
    return ii,xx,dict(counts)

def event_support(event,incoming,outgoing,window):
    p,a,b,q,existing=(int(v) for v in event)
    parent_ids=set(window['parent_side_prediction_ids'])
    anchors={p}
    if p in incoming:anchors.add(incoming[p])
    if not anchors&parent_ids:return False
    daughters=[set(x) for x in window['daughter_side_prediction_ids']]
    if len(daughters)!=2 or not all(daughters):return False
    A={a,*outgoing.get(a,[])};B={b,*outgoing.get(b,[])}
    return bool((A&daughters[0] and B&daughters[1]) or (B&daughters[0] and A&daughters[1]))

def labels_for_events(ids,incoming,outgoing,windows,global_match,gt_outgoing,gt_components):
    """Sparse event labels. Unassessable proposals remain -1, not negative.

    Window supports are exact fixed-node matches from the preceding audit.
    Positives must have two distinct supported daughter branches and cannot
    span conflicting known GT components. Final full-metric scoring is separate.
    """
    labels=np.full(len(ids),-1,dtype=np.int8);positive_event=np.full(len(ids),-1,dtype=np.int64)
    for i,event in enumerate(ids):
        p,a,b,q,existing=map(int,event)
        comps=[]
        for child in (a,b):
            gt=global_match.get(child)
            if gt is not None:known={gt_components[gt]}
            else:known={gt_components[global_match[j]] for j in outgoing.get(child,[]) if j in global_match}
            comps.append(known if len(known)==1 else set())
        cross=bool(comps[0] and comps[1] and comps[0]!=comps[1])
        hits=[int(w['gt_divider_id']) for w in windows if event_support(event,incoming,outgoing,w)]
        if hits and not cross:
            labels[i]=1;positive_event[i]=min(hits)
        elif cross or (p in global_match and gt_outgoing.get(global_match[p],[])):
            labels[i]=0
    return labels,positive_event

def fit_model(x,y,event_ids,config=CONFIG):
    """Train one regularized event classifier; inputs must contain training embryo only."""
    x=np.asarray(x,dtype=float);y=np.asarray(y);event_ids=np.asarray(event_ids)
    known=(y>=0);x=x[known];y=y[known];event_ids=event_ids[known]
    require(x.ndim==2 and len(x)==len(y) and np.isfinite(x).all(),'BAD_TRAINING_ARRAY')
    positive=set(int(i) for i in event_ids[y==1])
    require(len(positive)>=config.minimum_positive_events_per_training_embryo,'INSUFFICIENT_DISTINCT_TRAIN_DIVISIONS')
    require(np.sum(y==0)>=20,'INSUFFICIENT_KNOWN_NEGATIVE_EVENTS')
    mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale<1e-6]=1.
    z=np.clip((x-mean)/scale,-10,10)
    weights=np.zeros(len(y));weights[y==0]=.5/max(1,int(np.sum(y==0)))
    for e in positive:
        take=(y==1)&(event_ids==e);weights[take]=.5/(len(positive)*np.sum(take))
    start=time.monotonic()
    def objective(theta):
        require(time.monotonic()-start<=config.max_fit_seconds,'FIT_TIME_LIMIT')
        logits=z@theta[:-1]+theta[-1]
        loss=float(np.sum(weights*(np.logaddexp(0,logits)-y*logits))+.5*config.ridge*np.dot(theta[:-1],theta[:-1]))
        residual=weights*(expit(logits)-y)
        grad=np.r_[z.T@residual+config.ridge*theta[:-1],residual.sum()]
        require(math.isfinite(loss) and np.isfinite(grad).all(),'NONFINITE_OPTIMIZATION')
        return loss,grad
    opt=minimize(objective,np.zeros(z.shape[1]+1),jac=True,method='L-BFGS-B',options={'maxiter':150,'ftol':1e-10,'gtol':1e-7})
    require(opt.success and np.isfinite(opt.x).all(),'FIT_NOT_CONVERGED '+str(opt.message))
    scores=expit(z@opt.x[:-1]+opt.x[-1])
    # Threshold is chosen from training negatives only. No held-out label is consulted.
    threshold=max(config.minimum_score,float(np.max(scores[y==0]))+1e-6)
    threshold=min(threshold,1.000001)
    return {'mean':mean.tolist(),'scale':scale.tolist(),'coef':opt.x[:-1].tolist(),'intercept':float(opt.x[-1]),
      'threshold':threshold,'feature_names':list(FEATURES),'distinct_training_divisions':len(positive),
      'known_training_events':len(y),'training_positive_proposals':int(np.sum(y==1)),
      'training_negative_proposals':int(np.sum(y==0)),'training_tp_above_threshold':int(np.sum((y==1)&(scores>=threshold))),
      'training_fp_above_threshold':int(np.sum((y==0)&(scores>=threshold))),'optimizer_success':bool(opt.success),'iterations':int(opt.nit),
      'threshold_policy':'max(0.5, largest_training_negative_score + 1e-6); not a calibrated probability',
      'seconds':time.monotonic()-start}

def predict(model,x):
    z=np.clip((np.asarray(x,dtype=float)-np.asarray(model['mean']))/np.asarray(model['scale']),-10,10)
    out=expit(z@np.asarray(model['coef'])+float(model['intercept']))
    require(np.isfinite(out).all(),'NONFINITE_PREDICTIONS');return out

def apply_events(nodes,edges,ids,scores,threshold,config=CONFIG):
    """Decode nonconflicting event transactions; existing forks are protected."""
    incoming,outgoing=structure(nodes,edges);original={(int(e['source_id']),int(e['target_id'])):e for e in edges}
    result={k:dict(v) for k,v in original.items()};locked=set();changes=[];rejected=defaultdict(int)
    require(len(ids)==len(scores) and np.isfinite(scores).all(),'EVENT_SCORE_INVALID')
    order=sorted(range(len(ids)),key=lambda i:(-float(scores[i]),tuple(ids[i])))
    for i in order:
        p,a,b,q,existing=map(int,ids[i])
        if existing or scores[i]<threshold:continue
        affected={p,a,b}|({q} if q>=0 else set())
        if affected&locked:rejected['conflicting_event']+=1;continue
        if (p,a) not in result or len(outgoing.get(p,[]))!=1:rejected['not_unary_parent']+=1;continue
        if incoming.get(b,-1)!=q:rejected['incumbent_changed']+=1;continue
        if q>=0 and (len(outgoing.get(q,[]))!=1 or (q,b) not in result):rejected['protected_division']+=1;continue
        if len(changes)>=config.maximum_new_forks_per_movie:rejected['intervention_budget']+=1;continue
        if q>=0:del result[(q,b)]
        result[(p,b)]={'source_id':p,'target_id':b}
        locked.update(affected)
        changes.append({'parent':p,'existing_daughter':a,'new_daughter':b,'displaced_parent':q,
                        'event_score':float(scores[i]),'threshold':threshold})
    new=[result[k] for k in sorted(result)];newinc,newout=structure(nodes,new)
    for p,children in outgoing.items():
        if len(children)==2:require(newout.get(p)==children,'EXISTING_FORK_CHANGED')
    require(all(newinc[c]==r['parent'] for r in changes for c in (r['existing_daughter'],r['new_daughter'])),'TRANSACTION_INVALID')
    return new,{'accepted_new_forks':len(changes),'reassigned_children':sum(r['displaced_parent']>=0 for r in changes),
                'added_unassigned_children':sum(r['displaced_parent']<0 for r in changes),'changes':changes,'rejections':dict(rejected)}
